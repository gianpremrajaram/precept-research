"""One short episode per gate arm against a stub LLM (DSE-018).

Covers what the unit tests cannot: that the gate hook sits at the right seam in the LangGraph loop,
that a block actually re-prompts A with the DSE-045 feedback and delivers the *retried* message to
B, and - the load-bearing one - that ``gate=None`` leaves the ungated loop byte-identical, so RQ1's
frozen dataset semantics do not move.

The 8B-tier leg of the ticket's integration requirement (AC 5) is deferred: the GPU is allocated to
the RQ1 driver. This file is the stub-client half.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from numpy.typing import NDArray

from preceptx.agents.graph import EpisodeRunner
from preceptx.agents.prompts import GATE_FEEDBACK
from preceptx.config import ExperimentConfig, ModelConfig
from preceptx.data.schema import HandoffRecord
from preceptx.gate.calibration import StatisticCalibration
from preceptx.gate.integration import GateConfig, GateMode, RuntimeGate
from preceptx.gate.statistics import CosineStatistic
from preceptx.measure.featuriser import EncoderConfig, Featuriser

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"
STEPS = 4
FIRST, RETRIED = "push the load east", "push the load north by 0.4 then east"


class _ConstantEncoder:
    """Every text maps to the same unit vector, so ``CosineStatistic`` scores exactly 1.0.

    That makes the active arm's firing a property of the THRESHOLD alone, which is what these tests
    want to vary; the score itself is exercised against real text in the unit tier.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        return np.tile(np.array([1.0, 0.0]), (len(sentences), 1))


def _client() -> object:
    from preceptx.serving.client import LLMClient, ServingConfig

    return LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))


def _cell() -> ExperimentConfig:
    return ExperimentConfig(
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        seed=0,
    )


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "stop": "stop"}
        ],
    }


def _handler(request: httpx.Request) -> httpx.Response:
    """B always WAITs (the episode runs to budget); A's retry is distinguishable from its first."""
    if b"structured_outputs" in request.content:  # B's action call carries the schema
        return httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    retry = GATE_FEEDBACK.split(".")[0].encode() in request.content
    return httpx.Response(200, json=_completion(RETRIED if retry else FIRST))


def _cal(threshold: float, *, firing_rate: float = 0.5) -> StatisticCalibration:
    return StatisticCalibration(
        key="cosine",
        threshold=threshold,
        orientation=1.0,
        firing_rate=firing_rate,
        auroc=0.7,
        ece=0.1,
        n_classes=2,
        reliability=[],
    )


def _gate(
    tmp_path: Path,
    mode: GateMode,
    *,
    threshold: float = 0.5,
    firing_rate: float = 0.5,
    max_retries: int = 1,
    calibrated: bool = True,
) -> RuntimeGate:
    cal = _cal(threshold, firing_rate=firing_rate) if calibrated else None
    return RuntimeGate(
        GateConfig(mode=mode, statistic_key="cosine", max_retries=max_retries),
        statistic=CosineStatistic(),
        calibration=cal,
        featuriser=Featuriser(
            EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_ConstantEncoder()
        ),
    )


def _run(gate: RuntimeGate | None) -> list[HandoffRecord]:
    respx.post(CHAT).mock(side_effect=_handler)
    return EpisodeRunner(_client(), max_steps=STEPS, gate=gate).run_episode(_cell(), "ep")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------


@respx.mock
def test_no_gate_leaves_the_loop_untouched() -> None:
    records = _run(None)
    assert len(records) == STEPS
    assert all(r.message_delivered == FIRST for r in records)  # A was never re-prompted
    assert all(
        (r.gate_blocked, r.gate_retries, r.message_blocked) == (False, 0, None) for r in records
    )


@respx.mock
def test_off_and_fail_open_are_indistinguishable_from_no_gate(tmp_path: Path) -> None:
    # The frozen-dataset guarantee: three wirings that should not touch the data, and do not.
    ungated = _run(None)
    off = _run(_gate(tmp_path, "off"))
    open_ = _run(_gate(tmp_path, "active", calibrated=False))  # no threshold -> fail open
    assert [r.model_dump() for r in off] == [r.model_dump() for r in ungated]
    assert [r.model_dump() for r in open_] == [r.model_dump() for r in ungated]


@respx.mock
def test_active_gate_blocks_re_prompts_and_delivers_the_retried_message(tmp_path: Path) -> None:
    records = _run(_gate(tmp_path, "active", threshold=0.5))  # constant score 1.0 >= 0.5
    assert len(records) == STEPS
    for r in records:
        assert r.gate_blocked and r.gate_retries == 1
        assert r.message_blocked == FIRST  # what B would have seen with no gate
        assert r.message_delivered == RETRIED  # what B saw with one
        assert r.message_raw == RETRIED  # the retry replaces A's turn, it does not append to it


@respx.mock
def test_active_gate_passes_a_handoff_above_the_threshold(tmp_path: Path) -> None:
    records = _run(_gate(tmp_path, "active", threshold=2.0))  # unreachable: 1.0 < 2.0
    assert all(not r.gate_blocked and r.message_delivered == FIRST for r in records)


@respx.mock
def test_zero_retries_records_the_block_without_re_prompting(tmp_path: Path) -> None:
    records = _run(_gate(tmp_path, "active", threshold=0.5, max_retries=0))
    for r in records:
        assert r.gate_blocked and r.gate_retries == 0
        assert r.message_blocked == r.message_delivered == FIRST  # bounded at zero: A proceeds


@respx.mock
def test_exhausted_retries_still_proceed(tmp_path: Path) -> None:
    # The constant score means the retry is blocked too; the loop must deliver it anyway, not spin.
    records = _run(_gate(tmp_path, "active", threshold=0.5, max_retries=2))
    for r in records:
        assert r.gate_blocked and r.gate_retries == 2
        assert r.message_blocked == FIRST and r.message_delivered == RETRIED


@respx.mock
@pytest.mark.parametrize("mode", ["matched_random", "random_trigger"])
def test_controls_block_without_scoring(tmp_path: Path, mode: GateMode) -> None:
    # firing_rate/random_rate default to a firing arm; the controls fire on the SAME constant-score
    # handoffs the active arm would, but for reasons that have nothing to do with the score.
    gate = (
        _gate(tmp_path, mode, firing_rate=1.0)
        if mode == "matched_random"
        else RuntimeGate(GateConfig(mode="random_trigger", random_rate=1.0))
    )
    records = _run(gate)
    for r in records:
        assert r.gate_blocked and r.gate_retries == 1
        assert r.message_blocked == FIRST and r.message_delivered == RETRIED
