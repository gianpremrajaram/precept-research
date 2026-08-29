"""Integration: the four H6 arms run end to end on a small grid and the analysis lands (DSE-025).

Torch-free and offline - a mocked vLLM endpoint drives ``run_rq3b`` over a two-cell grid with a
stub encoder and a hand-built calibration, so the check is that the *wiring* closes: four arms
reach four datasets, the gate is constructed per arm from an imported threshold, and the analysis
emits its verdict, tables and figures. The full run is gated on compute.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from numpy.typing import NDArray

from preceptx.analysis.stats import build_provenance
from preceptx.config import ConfigError, ModelConfig
from preceptx.experiments.rq3b import GATE_MODES, RQ3bConfig, build_gate, run_rq3b, write_rq3b
from preceptx.experiments.rq3b import rq3b_sweeps as _arms
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for
from preceptx.gate.calibration import CalibrationReport, StatisticCalibration
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


class _StubEncoder:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        out = np.zeros((len(sentences), 16), dtype=np.float64)
        for i, s in enumerate(sentences):
            seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")
            out[i] = np.random.default_rng(seed).standard_normal(16)
        return out


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


_ACTION_CYCLE = itertools.cycle(("E", "E", "W"))


def _east_script(request: httpx.Request) -> httpx.Response:
    if b"structured_outputs" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": next(_ACTION_CYCLE)})))
    return httpx.Response(200, json=_completion("push the load east toward the goal"))


def _base(**over: object) -> SweepConfig:
    fields: dict[str, object] = {
        "conditions": ["C0"],
        "serialisations": ["numeric"],
        "difficulties": ["hard"],
        "seeds": [1, 2],
        "model": ModelConfig(name="m", revision="rev", tier="8b"),
        "max_steps": 6,
        "concurrency": 1,
    }
    fields.update(over)
    return SweepConfig(**fields)  # type: ignore[arg-type]


def _report(key: str = "cosine", *, firing_rate: float = 0.25) -> CalibrationReport:
    """A hand-built calibration standing in for one persisted by ``gate.calibration``."""
    return CalibrationReport(
        dataset_hash="calib",
        provenance=build_provenance(EncoderConfig(), ProbeConfig()),
        n=64,
        n_bins=5,
        ece_reliable=False,
        statistics=[
            StatisticCalibration(
                key=key,
                threshold=0.0,
                orientation=1.0,
                firing_rate=firing_rate,
                auroc=0.7,
                ece=None,
                n_classes=2,
                reliability=[],
            )
        ],
    )


@respx.mock
def test_the_four_arms_run_and_h6_lands(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_east_script)
    base = _base()
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))

    result = run_rq3b(
        base,
        client,
        root=tmp_path,
        report=_report(),
        # The statistic re-fits on the calibration set, and here that set is the ungated arm's own
        # dataset only because the fixture has nothing else; a real run points this at the pilot's.
        calibration_records=[],
        featuriser=feat,
        statistic_key="cosine",
        cfg=RQ3bConfig(n_boot=1000),
    )

    assert {m.mode for m in result.modes} == set(GATE_MODES)
    # Four arms, four directories: the property the whole design rests on.
    assert len({m.dataset_hash for m in result.modes}) == 4
    for mode, sweep in _arms(base, statistic_key="cosine").items():
        written = tmp_path / dataset_hash_for(sweep)
        assert written.is_dir(), f"{mode} wrote no dataset"
    assert all(m.n_episodes == 2 for m in result.modes)
    assert result.verdict  # a verdict is always produced, including the untestable one
    assert len(result.contrasts) == 6  # two outcomes x three controls

    out = write_rq3b(result, tmp_path / "rq3b")
    for name in ("rq3b.json", "rq3b_modes.csv", "rq3b_contrasts.csv", "verdict.md"):
        assert (out / name).is_file(), name


def test_a_gate_arm_without_its_threshold_fails_loud(tmp_path: Path) -> None:
    # Fails open at construction rather than silently: a gate that cannot score would be recorded
    # as a gated arm while behaving like the ungated one, which is the worst of both.
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    arms = _arms(_base(), statistic_key="cosine")
    with pytest.raises(ConfigError, match="no statistic 'cosine'"):
        build_gate(
            "active",
            arms["active"],
            report=_report(key="fail"),
            calibration_records=[],
            featuriser=feat,
        )


def test_only_the_active_arm_carries_a_scoring_statistic(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    arms = _arms(_base(), statistic_key="cosine")
    gates = {
        mode: build_gate(mode, sweep, report=_report(), calibration_records=[], featuriser=feat)
        for mode, sweep in arms.items()
    }
    assert gates["off"] is None
    active = gates["active"]
    assert active is not None and active._stat is not None
    for control in ("matched_random", "random_trigger"):
        gate = gates[control]
        # A control that held a fitted statistic would look, to a reader of the artefact, as though
        # it had seen the score it is defined to be blind to.
        assert gate is not None and gate._stat is None
        assert gate._cal is not None  # matched_random still needs the measured firing rate
