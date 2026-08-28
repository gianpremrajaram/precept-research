"""Offline tests for the RQ3a driver (DSE-064, DSE-065).

Three things are worth pinning here and they are all about honesty rather than plumbing. The corpus
digest must move when the corpus does, or a silently revised upstream upload changes a result
without changing its identity. The judge's call projection must match what the procedures actually
spend, because a pre-flight that under-reports is worse than none on a wall-clocked job. And
``VLLMJudge`` must distinguish a model that declines to answer (an abstention, recorded) from an
endpoint that is broken (an exception, loud) - the two look identical at the call site and mean
opposite things in the results table.

Nothing here touches the network or a corpus on disk beyond a hand-built fixture tree.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.logs import LogHandoffRecord, LogTraceRecord
from preceptx.experiments.cli import rq3a
from preceptx.experiments.rq3a import (
    RQ3aConfig,
    judge_all_at_once,
    judge_binary_search,
    judge_step_by_step,
    localisation_steps,
)
from preceptx.experiments.rq3a_run import (
    VLLMJudge,
    corpus_digest,
    corpus_paths,
    load_corpus,
    projected_judge_calls,
    run_rq3a,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig
from preceptx.serving.client import LLMClient, ServingConfig, ServingError

_FAST = RQ3aConfig(probe=ProbeConfig(n_repeats=1), n_boot=200)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


class _Encoder:
    """Deterministic content-hashed vectors; no torch, no download."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        rows = []
        for s in sentences:
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(4).tolist())
        return np.array(rows, dtype=np.float64)


def _featuriser(tmp_path: Path) -> Featuriser:
    return Featuriser(EncoderConfig(cache_dir=tmp_path / "emb"), encoder=_Encoder())


def _records(n_traces: int = 2, n_steps: int = 4) -> list[LogHandoffRecord]:
    names = ["planner", "coder"]
    return [
        LogHandoffRecord(
            corpus="traceelephant",
            trace_id=f"family/task{t}",
            step=i,
            agent_name=names[i % 2],
            receiver=names[(i + 1) % 2],
            is_handoff=True,
            observation=f"obs {t} {i}",
            message=("bad message" if i == 1 else f"fine message {i}"),
            trace_failed=True,
            annotations={"mistake_step": 1, "mistake_agent": "coder"},
        )
        for t in range(n_traces)
        for i in range(n_steps)
    ]


def _te_step(step_id: int, agent: str, prompt: str, reply: str) -> Any:
    return {
        "step_id": step_id,
        "agent_id": 1 if agent == "editor" else 2,
        "agent_name": agent,
        "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]},
        "output": {"choices": [{"message": {"content": reply, "tool_calls": []}}]},
        "tool_logs": [],
    }


def _write_corpus_root(root: Path) -> Path:
    """A root shaped like ``scripts/fetch_rq3a.sh`` leaves it, TraceElephant arm only."""
    d = root / "traceelephant" / "data" / "swe-agent-runs" / "task-1"
    d.mkdir(parents=True)
    (d / "step_records.json").write_text(
        json.dumps(
            [
                _te_step(1, "editor", "look", ""),
                _te_step(2, "bash", "run tests", "3 failed"),
                _te_step(3, "editor", "patch", "done"),
            ]
        )
    )
    (d / "trace_metadata.json").write_text(
        json.dumps(
            {
                "mistake_agent": "editor",
                "mistake_step": "2",
                "mistake_reason": "wrong file",
                "tests_status": {"FAIL_TO_PASS": {"success": [], "failure": ["t::a"]}},
            }
        )
    )
    return root


# ---------------------------------------------------------------------------------------------
# Corpus identity
# ---------------------------------------------------------------------------------------------


def test_the_corpus_digest_is_stable_across_reloads() -> None:
    assert corpus_digest(_records()) == corpus_digest(_records())


def test_the_corpus_digest_moves_when_the_corpus_does() -> None:
    """The whole point: a revised upstream upload must not reuse a frozen result's identity."""
    revised = _records()
    revised[3] = revised[3].model_copy(update={"message": "upstream edited this step"})
    assert corpus_digest(revised) != corpus_digest(_records())


def test_the_digest_is_independent_of_the_order_records_arrive_in() -> None:
    """Only TraceElephant's loader walks a sorted glob.

    Who&When follows parquet row order and MAST the JSON array's - deterministic for a given file,
    but not canonical across a rewrite of it. The digest is the identity stamped into every RQ3a
    manifest, so it has to be a function of content and not of file layout.
    """
    records = _records(n_traces=3, n_steps=4)
    shuffled = list(reversed(records))
    assert shuffled != records
    assert corpus_digest(shuffled) == corpus_digest(records)


def test_the_digest_sorts_trace_level_rows_that_carry_no_step() -> None:
    """``LogTraceRecord`` has no ``step``; sorting must not need a second code path for it."""
    rows = [
        LogTraceRecord(corpus="mast", trace_id=f"t{i}", system_name="s", trace_text=f"body {i}")
        for i in range(3)
    ]
    assert corpus_digest(list(reversed(rows))) == corpus_digest(rows)


def test_an_empty_corpus_is_an_error_not_a_digest() -> None:
    with pytest.raises(ConfigError, match="empty corpus"):
        corpus_digest([])


def test_corpus_paths_match_the_fetch_script_layout(tmp_path: Path) -> None:
    paths = corpus_paths(tmp_path)
    assert paths["traceelephant"] == tmp_path / "traceelephant" / "data"
    assert paths["who_and_when"] == tmp_path / "who_and_when"
    assert paths["mast"] == tmp_path / "mast" / "MAD_full_dataset.json"


def test_the_loader_reads_a_root_the_fetch_script_would_have_written(tmp_path: Path) -> None:
    records = load_corpus("traceelephant", _write_corpus_root(tmp_path))
    assert [r.agent_name for r in records] == ["editor", "bash", "editor"]
    assert [r.is_handoff for r in records] == [True, True, False]


# ---------------------------------------------------------------------------------------------
# The judge-call projection
# ---------------------------------------------------------------------------------------------


class _FixedJudge:
    """Answers one way to everything. The two settings are different worst cases, not one.

    ``answer=False`` is the worst case for step-by-step - it never short-circuits, so it pays *n* -
    but the *cheap* branch for binary search, which shrinks by ``mid + 1``. ``answer=True`` inverts
    both. A projection tested only against the all-no case would be untested on half its terms.
    """

    model_name = "stub"
    model_revision = "rev0"
    decoding = "greedy"

    def __init__(self, answer: bool) -> None:
        self.answer = answer

    def select_step(self, transcript: str, n_steps: int) -> int | None:
        return 0

    def contains_error(self, transcript: str) -> bool | None:
        return self.answer

    def is_error(self, transcript: str, step_text: str) -> bool | None:
        return self.answer


def _spend(steps: Any, judge: _FixedJudge) -> dict[str, int]:
    return {
        p.__name__: p(steps, judge).model_calls
        for p in (judge_all_at_once, judge_binary_search, judge_step_by_step)
    }


@pytest.mark.parametrize("answer", [True, False])
def test_the_projection_is_never_exceeded(answer: bool) -> None:
    """A pre-flight that under-reports is worse than none on a wall-clocked job."""
    records = _records(n_traces=3, n_steps=5)
    steps = localisation_steps(records, handoffs_only=True)
    spent = sum(_spend(steps, _FixedJudge(answer)).values())
    assert spent <= projected_judge_calls(records, handoffs_only=True)


def test_each_term_of_the_projection_is_a_real_worst_case() -> None:
    """And it must not be loose either, or the pre-flight over-reserves the node."""
    records = _records(n_traces=1, n_steps=5)
    steps = localisation_steps(records, handoffs_only=True)
    yes, no = _spend(steps, _FixedJudge(True)), _spend(steps, _FixedJudge(False))
    assert yes["judge_all_at_once"] == 1
    assert yes["judge_binary_search"] == math.ceil(math.log2(5))  # always-yes is the slow branch
    assert no["judge_step_by_step"] == 5  # never short-circuits, so it pays the whole trace
    assert 1 + yes["judge_binary_search"] + no["judge_step_by_step"] == projected_judge_calls(
        records, handoffs_only=True
    )


def test_the_projection_counts_only_the_steps_that_will_be_scored() -> None:
    records = _records(n_traces=1, n_steps=4)
    records[2] = records[2].model_copy(update={"is_handoff": False})
    assert projected_judge_calls(records, handoffs_only=True) < projected_judge_calls(
        records, handoffs_only=False
    )


# ---------------------------------------------------------------------------------------------
# VLLMJudge: abstention is decoded, failure is raised
# ---------------------------------------------------------------------------------------------


class _StubClient(LLMClient):
    """Returns a canned structured payload; raises instead when ``fail`` is set."""

    def __init__(self, payload: dict[str, Any], *, fail: bool = False) -> None:
        self._cfg = ServingConfig(model="stub-14b")
        self._payload = payload
        self._fail = fail
        self.calls = 0

    @property
    def config(self) -> ServingConfig:
        return self._cfg

    def structured(
        self,
        messages: Any,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        if self._fail:
            raise ServingError("endpoint down")
        return self._payload


def _judge(payload: dict[str, Any], *, fail: bool = False) -> VLLMJudge:
    return VLLMJudge(_StubClient(payload, fail=fail), revision="rev-abc")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({"step": 2}, 2), ({"step": -1}, None), ({"step": 99}, None), ({"step": "two"}, None)],
)
def test_an_unusable_index_is_an_abstention_not_a_pick(
    payload: dict[str, Any], expected: int | None
) -> None:
    assert _judge(payload).select_step("transcript", 5) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("yes", True), ("no", False), ("unsure", None), ("maybe", None)],
)
def test_unsure_is_a_decoded_answer_not_a_parse_failure(answer: str, expected: bool | None) -> None:
    judge = _judge({"answer": answer})
    assert judge.contains_error("segment") is expected
    assert judge.is_error("context", "step") is expected


def test_a_broken_endpoint_raises_rather_than_abstaining() -> None:
    """An abstention is a claim about the trace. An outage is not, and cannot be recorded as one."""
    judge = _judge({"step": 0}, fail=True)
    with pytest.raises(ServingError):
        judge.select_step("transcript", 3)


def test_the_judge_records_the_identity_the_replication_caveat_needs() -> None:
    judge = _judge({"step": 0})
    assert (judge.model_name, judge.model_revision) == ("stub-14b", "rev-abc")
    assert "temperature=0.0" in judge.decoding and "guided_json" in judge.decoding


# ---------------------------------------------------------------------------------------------
# The run and its manifest
# ---------------------------------------------------------------------------------------------


def test_a_judgeless_run_is_a_supported_mode_not_a_degraded_one(tmp_path: Path) -> None:
    """The offline arms carry the run; the judge rows are absent rather than silently zeroed."""
    root = _write_corpus_root(tmp_path)
    run = run_rq3a(
        "traceelephant", root, _featuriser(tmp_path), cfg=_FAST, with_mast=False, command=["x"]
    )
    methods = {m.method: m for m in run.result.methods}
    assert methods["schema_validity"].status == "ok"
    assert methods["mean_cosine"].status == "ok"
    assert not any(m.startswith("judge") for m in methods)
    assert run.manifest.judge_model is None
    # The transfer regime has no calibration to borrow, and says so rather than reading 0.0.
    assert methods["cpvi_transfer"].status != "ok"
    assert methods["cpvi_transfer"].reason


def test_the_manifest_identifies_the_corpus_it_actually_read(tmp_path: Path) -> None:
    root = _write_corpus_root(tmp_path)
    run = run_rq3a(
        "traceelephant", root, _featuriser(tmp_path), cfg=_FAST, with_mast=False, command=["x"]
    )
    assert run.manifest.corpus == "traceelephant"
    assert run.manifest.corpus_digest == corpus_digest(load_corpus("traceelephant", root))
    assert run.manifest.counts.traces == 1
    assert run.manifest.counts.handoffs == 2
    assert run.manifest.encoder_revision == EncoderConfig().revision
    # No MAST arm was run, so both its digest and its counts are absent rather than zeroed.
    assert run.manifest.mast_digest is None and run.manifest.mast_counts is None


# ---------------------------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------------------------


def test_dry_run_costs_the_judge_without_constructing_a_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRECEPTX_SERVING_SUBSTRATE", raising=False)
    root = _write_corpus_root(tmp_path)
    assert rq3a(["--root", str(root), "--judge", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "traces:           1" in out
    assert "handoffs:         2" in out
    expected = projected_judge_calls(load_corpus("traceelephant", root), handoffs_only=True)
    assert f"judge calls:      {expected}" in out


def test_a_judge_run_refuses_an_unlabelled_substrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model calls happened somewhere; an unlabelled artefact cannot say where, ever again."""
    monkeypatch.delenv("PRECEPTX_SERVING_SUBSTRATE", raising=False)
    root = _write_corpus_root(tmp_path)
    with pytest.raises(ConfigError, match="PRECEPTX_SERVING_SUBSTRATE"):
        rq3a(["--root", str(root), "--judge"])


def test_the_offline_run_needs_no_substrate_and_writes_its_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRECEPTX_SERVING_SUBSTRATE", raising=False)
    monkeypatch.setattr("preceptx.experiments.cli.Featuriser", lambda cfg: _featuriser(tmp_path))
    root = _write_corpus_root(tmp_path)
    out = tmp_path / "report"
    assert rq3a(["--root", str(root), "--no-mast", "--out", str(out)]) == 0
    assert (out / "rq3a.json").is_file()
    assert (out / "rq3a_localisation.csv").is_file()
    assert json.loads((out / "manifest.json").read_text())["corpus"] == "traceelephant"
