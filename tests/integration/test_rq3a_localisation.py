"""RQ3a end to end: corpus JSON on disk, replay labels, every method, one comparison table.

Chains the three RQ3a tickets that now exist - the loader (DSE-041), the counterfactual-replay
labeller (DSE-042) and this analysis (DSE-024) - so their record shapes are checked against each
other rather than against a shared assumption. The judge and the replay backend are stubs and the
encoder is deterministic: the point is that the path from corpus to comparison table runs with no
served model, because at corpus scale it is a budgeted experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from preceptx.experiments.rq3a import (
    JudgeBackend,
    RQ3aConfig,
    analyse_rq3a,
    manifest_metrics,
    replay_labels,
    results_table,
    write_rq3a,
)
from preceptx.experiments.rq3a_load import load_traceelephant
from preceptx.experiments.rq3a_replay import (
    ReplayBackend,
    ReplayBudget,
    ReplayOutcome,
    ReplayStep,
    label_by_replay,
    replay_steps,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig

_FAMILY = "swe-agent-runs-swe-bench"
_MISTAKE_STEP = 2


class _StepSensitiveBackend(ReplayBackend):
    """Substituting the annotated mistake step changes the outcome; substituting others does not."""

    def replay(self, step: ReplayStep, attempt: int) -> ReplayOutcome:
        return ReplayOutcome(failed=step.step != _MISTAKE_STEP, model_calls=1)


class _KeywordJudge(JudgeBackend):
    """Stands in for the served open-weight annotator; keys off the planted marker."""

    model_name = "stub-open-weight-8b"
    model_revision = "rev0"
    decoding = "greedy, temperature=0, seed=0"

    def select_step(self, transcript: str, n_steps: int) -> int | None:
        for i, block in enumerate(transcript.split("\n\n")):
            if "wrong file" in block:
                return i
        return None

    def contains_error(self, transcript: str) -> bool | None:
        return "wrong file" in transcript

    def is_error(self, transcript: str, step_text: str) -> bool | None:
        return "wrong file" in step_text


class _Encoder:
    """Deterministic content-hashed embeddings; dim0 marks the planted step."""

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
            seed = abs(hash(s)) % (2**32)
            rows.append(
                [1.0 if "wrong file" in s else 0.0, *np.random.default_rng(seed).normal(size=3)]
            )
        return np.array(rows, dtype=np.float64)


def _te_step(step_id: int, agent: str, prompt: str, reply: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "agent_id": abs(hash(agent)) % 8,
        "agent_name": agent,
        "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]},
        "output": {"choices": [{"message": {"content": reply, "tool_calls": []}}]},
        "tool_logs": [],
    }


def _write_corpus(root: Path, task: str) -> None:
    d = root / "data" / _FAMILY / task
    d.mkdir(parents=True)
    (d / "step_records.json").write_text(
        json.dumps(
            [
                # Alternating components, so steps 1-3 are inter-agent handoffs and step 4, the
                # last, is not. Step 2 carries both the planted marker and the annotation.
                _te_step(1, "planner", "look at /testbed", "editor: fix the failing test"),
                _te_step(2, "editor", "fix the bug", "edited the wrong file entirely"),
                _te_step(3, "bash", "run the tests", "still 3 failed"),
                _te_step(4, "editor", "try again", "no change"),
            ]
        )
    )
    (d / "trace_metadata.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "system_name": "swe-agent",
                "mistake_agent": "editor",
                "mistake_step": str(_MISTAKE_STEP),
                "mistake_reason": "wrong file",
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["t::a"]},
                    "PASS_TO_PASS": {"success": ["t::b"], "failure": []},
                },
            }
        )
    )


def test_corpus_to_comparison_table(tmp_path: Path) -> None:
    for task in ("psf__requests-1724", "psf__requests-2317", "psf__requests-3050"):
        _write_corpus(tmp_path, task)
    records = load_traceelephant(tmp_path / "data")
    assert {r.corpus for r in records} == {"traceelephant"}

    # DSE-042 supplies the refit arm's outcome variable; the annotation never does.
    budget = ReplayBudget(max_model_calls=200)
    labellings = [
        label_by_replay(
            replay_steps([r for r in records if r.trace_id == trace_id]),
            sorted({r.step for r in records if r.trace_id == trace_id and r.is_handoff}),
            3,
            budget,
            backend=_StepSensitiveBackend(),
        )
        for trace_id in sorted({r.trace_id for r in records})
    ]
    labels = replay_labels(labellings)
    assert labels and set(labels.values()) == {True, False}  # two classes, so refit can fit

    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "emb"), encoder=_Encoder())
    result = analyse_rq3a(
        records,
        feat,
        cfg=RQ3aConfig(probe=ProbeConfig(n_repeats=1), n_boot=200, audit_sample=5),
        judge=_KeywordJudge(),
        labels=labels,
    )

    by_method = {m.method: m for m in result.methods}
    assert by_method["cpvi_refit"].status == "ok"  # replay labels present and two-class
    assert by_method["cpvi_transfer"].status == "unavailable"  # no frozen simulator probe yet
    assert by_method["judge_step_by_step"].step_accuracy == 1.0  # the planted marker is found
    assert by_method["schema_validity"].status in {"ok", "not_applicable"}
    assert result.agreement is not None and result.agreement.n_sampled == 3

    table = results_table(result)
    assert len(table) == len(result.methods)
    assert set(table["status"]) <= {"ok", "unavailable", "not_applicable"}
    assert table.loc[table["method"] == "cpvi_transfer", "reason"].notna().all()

    out = write_rq3a(result, tmp_path / "rq3a")
    assert json.loads((out / "rq3a.json").read_text())["tie_policy"] == result.tie_policy
    json.dumps(manifest_metrics(result))  # the manifest block stays JSON-serialisable
