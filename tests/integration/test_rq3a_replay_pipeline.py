"""The replay labeller end to end on a fixture corpus, against a stubbed backend (DSE-042).

Starts from corpus JSON on disk rather than from hand-built records, so the loader's output shape
and the labeller's input shape are checked against each other rather than against a shared
assumption. No network, no environment, no served model: replay at scale is a budgeted experiment,
but the path from corpus to labelled steps must be exercisable without one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from preceptx.experiments.rq3a_load import load_traceelephant
from preceptx.experiments.rq3a_replay import (
    ReplayBackend,
    ReplayBudget,
    ReplayOutcome,
    ReplayPlan,
    ReplayStep,
    label_by_replay,
    manifest_metrics,
    project,
    render_projection,
    replay_steps,
    stratified_sample,
    trace_success_labels,
)


class FlakyBackend(ReplayBackend):
    """Deterministic stand-in for a real environment: later steps recover, early ones do not.

    Encodes the shape replay exists to detect - an outcome that depends on *which* step was
    substituted - and disagrees on one step so the agreement floor has something to catch.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, int, int]] = []

    def replay(self, step: ReplayStep, attempt: int) -> ReplayOutcome:
        self.seen.append((step.trace_id, step.step, attempt))
        if step.step == 2:  # the contested step: 2 of 3 replays fail
            return ReplayOutcome(failed=attempt != 1, model_calls=1)
        return ReplayOutcome(failed=step.step < 2, model_calls=1)


_FAMILY = "swe-agent-runs-swe-bench"


def _te_step(step_id: int, agent: str, prompt: str, reply: str) -> dict[str, Any]:
    """One TraceElephant step in the corpus's own layout (mirrors the E9-verified fixture)."""
    return {
        "step_id": step_id,
        "agent_id": 1 if agent == "editor" else 2,
        "agent_name": agent,
        "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]},
        "output": {"choices": [{"message": {"content": reply, "tool_calls": []}}]},
        "tool_logs": [],
    }


def _write_corpus(root: Path, task: str, *, failing: bool) -> None:
    d = root / "data" / _FAMILY / task
    d.mkdir(parents=True)
    (d / "step_records.json").write_text(
        json.dumps(
            [
                _te_step(1, "editor", "look at /testbed", "reading"),
                _te_step(2, "bash", "run the tests", "3 failed"),
                _te_step(3, "editor", "patch it", "patched"),
                _te_step(4, "bash", "run the tests", "ok"),
            ]
        )
    )
    (d / "trace_metadata.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "system_name": "swe-agent",
                "mistake_agent": "editor",
                "mistake_step": "2",
                "mistake_reason": "wrong file",
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["t::a"] if failing else []},
                    "PASS_TO_PASS": {"success": ["t::b"], "failure": []},
                },
            }
        )
    )


def test_corpus_to_labelled_steps(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "psf__requests-1724", failing=True)
    _write_corpus(tmp_path, "psf__requests-2317", failing=False)
    records = load_traceelephant(tmp_path / "data")

    # The cheap label survives regardless of whether replay is affordable.
    assert trace_success_labels(records) == {
        f"{_FAMILY}/psf__requests-1724": True,
        f"{_FAMILY}/psf__requests-2317": False,
    }

    # Plan and forecast before anything is executed.
    handoffs = [r for r in records if r.is_handoff]
    sample = stratified_sample(handoffs, 4, seed=11)
    plan = ReplayPlan(replays_per_step=3, calls_per_replay=(1, 2), seconds_per_call=2.4)
    budget = ReplayBudget(max_model_calls=100)
    projection = project(len(sample.steps), plan, budget)

    assert projection.permitted is True
    assert projection.projected_calls_min == 12
    assert projection.projected_calls_max == 24
    assert "PERMITTED" in render_projection(projection, sample)
    assert sample.allocated == {"True": 2, "False": 2}  # the balance is preserved

    # Execute the sampled steps of one trace.
    trace_id = f"{_FAMILY}/psf__requests-1724"
    trace = replay_steps([r for r in records if r.trace_id == trace_id])
    step_ids = sorted(s for t, s in sample.steps if t == trace_id)
    backend = FlakyBackend()
    labelling = label_by_replay(trace, step_ids, 3, budget, backend=backend, agreement_floor=0.75)

    assert len(labelling.labels) == len(step_ids)
    assert labelling.stopped_on_budget is False
    assert labelling.model_calls == 3 * len(step_ids)
    assert {s for _, s, _ in backend.seen} == set(step_ids)

    by_step = {x.step: x for x in labelling.labels}
    if 2 in by_step:  # the contested step arrives labelled AND flagged, never dropped
        contested = by_step[2]
        assert contested.outcome_failed is True
        assert contested.agreement < 0.75
        assert contested.below_floor is True

    # The whole run reduces to one JSON-serialisable manifest block.
    block = manifest_metrics(projection, sample, plan, labelling)
    json.dumps(block)
    assert block["replay"]["sampling"]["seed"] == 11
    assert block["replay"]["realised"]["model_calls"] == labelling.model_calls


def test_a_cap_below_the_forecast_refuses_before_the_backend_is_touched(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "psf__requests-1724", failing=True)
    records = load_traceelephant(tmp_path / "data")
    sample = stratified_sample(records, 4, seed=0)
    projection = project(
        len(sample.steps), ReplayPlan(replays_per_step=10), ReplayBudget(max_model_calls=5)
    )

    assert projection.permitted is False
    assert "no replay was executed" in projection.refusal
    assert "REFUSED" in render_projection(projection, sample)
