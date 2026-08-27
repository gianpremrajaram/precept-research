"""Offline tests for the counterfactual-replay labeller (DSE-042).

No backend is ever real here. The point of the module is that the labelling logic, the budget guard
and the dry-run projection are all exercisable without an environment or a served model, so these
tests are what prove that separation holds rather than merely being asserted in a docstring.
"""

from __future__ import annotations

import inspect
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from preceptx.data.logs import LogHandoffRecord
from preceptx.experiments.rq3a_replay import (
    ReplayBackend,
    ReplayBudget,
    ReplayError,
    ReplayOutcome,
    ReplayPlan,
    ReplayProjection,
    ReplayStep,
    label_by_replay,
    manifest_metrics,
    project,
    render_projection,
    replay_steps,
    stratified_sample,
    trace_success_labels,
)


class ScriptedBackend(ReplayBackend):
    """Replays a fixed list of outcomes, counting how many times it was actually invoked."""

    def __init__(self, outcomes: list[bool], calls_each: int = 1) -> None:
        self.outcomes = outcomes
        self.calls_each = calls_each
        self.invocations = 0

    def replay(self, step: ReplayStep, attempt: int) -> ReplayOutcome:
        failed = self.outcomes[self.invocations % len(self.outcomes)]
        self.invocations += 1
        return ReplayOutcome(failed=failed, model_calls=self.calls_each)


def _record(trace_id: str, step: int, *, failed: bool | None = None) -> LogHandoffRecord:
    return LogHandoffRecord(
        corpus="traceelephant",
        trace_id=trace_id,
        step=step,
        agent_name="planner",
        receiver="coder",
        is_handoff=True,
        observation=f"obs {step}",
        message=f"msg {step}",
        trace_failed=failed,
        annotations={"mistake_agent": "planner", "mistake_step": step},
    )


def _trace(n: int) -> list[ReplayStep]:
    return replay_steps([_record("t1", i) for i in range(n)])


# --------------------------------------------------------------------------------------------
# The structural guards - the labeller cannot reach the annotations, the dry run cannot call out
# --------------------------------------------------------------------------------------------


def test_replay_step_has_no_annotation_field() -> None:
    assert "annotations" not in ReplayStep.model_fields
    assert set(ReplayStep.model_fields) == {"trace_id", "step", "observation", "message"}


def test_replay_steps_drops_annotations_and_outcome() -> None:
    steps = replay_steps([_record("t1", 0, failed=True)])
    assert not hasattr(steps[0], "annotations")
    assert not hasattr(steps[0], "trace_failed")


def test_replay_step_refuses_an_annotation_smuggled_in_as_an_extra() -> None:
    with pytest.raises(ValueError):
        ReplayStep(trace_id="t", step=0, observation="o", message="m", annotations={"a": 1})


def test_labeller_consumes_replay_steps_not_log_records() -> None:
    """The no-annotation discipline is in the signature, mirroring ``prospective_twin``'s no-Y."""
    params = inspect.signature(label_by_replay).parameters
    assert list(params)[:4] == ["trace", "step_ids", "n_replays", "budget"]
    assert "ReplayStep" in str(params["trace"].annotation)


def test_project_takes_no_backend_so_a_dry_run_cannot_issue_a_call() -> None:
    assert "backend" not in inspect.signature(project).parameters


# --------------------------------------------------------------------------------------------
# Majority voting and the agreement floor
# --------------------------------------------------------------------------------------------


def test_majority_vote_over_replays() -> None:
    backend = ScriptedBackend([True, True, True, False, False])
    out = label_by_replay(_trace(1), [0], 5, ReplayBudget(max_model_calls=100), backend=backend)
    (label,) = out.labels
    assert label.outcome_failed is True
    assert label.agreement == pytest.approx(0.6)
    assert label.n_replays == 5
    assert label.below_floor is False


def test_a_tie_reads_false_and_arrives_flagged_never_resolved_silently() -> None:
    backend = ScriptedBackend([True, True, False, False])
    out = label_by_replay(_trace(1), [0], 4, ReplayBudget(max_model_calls=100), backend=backend)
    (label,) = out.labels
    assert label.outcome_failed is False
    assert label.agreement == pytest.approx(0.5)
    assert label.below_floor is True


def test_steps_below_the_floor_are_flagged_not_dropped() -> None:
    backend = ScriptedBackend([True, False])
    out = label_by_replay(
        _trace(3), [0, 1, 2], 2, ReplayBudget(max_model_calls=100), backend=backend
    )
    assert len(out.labels) == 3  # the whole point: nothing disappears
    assert all(x.below_floor for x in out.labels)


def test_the_agreement_floor_is_configurable() -> None:
    backend = ScriptedBackend([True, True, False])
    strict = label_by_replay(
        _trace(1), [0], 3, ReplayBudget(max_model_calls=99), backend=backend, agreement_floor=0.9
    )
    backend.invocations = 0
    lax = label_by_replay(
        _trace(1), [0], 3, ReplayBudget(max_model_calls=99), backend=backend, agreement_floor=0.5
    )
    assert strict.labels[0].below_floor is True
    assert lax.labels[0].below_floor is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_replays": 0}, "n_replays"),
        ({"agreement_floor": 0.0}, "agreement_floor"),
        ({"agreement_floor": 1.5}, "agreement_floor"),
    ],
)
def test_bad_labelling_arguments_fail_loud(kwargs: dict[str, float], match: str) -> None:
    args = {"n_replays": 3, "agreement_floor": 0.6, **kwargs}
    with pytest.raises(ReplayError, match=match):
        label_by_replay(
            _trace(1),
            [0],
            int(args["n_replays"]),
            ReplayBudget(max_model_calls=10),
            backend=ScriptedBackend([True]),
            agreement_floor=args["agreement_floor"],
        )


def test_an_unknown_step_id_fails_loud_rather_than_being_skipped() -> None:
    with pytest.raises(ReplayError, match="not present in the trace"):
        label_by_replay(
            _trace(2), [0, 7], 1, ReplayBudget(max_model_calls=10), backend=ScriptedBackend([True])
        )


# --------------------------------------------------------------------------------------------
# The budget guard - enforced at the call site, not on the forecast
# --------------------------------------------------------------------------------------------


def test_the_cap_stops_the_backend_being_called_at_all_once_reached() -> None:
    backend = ScriptedBackend([True])
    out = label_by_replay(
        _trace(4), [0, 1, 2, 3], 3, ReplayBudget(max_model_calls=4), backend=backend
    )
    assert backend.invocations == 4  # not 12: the guard sits before the call, not after
    assert out.model_calls == 4
    assert out.stopped_on_budget is True


def test_steps_never_reached_are_recorded_rather_than_omitted() -> None:
    out = label_by_replay(
        _trace(3), [0, 1, 2], 2, ReplayBudget(max_model_calls=2), backend=ScriptedBackend([True])
    )
    assert len(out.labels) == 3
    assert out.labels[0].outcome_failed is True
    unreached = out.labels[-1]
    assert unreached.outcome_failed is None
    assert unreached.n_replays == 0
    assert unreached.budget_exhausted is True


def test_a_multi_call_replay_overshoots_by_at_most_one_replay() -> None:
    """A replay that costs 3 calls cannot be un-sent once started; the cap holds to within one."""
    backend = ScriptedBackend([True], calls_each=3)
    out = label_by_replay(
        _trace(5), list(range(5)), 1, ReplayBudget(max_model_calls=5), backend=backend
    )
    assert out.model_calls == 6  # 5 was crossed by the second replay, which was already in flight
    assert out.model_calls <= 5 + 3


def test_a_partially_replayed_step_is_marked_exhausted_but_still_labelled() -> None:
    backend = ScriptedBackend([True, True, False])
    out = label_by_replay(_trace(1), [0], 3, ReplayBudget(max_model_calls=2), backend=backend)
    (label,) = out.labels
    assert label.n_replays == 2
    assert label.budget_exhausted is True
    assert label.outcome_failed is True


# --------------------------------------------------------------------------------------------
# The dry-run projection
# --------------------------------------------------------------------------------------------


def test_projection_reports_a_band_when_a_replay_may_cost_several_calls() -> None:
    p = project(
        300,
        ReplayPlan(replays_per_step=5, calls_per_replay=(1, 3)),
        ReplayBudget(max_model_calls=5000),
    )
    assert (p.projected_calls_min, p.projected_calls_max) == (1500, 4500)
    assert p.permitted is True
    assert p.refusal == ""


def test_projection_refuses_on_the_minimum_not_the_maximum() -> None:
    budget = ReplayBudget(max_model_calls=5000)
    plan = ReplayPlan(replays_per_step=5, calls_per_replay=(1, 3))
    # 1200 x 5 = 6000 minimum: even the cheapest execution overruns, so refuse before sending.
    refused = project(1200, plan, budget)
    assert refused.permitted is False
    assert "6000" in refused.refusal and "5000" in refused.refusal
    # 900 x 5 = 4500 minimum but 13500 maximum: permitted, because the guard stops the run live.
    assert project(900, plan, budget).permitted is True


def test_gpu_seconds_is_advisory_and_absent_without_a_calibration() -> None:
    budget = ReplayBudget(max_model_calls=10_000)
    assert project(100, ReplayPlan(), budget).estimated_gpu_seconds is None
    calibrated = project(
        100,
        ReplayPlan(replays_per_step=5, seconds_per_call=2.4, calibration_source="smoke 1580259"),
        budget,
    )
    assert calibrated.estimated_gpu_seconds == pytest.approx(1200.0)
    assert calibrated.calibration_source == "smoke 1580259"


def test_projection_carries_no_currency_field() -> None:
    """Budgets are calls and seconds. A monetary figure would be invented, not measured."""
    fields = set(ReplayProjection.model_fields)
    assert not fields & {"cost", "usd", "dollars", "price", "spend"}


@pytest.mark.parametrize("band", [(0, 1), (3, 2)])
def test_an_impossible_calls_per_replay_band_fails_loud(band: tuple[int, int]) -> None:
    with pytest.raises(ReplayError, match="calls_per_replay"):
        project(10, ReplayPlan(calls_per_replay=band), ReplayBudget(max_model_calls=10))


def test_the_dry_run_report_names_the_decision_and_the_cap() -> None:
    records = [_record("t1", i, failed=i % 2 == 0) for i in range(10)]
    sample = stratified_sample(records, 4, seed=0)
    text = render_projection(project(4, ReplayPlan(), ReplayBudget(max_model_calls=100)), sample)
    assert "PERMITTED" in text
    assert "Hard model-call cap:" in text
    assert "advisory" in text


# --------------------------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------------------------


def test_sampling_preserves_the_outcome_balance() -> None:
    records = [_record("t1", i, failed=True) for i in range(80)]
    records += [_record("t2", i, failed=False) for i in range(20)]
    sample = stratified_sample(records, 10, seed=7)
    assert sample.allocated == {"True": 8, "False": 2}
    assert len(sample.steps) == 10


def test_none_is_its_own_stratum() -> None:
    """176 of TraceElephant's 220 traces have no annotation-free outcome; folding them reweights."""
    records = [_record("t1", i, failed=None) for i in range(60)]
    records += [_record("t2", i, failed=True) for i in range(40)]
    sample = stratified_sample(records, 10, seed=1)
    assert sample.allocated == {"None": 6, "True": 4}


def test_sampling_is_deterministic_for_a_seed_and_moves_with_it() -> None:
    records = [_record("t1", i, failed=i % 3 == 0) for i in range(50)]
    a = stratified_sample(records, 12, seed=3)
    b = stratified_sample(records, 12, seed=3)
    c = stratified_sample(records, 12, seed=4)
    assert a.steps == b.steps
    assert a.steps != c.steps


def test_the_sampling_rule_travels_with_the_sample() -> None:
    sample = stratified_sample([_record("t1", 0)], 1, seed=0)
    assert "proportional allocation" in sample.rule
    assert "largest-remainder" in sample.rule
    assert sample.seed == 0


def test_sampling_an_empty_corpus_fails_loud() -> None:
    with pytest.raises(ReplayError, match="no records"):
        stratified_sample([], 5, seed=0)


@given(n=st.integers(min_value=0, max_value=60), seed=st.integers(min_value=0, max_value=2**31))
def test_a_sample_is_exactly_n_distinct_steps(n: int, seed: int) -> None:
    records = [_record(f"t{i % 4}", i, failed=(i % 3 == 0) or None) for i in range(40)]
    sample = stratified_sample(records, n, seed=seed)
    assert len(sample.steps) == min(n, len(records))
    assert len(set(sample.steps)) == len(sample.steps)
    assert sum(sample.allocated.values()) == len(sample.steps)


# --------------------------------------------------------------------------------------------
# The cheap label and the manifest block
# --------------------------------------------------------------------------------------------


def test_trace_success_is_computed_for_every_trace_including_the_unknowns() -> None:
    records = [
        _record("a", 0, failed=True),
        _record("b", 0, failed=None),
        _record("c", 0, failed=False),
    ]
    assert trace_success_labels(records) == {"a": True, "b": None, "c": False}


def test_manifest_block_carries_forecast_sampling_and_realised_spend() -> None:
    records = [_record("t1", i, failed=True) for i in range(6)]
    sample = stratified_sample(records, 3, seed=2)
    plan = ReplayPlan(replays_per_step=2)
    projection = project(3, plan, ReplayBudget(max_model_calls=20))
    labelling = label_by_replay(
        _trace(3), [0, 1, 2], 2, ReplayBudget(max_model_calls=20), backend=ScriptedBackend([True])
    )
    block = manifest_metrics(projection, sample, plan, labelling)["replay"]

    assert set(block) == {"plan", "projection", "sampling", "realised"}
    assert block["sampling"]["seed"] == 2
    assert block["sampling"]["rule"] == sample.rule
    assert block["realised"]["model_calls"] == 6
    assert block["realised"]["stopped_on_budget"] is False
    json.dumps({"replay": block})  # must survive the manifest's JSON round trip


def test_manifest_block_omits_realised_spend_before_a_run() -> None:
    sample = stratified_sample([_record("t1", 0)], 1, seed=0)
    block = manifest_metrics(
        project(1, ReplayPlan(), ReplayBudget(max_model_calls=5)), sample, ReplayPlan()
    )["replay"]
    assert "realised" not in block


def test_a_negative_step_count_fails_loud_rather_than_projecting_nothing() -> None:
    with pytest.raises(ReplayError, match="selected_steps"):
        project(-1, ReplayPlan(), ReplayBudget(max_model_calls=10))


def test_a_negative_sample_size_fails_loud_rather_than_returning_empty() -> None:
    with pytest.raises(ReplayError, match="n must be non-negative"):
        stratified_sample([_record("t1", 0)], -1, seed=0)
