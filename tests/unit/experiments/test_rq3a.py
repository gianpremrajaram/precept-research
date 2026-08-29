"""Offline tests for the RQ3a localisation analysis (DSE-024).

Nothing here calls a served model or touches a corpus: the judge is a scripted stub, the encoder is
a deterministic fake, and the annotations are planted, so the known-answer cases (which method finds
the planted step, what happens when it abstains) are exact rather than approximate. The two
structural guarantees - scorers cannot see annotations, and an unavailable regime keeps its row -
are asserted here rather than only described in the module docstring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.data.logs import LogHandoffRecord, LogTraceRecord
from preceptx.experiments.rq3a import (
    JudgeBackend,
    LocalisationStep,
    MethodScores,
    RQ3aConfig,
    StepScore,
    analyse_rq3a,
    cosine_scores,
    evaluate,
    judge_agreement,
    judge_all_at_once,
    judge_binary_search,
    judge_step_by_step,
    localisation_steps,
    manifest_metrics,
    mast_category,
    outcome_census,
    refit_scores,
    results_table,
    schema_validity_scores,
    trace_outcome_labels,
    trace_targets,
    transfer_scores,
    write_rq3a,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig

_FAST = RQ3aConfig(probe=ProbeConfig(n_repeats=1), n_boot=200)


class _Encoder:
    """dim0 recovers a planted 'bad' token, the rest is content-hashed noise."""

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
            flag = 1.0 if "bad" in s else 0.0
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append([flag, *np.random.default_rng(seed).standard_normal(3).tolist()])
        return np.array(rows, dtype=np.float64)


def _featuriser(tmp_path: Path) -> Featuriser:
    return Featuriser(EncoderConfig(cache_dir=tmp_path / "emb"), encoder=_Encoder())


def _trace(
    trace_id: str,
    n_steps: int,
    *,
    mistake_step: int | str | None = 1,
    mistake_agent: str | None = "coder",
    agents: list[str] | None = None,
    trace_failed: bool | None = None,
) -> list[LogHandoffRecord]:
    names = agents or ["planner", "coder"] * n_steps
    return [
        LogHandoffRecord(
            corpus="traceelephant",
            trace_id=trace_id,
            step=i,
            agent_name=names[i],
            receiver=names[(i + 1) % len(names)],
            is_handoff=True,
            observation=f"obs {trace_id} {i}",
            message=("bad message" if i == mistake_step else f"fine message {i}"),
            trace_failed=trace_failed,
            annotations={"mistake_step": mistake_step, "mistake_agent": mistake_agent},
        )
        for i in range(n_steps)
    ]


class _ScriptedJudge(JudgeBackend):
    """Always names the step whose message contains 'bad'; ``fails`` makes it abstain instead."""

    model_name = "stub-8b"
    model_revision = "rev0"
    decoding = "greedy, temperature=0, seed=0"

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls = 0

    def select_step(self, transcript: str, n_steps: int) -> int | None:
        self.calls += 1
        if self.fails:
            return None
        for i, block in enumerate(transcript.split("\n\n")):
            if "bad" in block:
                return i
        return None

    def contains_error(self, transcript: str) -> bool | None:
        self.calls += 1
        return None if self.fails else "bad" in transcript

    def is_error(self, transcript: str, step_text: str) -> bool | None:
        self.calls += 1
        return None if self.fails else "bad" in step_text


# ------------------------------------------------------------------ the annotation boundary


def test_localisation_step_cannot_carry_annotations() -> None:
    """The structural guard: the scoring view has no field an annotation could arrive in."""
    assert "annotations" not in LocalisationStep.model_fields
    assert "trace_failed" not in LocalisationStep.model_fields
    steps = localisation_steps(_trace("t0", 4))
    assert all(not hasattr(s, "annotations") for s in steps)


def test_localisation_steps_filter_to_handoffs_by_default() -> None:
    records = _trace("t0", 3)
    records[1] = records[1].model_copy(update={"is_handoff": False})
    assert [s.step for s in localisation_steps(records)] == [0, 2]
    assert [s.step for s in localisation_steps(records, handoffs_only=False)] == [0, 1, 2]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(2, 2), ("3", 3), ("", None), (None, None), (True, None), ("step two", None)],
)
def test_trace_targets_parse_only_real_step_annotations(raw: object, expected: int | None) -> None:
    records = _trace("t0", 3, mistake_step=raw)  # type: ignore[arg-type]
    assert trace_targets(records)["t0"].decisive_step == expected


# ------------------------------------------------------------------ metrics


def _scores(method: str, steps: list[LocalisationStep], risks: list[float]) -> MethodScores:
    return MethodScores(
        method=method,
        scores=[
            StepScore(trace_id=s.trace_id, step=s.step, risk=r)
            for s, r in zip(steps, risks, strict=True)
        ],
    )


def test_evaluate_recovers_a_planted_decisive_step() -> None:
    records = _trace("t0", 4, mistake_step=1, mistake_agent="coder")
    steps = localisation_steps(records)
    targets = trace_targets(records)
    perfect = _scores("perfect", steps, [0.0, 1.0, 0.0, 0.0])
    m = evaluate(perfect, steps, targets, n_boot=200)
    assert (m.status, m.step_accuracy, m.mrr, m.top_k_accuracy) == ("ok", 1.0, 1.0, 1.0)
    assert m.agent_accuracy == 1.0  # step 1 is the coder's, which is what the corpus annotated
    assert m.n_traces_evaluated == 1
    assert all(ci is not None for ci in (m.step_accuracy_ci, m.mrr_ci))  # never a bare rate

    worst = _scores("worst", steps, [1.0, 0.0, 0.9, 0.8])
    m2 = evaluate(worst, steps, targets, top_k=2, n_boot=200)
    assert (m2.step_accuracy, m2.top_k_accuracy) == (0.0, 0.0)
    assert m2.mrr == pytest.approx(0.25)  # the decisive step ranks last of four


def test_evaluate_ties_cannot_win_by_input_order() -> None:
    """A method that scores everything alike must not score as if it had picked first."""
    records = _trace("t0", 4)
    steps = localisation_steps(records)
    flat = evaluate(_scores("flat", steps, [0.5] * 4), steps, trace_targets(records), n_boot=200)
    assert flat.step_accuracy == 0.0
    assert flat.mrr == pytest.approx(1 / 2.5)  # average rank over four tied steps


def test_evaluate_reports_targets_that_fall_outside_the_scored_steps() -> None:
    """An annotated step on a non-handoff turn is bookkeeping, not a miss - and is counted."""
    records = _trace("t0", 3, mistake_step=1)
    records[1] = records[1].model_copy(update={"is_handoff": False})
    steps = localisation_steps(records)
    m = evaluate(_scores("s", steps, [1.0, 0.0]), steps, trace_targets(records), n_boot=200)
    assert m.status == "not_applicable"
    assert m.n_traces_target_off_boundary == 1
    assert m.step_accuracy is None  # an empty cell, never a zero that reads as a measurement


def test_evaluate_keeps_the_row_when_a_method_is_unavailable() -> None:
    m = evaluate(
        MethodScores(method="cpvi_transfer", status="unavailable", reason="no probe"),
        localisation_steps(_trace("t0", 3)),
        trace_targets(_trace("t0", 3)),
    )
    assert (m.method, m.status, m.reason) == ("cpvi_transfer", "unavailable", "no probe")
    assert m.step_accuracy is None


# ------------------------------------------------------------------ baselines and regimes


def test_baselines_score_every_step(tmp_path: Path) -> None:
    records = _trace("t0", 3)
    records[2] = records[2].model_copy(update={"message": "  "})
    steps = localisation_steps(records)
    schema = schema_validity_scores(steps)
    assert [s.risk for s in schema.scores] == [0.0, 0.0, 1.0]  # the empty emission is the suspect
    cos = cosine_scores(steps, _featuriser(tmp_path))
    assert len(cos.scores) == 3 and all(-1.0001 <= s.risk <= 1.0001 for s in cos.scores)


def test_transfer_is_unavailable_without_a_statistic_or_an_orientation(tmp_path: Path) -> None:
    steps = localisation_steps(_trace("t0", 3))
    feat = _featuriser(tmp_path)
    none_yet = transfer_scores(steps, feat, key="s_info", dir=None, orientation=-1.0)
    assert none_yet.status == "unavailable" and "arena track" in (none_yet.reason or "")

    unsigned = transfer_scores(steps, feat, key="s_info", dir=tmp_path, orientation=None)
    assert unsigned.status == "unavailable"
    assert "sign" in (unsigned.reason or "")  # refusing to guess, not defaulting to +1

    missing = transfer_scores(steps, feat, key="s_info", dir=tmp_path, orientation=-1.0)
    assert missing.status == "unavailable" and "s_info" in (missing.reason or "")


def test_transfer_resolves_a_key_retired_by_dse_061(tmp_path: Path) -> None:
    """A manifest or config naming 'info' must find the survivor, not report a missing statistic."""
    steps = localisation_steps(_trace("t0", 3))
    out = transfer_scores(steps, _featuriser(tmp_path), key="info", dir=tmp_path, orientation=1.0)
    assert out.status == "unavailable"
    assert "'fail'" in (out.reason or "")  # resolved past the retired alias before looking


# ------------------------------------------------------------------ the refit regime's outcome


def test_trace_outcome_labels_drop_unlabelled_traces() -> None:
    records = _trace("t0", 3, trace_failed=True) + _trace("t1", 3, trace_failed=None)
    labels = trace_outcome_labels(records)
    assert set(labels) == {("t0", 0), ("t0", 1), ("t0", 2)}
    assert all(labels.values())


def test_outcome_census_reports_a_usable_fallback_when_both_classes_are_present() -> None:
    records = _trace("t0", 3, trace_failed=True) + _trace("t1", 3, trace_failed=False)
    census = outcome_census(records, source="trace-outcome")
    assert census.usable and census.reason == ""
    assert (census.traces_failed, census.traces_succeeded, census.traces_unlabelled) == (1, 1, 0)
    assert census.steps_labelled == 6


def test_outcome_census_measures_the_degeneracy_rather_than_asserting_it() -> None:
    """The shape of both real corpora: failures and unlabelled traces, no non-failure class."""
    records = _trace("t0", 3, trace_failed=True) + _trace("t1", 3, trace_failed=None)
    census = outcome_census(records, source="trace-outcome")
    assert not census.usable
    assert (census.traces_failed, census.traces_succeeded, census.traces_unlabelled) == (1, 0, 1)
    assert "1 failed, 0 succeeded, 1 unlabelled of 2 traces" in census.reason
    assert "DSE-042" in census.reason  # replay is named as the only remaining route


def test_refit_is_not_applicable_on_a_single_class_corpus(tmp_path: Path) -> None:
    """Who&When is 184/184 failures; a probe has nothing to separate, and that is not an error."""
    steps = localisation_steps(_trace("t0", 4))
    labels = {(s.trace_id, s.step): True for s in steps}
    out = refit_scores(steps, _featuriser(tmp_path), labels, ProbeConfig(n_repeats=1))
    assert out.status == "not_applicable" and "single-class" in (out.reason or "")


def test_refit_scores_the_labelled_steps_and_says_how_many(tmp_path: Path) -> None:
    steps = localisation_steps(_trace("t0", 4) + _trace("t1", 4))
    labels = {(s.trace_id, s.step): ("bad" in s.message) for s in steps[:-1]}
    out = refit_scores(steps, _featuriser(tmp_path), labels, ProbeConfig(n_repeats=1))
    assert out.status == "ok"
    assert len(out.scores) == len(steps) - 1
    assert "7 of 8" in (out.reason or "")


def test_refit_is_unavailable_without_replay_labels(tmp_path: Path) -> None:
    steps = localisation_steps(_trace("t0", 3))
    out = refit_scores(steps, _featuriser(tmp_path), {}, ProbeConfig(n_repeats=1))
    assert out.status == "unavailable" and "no replay outcome labels" in (out.reason or "")


def test_refit_reasons_name_the_label_source_they_actually_used(tmp_path: Path) -> None:
    """ "No labels" and "labels that cannot separate" are different findings about a corpus.

    Before DSE-024 every unavailable refit row blamed DSE-042 for not having run, which reads as
    "pending" - and on both RQ3a corpora the truth is stronger: no annotation-free non-failure class
    exists, so no amount of running the labeller as specified would produce one.
    """
    steps = localisation_steps(_trace("t0", 4))
    feat, probe = _featuriser(tmp_path), ProbeConfig(n_repeats=1)

    absent = refit_scores(steps, feat, {}, probe, label_source="trace-outcome")
    assert absent.status == "unavailable"
    assert "no trace-outcome outcome labels" in (absent.reason or "")

    single = refit_scores(
        steps,
        feat,
        {(s.trace_id, s.step): True for s in steps},
        probe,
        label_source="trace-outcome",
    )
    assert single.status == "not_applicable"
    assert "trace-outcome labels are single-class" in (single.reason or "")


# ------------------------------------------------------------------ the judge replications


def test_all_three_judge_procedures_find_the_planted_step() -> None:
    records = _trace("t0", 8, mistake_step=5)
    steps = localisation_steps(records)
    targets = trace_targets(records)
    for scorer in (judge_all_at_once, judge_binary_search, judge_step_by_step):
        scores = scorer(steps, _ScriptedJudge())
        m = evaluate(scores, steps, targets, n_boot=200)
        assert m.step_accuracy == 1.0, scores.method
        assert scores.n_abstained == 0


def test_binary_search_costs_log_calls_and_step_by_step_costs_linear_ones() -> None:
    steps = localisation_steps(_trace("t0", 8, mistake_step=5))
    bisect, walk = _ScriptedJudge(), _ScriptedJudge()
    judge_binary_search(steps, bisect)
    judge_step_by_step(steps, walk)
    assert bisect.calls == 3  # ceil(log2(8)) narrowing questions
    assert walk.calls == 6  # one per step up to and including the hit


def test_a_failing_judge_abstains_and_never_falls_back_to_the_annotation() -> None:
    records = _trace("t0", 4, mistake_step=2)
    steps = localisation_steps(records)
    for scorer in (judge_all_at_once, judge_binary_search, judge_step_by_step):
        scores = scorer(steps, _ScriptedJudge(fails=True))
        assert scores.n_abstained == 1, scores.method
        assert all(s.risk == 0.0 for s in scores.scores)  # a flat, unrankable trace
        assert evaluate(scores, steps, trace_targets(records), n_boot=200).step_accuracy == 0.0


def test_judge_agreement_is_kappa_against_the_existing_annotation() -> None:
    records = _trace("t0", 4, mistake_step=1, mistake_agent="coder") + _trace(
        "t1", 4, mistake_step=2, mistake_agent="planner"
    )
    steps = localisation_steps(records)
    audit = judge_agreement(
        judge_all_at_once(steps, _ScriptedJudge()), steps, trace_targets(records), n_sample=10
    )
    assert audit.n_sampled == 2
    assert audit.kappa is not None
    assert "existing annotation" in audit.comparison  # named for what it is, not "human agreement"


def test_agreement_is_reported_as_unavailable_when_the_judge_was_not_run() -> None:
    steps = localisation_steps(_trace("t0", 3))
    audit = judge_agreement(
        MethodScores(method="judge_all_at_once", status="unavailable", reason="no backend"),
        steps,
        trace_targets(_trace("t0", 3)),
    )
    assert audit.kappa is None and "no backend" in (audit.reason or "")


# ------------------------------------------------------------------ MAST


def _mast(n: int, tmp_path: Path) -> list[LogTraceRecord]:
    return [
        LogTraceRecord(
            corpus="mast",
            trace_id=f"sys/bench/{i}",
            system_name="sys",
            trace_text=("bad handoff " if i % 2 else "clean handoff ") * 3 + f"trace {i}",
            annotations={"1.1": 0, "2.1": int(i % 2), "3.1": 0},
        )
        for i in range(n)
    ]


def test_mast_arm_reports_bits_and_records_that_cpvi_is_undefined(tmp_path: Path) -> None:
    out = mast_category(_mast(12, tmp_path), _featuriser(tmp_path), ProbeConfig(n_repeats=1))
    assert out.status == "ok"
    assert (out.n_traces, out.n_positive) == (12, 6)
    assert out.category_information_bits is not None and out.category_information_ci is not None
    assert out.cpvi_status == "not_applicable"  # no observation/message split on this corpus
    assert "unsegmented transcript" in out.cpvi_reason


def test_mast_arm_is_not_applicable_when_the_category_is_single_class(tmp_path: Path) -> None:
    traces = [t.model_copy(update={"annotations": {"2.1": 0}}) for t in _mast(4, tmp_path)]
    out = mast_category(traces, _featuriser(tmp_path), ProbeConfig(n_repeats=1))
    assert out.status == "not_applicable" and out.category_information_bits is None


# ------------------------------------------------------------------ the driver


def test_analyse_rq3a_runs_every_method_and_keeps_unavailable_rows(tmp_path: Path) -> None:
    records = _trace("t0", 6, mistake_step=3) + _trace("t1", 6, mistake_step=2)
    labels = {
        (r.trace_id, r.step): ("bad" in r.message) for r in records if r.trace_id == "t0" or r.step
    }
    result = analyse_rq3a(
        records,
        _featuriser(tmp_path),
        cfg=_FAST,
        judge=_ScriptedJudge(),
        labels=labels,
        mast_traces=_mast(12, tmp_path),
    )
    by_method = {m.method: m for m in result.methods}
    assert set(by_method) == {
        "schema_validity",
        "mean_cosine",
        "cpvi_transfer",
        "cpvi_refit",
        "judge_all_at_once",
        "judge_binary_search",
        "judge_step_by_step",
    }
    assert by_method["cpvi_transfer"].status == "unavailable"  # no frozen probe exists yet
    assert by_method["judge_all_at_once"].step_accuracy == 1.0
    assert result.judge is not None and result.judge.model_name == "stub-8b"
    assert "open-weight re-implementation" in result.judge.published_result
    assert result.mast is not None and result.agreement is not None
    assert result.n_traces == 2 and result.n_steps_scored == 12

    table = results_table(result)
    assert len(table) == 7  # every method keeps its row, unavailable ones included
    out = write_rq3a(result, tmp_path / "rq3a")
    assert (out / "rq3a.json").exists() and (out / "rq3a_localisation.csv").exists()

    block = manifest_metrics(result)["rq3a"]
    assert isinstance(block, dict)
    assert block["tie_policy"] == result.tie_policy
    assert "cpvi_transfer" in block["unavailable"]
    assert block["outcomes"]["source"] == "replay"  # labels were supplied, so no fallback was used


def test_analyse_falls_back_to_the_trace_outcome_when_replay_has_not_run(tmp_path: Path) -> None:
    """DSE-042's stated fallback, actually wired - and its census records whether it could work."""
    # Six traces, not two: the refit cross-fits by trace, so a two-trace grid gives every fold a
    # single-class training set and the probe cannot fit at all.
    records = [
        r for i in range(6) for r in _trace(f"t{i}", 6, mistake_step=3, trace_failed=bool(i % 2))
    ]
    result = analyse_rq3a(records, _featuriser(tmp_path), cfg=_FAST)
    assert result.outcomes.source == "trace-outcome" and result.outcomes.usable
    assert {m.method: m.status for m in result.methods}["cpvi_refit"] == "ok"

    degenerate = analyse_rq3a(
        _trace("t0", 6, trace_failed=True) + _trace("t1", 6, trace_failed=True),
        _featuriser(tmp_path),
        cfg=_FAST,
    )
    assert not degenerate.outcomes.usable
    by_method = {m.method: m for m in degenerate.methods}
    assert by_method["cpvi_refit"].status == "not_applicable"
    assert "trace-outcome labels are single-class" in (by_method["cpvi_refit"].reason or "")
