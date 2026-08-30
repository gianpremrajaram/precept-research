"""DSE-020 RQ1 driver: the factorial assembles correctly and the analysis recovers a known gradient.

The synthetic fixture is the mandated known-answer case (CLAUDE.md), and is built so CPVI genuinely
*mediates* condition -> success rather than being a parallel consequence of it: one per-episode flag
(is this episode's message informative?) sets the message, the per-handoff progress label, and the
terminal outcome together, and the fraction of informative episodes falls C0->C4. So a correct
analysis must surface the success gradient, the CPVI gradient, a negative C4 handoff coefficient,
corrected contrast p-values, AND a positive CPVI->success path with a negative C4 indirect effect.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.rq1 import (
    RQ1Config,
    action_agreement,
    analyse_rq1,
    rq1_sweep,
    signal_decomposition,
    write_rq1,
)
from preceptx.experiments.sweep import expand
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.sim.feasibility import oracle_action

# Three poses whose oracle actions are not all the same, so the agreement null is assessable:
# flat wants E, +30 deg wants ROT-, -30 deg wants ROT+.
_ANGLES = [0.0, math.radians(30.0), math.radians(-30.0)]


def _oracle_or_habit(step: int, informative: bool) -> str:
    """The oracle's own action when the channel works, a fixed push when it does not."""
    return oracle_action(_ANGLES[step % len(_ANGLES)]) if informative else "E"


_MODEL = ModelConfig(name="m", revision="rev", tier="8b")
# Keep the model-refit mediation bootstrap small so the unit suite stays well under its 30s budget.
_FAST = RQ1Config(n_boot=300, n_boot_mediation=50)


class _MsgEncoder:
    """dim0 recovers the progress token ('progress'/'stuck'); a 'noise' message carries none."""

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
            flag = 1.0 if "progress" in s else (-1.0 if "stuck" in s else 0.0)
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append([flag, *np.random.default_rng(seed).standard_normal(3).tolist()])
        return np.array(rows, dtype=np.float64)


def test_rq1_sweep_assembles_full_factorial() -> None:
    sweep = rq1_sweep(
        _MODEL, seeds=[1, 2], serialisations=["numeric", "grid"], difficulties=["hard"]
    )
    cells = expand(sweep)
    assert len(cells) == 5 * 2 * 1 * 2  # |C| * |S| * |D| * |seeds|
    assert {c.condition for c in cells} == {"C0", "C1", "C2", "C3", "C4"}


def _gradient_records(n_seeds: int = 6) -> list[HandoffRecord]:
    # Two falling-with-condition drivers, kept separate:
    #  - true per-handoff progress (rate falls C0->C4) is the handoff outcome y_binary_progress;
    #  - the per-episode "informative" flag (count falls C0->C4) drives the message, CPVI, success.
    # An informative episode's message reveals its true progress ("progress"/"stuck", so CPVI lifts
    # whether progress is 0 or 1); a non-informative one emits constant "noise". Noise rows span
    # conditions, so they carry mixed y - noise is genuinely uninformative and CPVI falls C0->C4. In
    # mixed conditions C1-C3, informative episodes carry high CPVI and succeed while non-informative
    # ones do not - that within-condition spread identifies the positive CPVI->success path.
    informative_count = {"C0": 6, "C1": 5, "C2": 3, "C3": 1, "C4": 0}  # of n_seeds=6 episodes
    progress_rate = {"C0": 0.9, "C1": 0.7, "C2": 0.5, "C3": 0.3, "C4": 0.1}  # per-handoff true rate
    records: list[HandoffRecord] = []
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        c: Condition = cond  # type: ignore[assignment]
        n_handoffs = n_seeds * 3
        n_prog = round(progress_rate[cond] * n_handoffs)
        prog_flags = [1] * n_prog + [0] * (n_handoffs - n_prog)
        h = 0
        for seed in range(n_seeds):
            informative = seed < informative_count[cond]
            for step in range(3):
                y = prog_flags[h]
                h += 1
                # Lengths differ with informativeness on purpose: that is the confound C1 creates
                # by construction, and the DSE-044 length covariate exists to control for it.
                msg = ("clear progress" if y else "we are stuck") if informative else "noise"
                records.append(
                    HandoffRecord(
                        episode_id=f"{cond}-s{seed}",
                        step=step,
                        condition=c,
                        serialisation="numeric",
                        difficulty="hard",
                        model="m",
                        seed=seed,
                        state={},
                        state_str=f"state {cond} s{seed} {step}",  # no outcome token in the state
                        observation=f"state {cond} s{seed} {step}",
                        message_raw=msg,
                        message_delivered=msg,
                        # A real pose and a real action: `action_agreement` scores B against the
                        # oracle, and an empty payload is a shape no recorded handoff ever has.
                        # The informative arm follows the oracle; the rest push east regardless.
                        action={"action": _oracle_or_habit(step, informative)},
                        pre_state={"angle": _ANGLES[step % len(_ANGLES)]},
                        post_state={},
                        progress=0.0,
                        success=informative,  # the working channel is what gets the load home
                        collision=False,
                        stuck=False,
                        y_binary_progress=bool(y),
                        y_terminal_success=informative,
                    )
                )
    return records


def test_single_condition_grid_refuses_the_gradient_estimands(tmp_path: Path) -> None:
    """DSE-067: a C0-only capability grid must not report a zero gradient as a finding.

    Both v9 arms did exactly that. The C0-minus-hardest gap becomes C0 minus itself - identically
    zero for every seed, which reads as a perfectly seed-stable ordering - and C(condition) becomes
    rank-deficient, which statsmodels answers with a boundary fit rather than a refusal.
    """
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = [r for r in _gradient_records(n_seeds=20) if r.condition == "C0"]
    result, _ = analyse_rq1(records, feat, dataset_hash="d0", cfg=_FAST)

    ss = result.seed_sensitivity
    assert ss.metric == "success_rate"  # not the self-subtracting gap
    assert "self-subtraction" in ss.reason  # and the artefact says why it switched
    assert ss.n_seeds == 20
    assert ss.dispersion is None or ss.dispersion >= 0.0

    mm = result.mixed_model
    assert not mm.converged  # refused, not fitted to a boundary
    assert mm.coef_no_mediator == {}  # no contrast was invented
    assert mm.mediations == []
    assert "rank-deficient" in mm.mediation_note


def test_analyse_rq1_recovers_the_gradient(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = _gradient_records()
    result, scores = analyse_rq1(records, feat, dataset_hash="d0", cfg=_FAST)

    order = [c.condition for c in result.conditions]
    assert order == ["C0", "C1", "C2", "C3", "C4"]
    assert result.conditions[0].success_rate > result.conditions[-1].success_rate  # H1 outcome
    assert result.conditions[0].mean_cpvi > result.conditions[-1].mean_cpvi  # CPVI gradient
    assert all(np.isfinite(c.pvi_cpvi_gap) for c in result.conditions)  # the gap is always reported

    mm = result.mixed_model
    assert mm.coef_no_mediator["C4"] < 0.0  # H1 handoff model: degradation relative to C0
    assert np.isfinite(mm.diagnostic_cpvi_coef)  # within-episode diagnostic still computed
    assert mm.mediation_outcome == "episode_success"  # H2 tests the headline DV, not progress
    assert "mediated by CPVI" in mm.mediation_note

    # H2 episode-level mediation: the channel suppresses success *through* lowered CPVI (a*b < 0).
    assert mm.path_b > 0.0  # more episode-mean CPVI -> more success
    c4_med = next(m for m in mm.mediations if m.condition == "C4")
    assert c4_med.path_a < 0.0  # C4 carries less CPVI than C0
    assert c4_med.indirect < 0.0  # negative indirect effect: degradation flows through CPVI
    assert all(np.isfinite(c4_med.indirect_ci))  # the indirect effect is reported with an interval
    assert c4_med.indirect_n_draws > 0  # the CI's retained-draw count is visible (P2-6)

    c4 = next(c for c in result.contrasts if c.condition == "C4")
    assert c4.cliffs_delta < 0.0  # C4 episodes succeed less often than C0
    assert np.isfinite(c4.steps_delta)  # efficiency endpoint reported per contrast (P1-11)
    assert all(np.isfinite(c4.steps_delta_ci))
    assert c4.p_corrected >= c4.p_raw - 1e-12  # correction never increases significance
    assert result.seed_sensitivity.n_seeds == 6  # per-seed C0-minus-hardest gap, one value per seed
    assert result.seed_sensitivity.mean > 0.0  # the gradient holds on average across seeds

    # Path b reported both raw and with the DSE-044 length covariate; C1 confounds length by design.
    assert np.isfinite(mm.path_b_length_controlled)

    # DSE-043: random labels cannot be predicted, so selectivity is essentially the whole score.
    assert abs(result.control_mean_cpvi) < 0.06
    assert result.selectivity > 0.0
    assert all(c.selectivity > 0.0 for c in result.conditions if c.condition == "C0")
    assert np.isfinite(result.partial_spearman_length)

    # Per-handoff scores are returned row-aligned to the records (P1-17: RQ2's join key).
    assert list(scores.columns) == [
        "episode_id",
        "step",
        "condition",
        "seed",
        "cpvi",
        "cpvi_sd",
        "pvi",
        "msg_tokens",
    ]
    assert (scores["cpvi_sd"] >= 0.0).all()  # across-repeat spread persisted (DSE-044)
    assert len(scores) == len(records)
    assert scores["episode_id"].tolist() == [r.episode_id for r in records]

    # Provenance rides the result (P1-8): encoder + probe + code identity.
    assert result.provenance.encoder_name == EncoderConfig().name
    assert len(result.provenance.git_sha) == 40


def test_write_rq1_emits_table_json_and_scores(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result, scores = analyse_rq1(_gradient_records(), feat, dataset_hash="d0", cfg=_FAST)
    out = write_rq1(result, tmp_path / "rq1", scores=scores)
    assert (out / "rq1.json").exists()
    assert (out / "rq1_results.csv").exists()
    persisted = pd.read_parquet(out / "scores.parquet")
    assert len(persisted) == len(scores)  # the per-handoff distribution survives persistence
    # matplotlib is the optional viz extra; absent it the figures dict stays empty (no crash).


def test_length_matched_control_is_reported_for_every_condition(tmp_path: Path) -> None:
    """PREREGISTRATION section 5 promises both length controls, so both must reach the result.

    The covariate model (``path_b_length_controlled``) and the overlap-restricted contrast answer
    the same objection differently: one adjusts inside the model, the other refuses to compare
    outside the region where both conditions supply episodes. A result carrying only the first
    would under-deliver on the pre-registration.
    """
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result, _ = analyse_rq1(_gradient_records(), feat, dataset_hash="d0", cfg=_FAST)

    matched = {m.condition: m for m in result.length_matched}
    assert set(matched) == {"C1", "C2", "C3", "C4"}  # every Ck against the C0 reference, C0 aside
    for m in matched.values():
        # Both outcomes are stratified identically, so their bookkeeping cannot disagree.
        assert m.success.n_total == m.cpvi.n_total
        assert m.success.n_kept == m.cpvi.n_kept <= m.success.n_total
        assert m.success.n_bins == m.cpvi.n_bins
        # The unrestricted difference is always available, even where the overlap is too thin.
        assert np.isfinite(m.success.delta_unrestricted)
        if m.success.interpretable:
            assert np.isfinite(m.success.delta)
        else:
            assert np.isnan(m.success.delta) and "overlap" in m.success.note


def _decomposition_records(y_flags: list[int], n_episodes: int = 2) -> list[HandoffRecord]:
    """Handoffs in one condition carrying planted progress labels, two episodes deep."""
    per_ep = len(y_flags) // n_episodes
    return [
        HandoffRecord(
            episode_id=f"C0-s{i // per_ep}",
            step=i % per_ep,
            condition="C0",
            serialisation="numeric",
            difficulty="hard",
            model="m",
            seed=i // per_ep,
            state={},
            state_str="s",
            observation="s",
            message_raw="m",
            message_delivered="m",
            action={"action": "E"},
            pre_state={"angle": _ANGLES[i % len(_ANGLES)]},
            post_state={},
            progress=0.0,
            success=False,
            collision=False,
            stuck=False,
            y_binary_progress=bool(flag),
            y_terminal_success=False,
        )
        for i, flag in enumerate(y_flags)
    ]


def test_signal_decomposition_recovers_planted_absent_and_unused_populations() -> None:
    """DSE-046: the 2x2 and the two rates on a fixture whose cells are known by construction.

    CPVI is passed in rather than estimated, so the test pins the decomposition itself: three
    absent-signal handoffs (below the within-condition median, no progress) and one unused-signal
    handoff (above it, no progress) out of eight.
    """
    cpvi = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    y_flags = [1, 1, 1, 0, 1, 0, 0, 0]
    records = _decomposition_records(y_flags)
    y = np.array(y_flags, dtype=int)

    dec = signal_decomposition(records, cpvi, y, _FAST)
    assert len(dec) == 1
    d = dec[0]
    assert d.median_cpvi == 0.5  # within-condition median of the eight scores
    assert (d.low_cpvi_no_progress, d.low_cpvi_progress) == (3, 1)
    assert (d.high_cpvi_no_progress, d.high_cpvi_progress) == (1, 3)
    assert d.absent_signal_rate == 3 / 8
    assert d.unused_signal_rate == 1 / 8
    # The two rates decompose the condition's no-progress rate; that is what makes them additive
    # rather than two conditionals, one of which would be one minus the other.
    assert d.absent_signal_rate + d.unused_signal_rate == y_flags.count(0) / len(y_flags)
    for lo, hi in (d.absent_signal_ci, d.unused_signal_ci):
        assert lo <= hi  # reported with an interval, never as a bare rate


def test_signal_decomposition_ties_go_low_and_leave_the_unused_cell_empty() -> None:
    """Every score equal to the median is a 'low' handoff, so the split stays deterministic."""
    cpvi = np.full(8, 0.25, dtype=np.float64)
    y_flags = [1, 0, 1, 0, 1, 0, 1, 0]
    dec = signal_decomposition(_decomposition_records(y_flags), cpvi, np.array(y_flags), _FAST)[0]
    assert dec.low_cpvi_no_progress + dec.low_cpvi_progress == 8
    assert dec.unused_signal_rate == 0.0
    assert dec.absent_signal_rate == 0.5


def test_analyse_rq1_reports_the_decomposition_for_every_condition(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = _gradient_records()
    result, _ = analyse_rq1(records, feat, dataset_hash="d0", cfg=_FAST)

    dec = {d.condition: d for d in result.signal_decomposition}
    assert list(dec) == ["C0", "C1", "C2", "C3", "C4"]
    for cond, d in dec.items():
        cells = (
            d.low_cpvi_no_progress
            + d.low_cpvi_progress
            + d.high_cpvi_no_progress
            + d.high_cpvi_progress
        )
        assert cells == d.n_handoffs  # the 2x2 partitions the condition's handoffs
        fails = sum(1 for r in records if r.condition == cond and r.y_binary_progress is False)
        assert d.absent_signal_rate + d.unused_signal_rate == fails / d.n_handoffs


# ------------------------------------------- per-condition receiver competence (the E3 mechanism)


def _pose_records(condition: Condition, taken: str | None, n_ep: int = 4) -> list[HandoffRecord]:
    """Episodes of 10 handoffs cycling three poses; ``taken=None`` follows the oracle."""
    return [
        HandoffRecord(
            episode_id=f"{condition}-e{e}",
            step=i,
            condition=condition,
            serialisation="numeric",
            difficulty="hard",
            model="m",
            seed=e,
            state={},
            state_str="s",
            observation="s",
            message_raw="m",
            message_delivered="m",
            action={"action": taken or oracle_action(_ANGLES[i % len(_ANGLES)])},
            pre_state={"angle": _ANGLES[i % len(_ANGLES)]},
            post_state={},
            progress=0.0,
            success=False,
            collision=False,
            stuck=False,
            y_binary_progress=bool(i % 2),
            y_terminal_success=False,
        )
        for e in range(n_ep)
        for i in range(10)
    ]


def test_action_agreement_separates_a_pose_reader_from_a_habit() -> None:
    """The finding the E3 re-gate turns on: pooling hides *which* condition was blind."""
    reader = _pose_records("C0", None)
    habit = _pose_records("C4", "E")
    rows = {a.condition: a for a in action_agreement(reader + habit, n_perm=50)}
    assert rows["C0"].agreement == 1.0 and rows["C0"].p_value == pytest.approx(1 / 51)
    # A state-blind agent scores its own null by construction, whatever its habits are.
    assert rows["C4"].agreement == pytest.approx(rows["C4"].null_mean)
    assert rows["C4"].p_value > 0.05


def test_action_agreement_null_is_invariant_to_the_rest_of_the_grid() -> None:
    """A condition's null must not depend on which *other* conditions share the dataset.

    One RNG stream over the whole loop made C4's p-value a function of how many conditions were
    scored before it, so the same arm read differently on a four-condition grid than on a two-
    condition one - and A2's decision rule compares a two-condition arm against E3's C4 number.
    """
    habit = _pose_records("C4", "E")
    alone = {a.condition: a for a in action_agreement(habit, n_perm=50)}
    with_others = {
        a.condition: a for a in action_agreement(_pose_records("C0", None) + habit, n_perm=50)
    }
    # JSON, not model_dump: the rotation fields are NaN here and NaN != NaN under ==.
    assert alone["C4"].model_dump_json() == with_others["C4"].model_dump_json()


def test_action_agreement_reports_oscillation_and_tie_free_direction() -> None:
    rows = {a.condition: a for a in action_agreement(_pose_records("C0", None), n_perm=20)}
    c0 = rows["C0"]
    # The oracle alternates ROT-/ROT+ across the two non-flat poses, so a perfect follower reverses
    # on every consecutive rotation pair - flip rate 1.0 - and never turns the wrong way.
    assert c0.rotation_direction_agreement == 1.0 and c0.n_rotations > 0
    assert c0.rotation_flip_rate == pytest.approx(1.0)


def test_action_agreement_flip_rate_is_nan_when_nothing_rotates() -> None:
    """NaN, not 0.0: a condition that never rotates has no oscillation, not perfect stability."""
    (row,) = action_agreement(_pose_records("C1", "E"), n_perm=20)
    assert math.isnan(row.rotation_flip_rate) and math.isnan(row.rotation_direction_agreement)
    assert row.n_rotations == 0
