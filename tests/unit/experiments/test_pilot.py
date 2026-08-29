"""DSE-019 pilot gates: G1/G2/G3 known-answer fixtures, recommendation logic, report render."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.schema import Condition, Difficulty, HandoffRecord
from preceptx.experiments.pilot import (
    PilotConfig,
    g1_capability,
    g2_signal,
    g3_correctness,
    g3_groundedness,
    render_report,
    run_pilot,
    write_pilot_report,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.sim.feasibility import oracle_action


class _MsgEncoder:
    """4-dim stub: dim0 recovers the outcome token in the text, dims 1-3 are stable hash noise.

    A C0 message that names success/failure is linearly informative about Y (dim0); a C4 'noise'
    message is not (dim0=0) - so CPVI lifts on C0 rows and not C4, which is the G2 gradient.
    """

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
            flag = 1.0 if "success" in s else (-1.0 if "failure" in s else 0.0)
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append([flag, *np.random.default_rng(seed).standard_normal(3).tolist()])
        return np.array(rows, dtype=np.float64)


def _rec(
    ep: str,
    step: int,
    condition: Condition,
    *,
    success: bool,
    difficulty: Difficulty = "hard",
    message: str = "hold",
    state: dict[str, float] | None = None,
    y_progress: bool | None = None,  # per-handoff progress label; defaults to the episode outcome
    observation: str | None = None,  # None -> a restricted view distinct from A's state_str
    angle: float | None = None,  # load pose (rad); None -> step-varying, as a real episode
    taken: str | None = None,  # B's macro action; None -> the oracle's (a competent pair)
) -> HandoffRecord:
    angle = math.radians(15.0 * step) if angle is None else angle
    return HandoffRecord(
        episode_id=ep,
        step=step,
        condition=condition,
        serialisation="numeric",
        difficulty=difficulty,
        model="m",
        seed=int(ep[-1]) if ep[-1].isdigit() else 0,
        state=state or {},
        state_str=f"state {ep} s{step}",  # no outcome token -> e_s carries no Y signal
        observation=observation if observation is not None else f"partial {ep} s{step}",
        message_raw=message,
        message_delivered=message,
        action={"action": taken or oracle_action(angle)},
        pre_state={"angle": angle},
        post_state={},
        progress=0.0,
        success=success,
        collision=False,
        stuck=False,
        y_binary_progress=success if y_progress is None else y_progress,
        y_terminal_success=success,
    )


def test_g1_capability_passes_above_floor_fails_below() -> None:
    records = [_rec(f"c0_{i}", 0, "C0", success=i < 3, difficulty="easy") for i in range(4)]
    assert g1_capability(records, PilotConfig(g1_success_floor=0.5)).passed  # 3/4 succeed
    high = g1_capability(records, PilotConfig(g1_success_floor=0.9))
    assert not high.passed and high.value == 0.75


def test_g1_ignores_hard_difficulty() -> None:
    # The pilot cell crosses C0 with easy and hard. A pair that solves every easy episode and no
    # hard one scores 1.0 on the capability question and 0.5 on the mixed average - which would
    # have sat exactly on the floor and passed the gate for the wrong reason.
    records = [_rec(f"e{i}", 0, "C0", success=True, difficulty="easy") for i in range(3)]
    records += [_rec(f"h{i}", 0, "C0", success=False, difficulty="hard") for i in range(3)]
    res = g1_capability(records, PilotConfig())
    assert res.value == 1.0 and res.detail["n_easy_c0_episodes"] == 3.0


def test_g1_requires_easy_c0() -> None:
    records = [_rec("c4_0", 0, "C4", success=True), _rec("c0_0", 0, "C0", success=True)]
    with pytest.raises(ConfigError, match="easy C0"):
        g1_capability(records, PilotConfig())


def _gradient_dataset() -> list[HandoffRecord]:
    """C0: 80% success with outcome-naming messages; C4: 20% success with noise messages."""
    records: list[HandoffRecord] = []
    for i in range(10):  # C0 episodes, interleaved easy/hard so G1 has its own cell (4/5 easy)
        ok = i < 8
        msg = "report success" if ok else "report failure"
        diff: Difficulty = "easy" if i % 2 == 0 else "hard"
        records += [
            _rec(f"c0_{i}", s, "C0", success=ok, message=msg, difficulty=diff) for s in range(2)
        ]
    for i in range(10):  # C4 episodes
        ok = i < 2
        records += [_rec(f"c4_{i}", s, "C4", success=ok, message="channel noise") for s in range(2)]
    return records


def test_g2_signal_detects_both_gaps(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    res = g2_signal(_gradient_dataset(), feat, PilotConfig())
    assert res.passed
    assert res.detail["success_gap"] == pytest.approx(0.6)  # 0.8 - 0.2
    assert res.detail["cpvi_gap"] > 0.0  # informative C0 messages lift CPVI over noisy C4


def test_g2_guards_single_progress_class(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = [_rec(f"c0_{i}", 0, "C0", success=True) for i in range(3)]
    records += [_rec(f"c4_{i}", 0, "C4", success=True) for i in range(3)]  # every label positive
    res = g2_signal(records, feat, PilotConfig())
    # Unmeasurable, not failed: with one progress class CPVI has nothing to predict, and a FAIL here
    # would spend the retune and eventually invoke the fallback on an absence of data.
    assert not res.passed and not res.assessable and "UNASSESSABLE" in res.note


def test_g2_cpvi_uses_progress_labels_not_terminal_success(tmp_path: Path) -> None:
    # P1-2: every episode succeeds (terminal success is single-class - the old label would have
    # bailed as unmeasurable), but per-handoff progress varies, so the progress-labelled CPVI gap
    # is computable and positive for informative-C0 vs noise-C4 messages.
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records: list[HandoffRecord] = []
    for i in range(6):
        for s in range(2):
            prog = s == 0
            msg = "report success" if prog else "report failure"  # names the PROGRESS label
            records.append(_rec(f"c0_{i}", s, "C0", success=True, message=msg, y_progress=prog))
    for i in range(6):
        for s in range(2):
            records.append(
                _rec(f"c4_{i}", s, "C4", success=True, message="channel noise", y_progress=s == 0)
            )
    res = g2_signal(records, feat, PilotConfig())
    assert "cpvi_gap" in res.detail  # measurable despite single-class terminal success
    assert res.detail["cpvi_gap"] > 0.0


def test_g2_measures_cpvi_even_when_the_receiver_sees_the_whole_state(tmp_path: Path) -> None:
    # CPVI is V-*usable* information: a message can add information a bounded probe cannot extract
    # from the state embedding even when the receiver holds that state verbatim. E3-local measured
    # +0.19 bits in C0, where observation == state_str in every record, so shared observation must
    # not be treated as a structural zero.
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = [
        _rec(f"{ep}_{i}", s, c, success=ok, message=msg, observation=f"state {ep}_{i} s{s}")
        for c, ep, ok, msg in (
            ("C0", "c0", True, "report success"),
            ("C0", "c0b", False, "report failure"),
            ("C4", "c4", True, "channel noise"),
            ("C4", "c4b", False, "channel noise"),
        )
        for i in range(3)
        for s in range(2)
    ]
    res = g2_signal(records, feat, PilotConfig())
    assert res.assessable and "cpvi_gap" in res.detail
    assert res.detail["cpvi_gap"] > 0.0  # informative C0 messages still lift CPVI over noisy C4


def test_an_unassessable_gate_never_escalates_to_the_fallback(tmp_path: Path) -> None:
    # An unassessable gate is not a failed one: even on the second attempt it holds at retune_once
    # rather than invoking the pivot, and the note says why no verdict is available.
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = [  # every handoff positive -> CPVI undefined -> G2 unassessable
        _rec(f"{c}_{i}", 0, c, success=True, difficulty="easy")
        for c in ("C0", "C4")
        for i in range(3)
    ]
    report = run_pilot(records, feat, attempt=2)
    assert report.recommendation == "retune_once"
    assert "does not spend the retune" in report.recommendation_note


def test_g3_credits_geometry_the_sender_was_shown_but_state_does_not_carry() -> None:
    # `state` holds the load body only. A message correctly citing the wall abscissae and the slit
    # interval printed in `state_str` was scored as hallucinating them: G3 read 0.720 on a run whose
    # messages fabricated nothing. The truth set is what the sender was shown.
    rec = _rec(
        "e0",
        0,
        "C0",
        success=True,
        message="load (5.00, 3.00); slit 2.10 to 3.90; wall x=4.00",
        state={"com_x": 5.0, "com_y": 3.0},
    ).model_copy(
        update={
            "state_str": "load=(5.0000, 3.0000)\nwalls_x=(4.0000, 8.0000)\nslit_y=(2.1000, 3.9000)"
        }
    )
    assert g3_groundedness([rec], PilotConfig()).value == 1.0


def test_g3_truth_set_excludes_the_v5_action_history() -> None:
    """G3 scores messages against *geometry*, and v5's `recent=` line is not geometry.

    It lists the last four actions and the geodesic distance each gained. Leaving it in the truth
    set widened the admissible set from geometry to geometry-union-gains: with `g3_abs_tol = 0.5`
    and gains clustering in 0-1.5, a fabricated small-magnitude claim matched a gain and was scored
    grounded. The inflation is single-sided, so it degrades exactly what G3 certifies.
    """
    history = "recent=((N, +0.30), (E, +0.85), (N, +0.30))  # ... net +1.45 over the last 3"
    scene = "load=(1.8000, 1.8000)\nslit_y=(2.1000, 3.9000)"
    # 0.85 is no wall, slit, load or goal coordinate - it appears only as a gain in the history.
    fabricated = _rec(
        "e0",
        0,
        "C0",
        success=True,
        message="the load sits 0.85 below the slit",
        state={"com_x": 1.8, "com_y": 1.8},
    ).model_copy(update={"state_str": f"{scene}\n{history}"})
    assert g3_groundedness([fabricated], PilotConfig()).value == 0.0

    # The geometry half of the same state_str still grounds a true claim.
    honest = fabricated.model_copy(update={"message_delivered": "the slit starts at 2.10"})
    assert g3_groundedness([honest], PilotConfig()).value == 1.0


def test_g3_grounded_passes_hallucinated_fails() -> None:
    state = {"com_x": 5.0, "com_y": 3.0}
    grounded = [
        _rec(f"g{i}", 0, "C0", success=True, message="load at (5.00, 3.00)", state=state)
        for i in range(3)
    ]
    assert g3_groundedness(grounded, PilotConfig()).passed  # both numbers match the true state
    hallucinated = [
        _rec(f"h{i}", 0, "C0", success=True, message="load at (99.0, 88.0)", state=state)
        for i in range(3)
    ]
    bad = g3_groundedness(hallucinated, PilotConfig())
    assert not bad.passed and bad.value == 0.0  # fabricated coordinates ground nothing


def _pose_cell(taken: str | None) -> list[HandoffRecord]:
    """One episode of 20 handoffs alternating aligned (oracle E) and 30 deg off (ROT-)."""
    return [
        _rec("e0", i, "C0", success=True, angle=math.radians(30.0 * (i % 2)), taken=taken)
        for i in range(20)
    ]


def test_g3_correctness_fails_a_state_blind_agent() -> None:
    # Always-push-east is the projection-blindness failure and always-rotate is attempt 2's. Both
    # are invariant under permutation, so each sits exactly on its own null - which is the point of
    # a permutation threshold: neither can pass by being lucky about the base rate.
    for habit in ("E", "ROT-"):
        res = g3_correctness(_pose_cell(habit), n_perm=50)
        assert res.value == pytest.approx(res.detail["state_blind_mean"]) and not res.passed
        assert res.assessable and res.detail["p_value"] == pytest.approx(1.0)


def test_g3_correctness_passes_an_agent_that_reads_the_pose() -> None:
    res = g3_correctness(_pose_cell(None), n_perm=50)  # None -> the oracle's own action
    assert res.passed and res.value == 1.0
    assert res.detail["p_value"] == pytest.approx(1 / 51)  # dominates every permutation


def test_g3_correctness_is_unassessable_when_the_oracle_never_varies() -> None:
    # Every pose aligned: the reference action is constant, so agreement measures habit, not
    # reading. Unassessable rather than passed - the same discipline G2 applies to a flat label.
    records = [_rec(f"e{i}", 0, "C0", success=True, angle=0.0) for i in range(4)]
    res = g3_correctness(records, n_perm=50)
    assert not res.assessable and not res.passed and "UNASSESSABLE" in res.note


def test_run_pilot_recommendation_tracks_attempt(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    passing = run_pilot(_gradient_dataset(), feat, cfg=PilotConfig(), dataset_hash="d")
    assert passing.recommendation == "proceed"  # all three gates pass on the gradient dataset
    # Force a fail by an impossible floor; the recommendation escalates only after the one retune.
    strict = PilotConfig(g1_success_floor=1.0)
    retune = run_pilot(_gradient_dataset(), feat, cfg=strict, attempt=1)
    pivot = run_pilot(_gradient_dataset(), feat, cfg=strict, attempt=2)
    assert retune.recommendation == "retune_once"
    assert pivot.recommendation == "fallback"


def test_run_pilot_holds_proceed_below_seed_floor(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    # All three gates pass, but demand more seeds than the data has: a few-seed pass is noise,
    # so the proceed verdict is held back to retune rather than greenlighting the full sweep.
    report = run_pilot(_gradient_dataset(), feat, cfg=PilotConfig(min_seeds_for_proceed=99))
    assert all(g.passed for g in report.gates)
    assert report.recommendation == "retune_once"
    assert "seed" in report.recommendation_note
    assert report.n_seeds == 10  # the gradient dataset spans 10 distinct seeds


def test_render_and_write_report(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    report = run_pilot(_gradient_dataset(), feat, cfg=PilotConfig(), dataset_hash="d")
    text = render_report(report)
    assert "Pilot gate report" in text and "PASS" in text and "Fallback ladder" in text
    out = write_pilot_report(report, tmp_path / "rep")
    assert (out / "pilot.json").exists() and (out / "pilot.md").exists()


def test_pilot_report_embeds_provenance(tmp_path: Path) -> None:
    # The re-gate verdict is a result of record: it must carry the encoder revision and probe
    # config its G2 CPVI number was computed under, and the one-pager must show them.
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    report = run_pilot(_gradient_dataset(), feat, dataset_hash="d")
    assert report.provenance is not None
    assert report.provenance.encoder_revision == EncoderConfig().revision
    assert "encoder:" in render_report(report)
