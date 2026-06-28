"""DSE-019 pilot gates: G1/G2/G3 known-answer fixtures, recommendation logic, report render."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.pilot import (
    PilotConfig,
    g1_capability,
    g2_signal,
    g3_groundedness,
    render_report,
    run_pilot,
    write_pilot_report,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser


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
    message: str = "hold",
    state: dict[str, float] | None = None,
) -> HandoffRecord:
    return HandoffRecord(
        episode_id=ep,
        step=step,
        condition=condition,
        serialisation="numeric",
        difficulty="hard",
        model="m",
        seed=int(ep[-1]) if ep[-1].isdigit() else 0,
        state=state or {},
        state_str=f"state {ep} s{step}",  # no outcome token -> e_s carries no Y signal
        message_raw=message,
        message_delivered=message,
        action={},
        pre_state={},
        post_state={},
        progress=0.0,
        success=success,
        collision=False,
        stuck=False,
        y_terminal_success=success,
    )


def test_g1_capability_passes_above_floor_fails_below() -> None:
    records = [_rec(f"c0_{i}", 0, "C0", success=i < 3) for i in range(4)]  # 3/4 succeed
    assert g1_capability(records, PilotConfig(g1_success_floor=0.5)).passed
    high = g1_capability(records, PilotConfig(g1_success_floor=0.9))
    assert not high.passed and high.value == 0.75


def test_g1_requires_c0() -> None:
    records = [_rec("c4_0", 0, "C4", success=True)]
    with pytest.raises(ConfigError, match="C0"):
        g1_capability(records, PilotConfig())


def _gradient_dataset() -> list[HandoffRecord]:
    """C0: 80% success with outcome-naming messages; C4: 20% success with noise messages."""
    records: list[HandoffRecord] = []
    for i in range(10):  # C0 episodes
        ok = i < 8
        msg = "report success" if ok else "report failure"
        records += [_rec(f"c0_{i}", s, "C0", success=ok, message=msg) for s in range(2)]
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


def test_g2_guards_single_outcome_class(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    records = [_rec(f"c0_{i}", 0, "C0", success=True) for i in range(3)]
    records += [_rec(f"c4_{i}", 0, "C4", success=True) for i in range(3)]  # all succeed
    res = g2_signal(records, feat, PilotConfig())
    assert not res.passed and "single outcome class" in res.note  # CPVI unmeasurable, not a crash


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


def test_render_and_write_report(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    report = run_pilot(_gradient_dataset(), feat, cfg=PilotConfig(), dataset_hash="d")
    text = render_report(report)
    assert "Pilot gate report" in text and "PASS" in text and "Fallback ladder" in text
    out = write_pilot_report(report, tmp_path / "rep")
    assert (out / "pilot.json").exists() and (out / "pilot.md").exists()
