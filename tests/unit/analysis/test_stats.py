"""DSE-028 analysis primitives: effect sizes, bootstrap CI, corrections, seed sensitivity."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from preceptx.analysis.stats import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    correct_pvalues,
    load_analysis_frame,
    seed_sensitivity,
)
from preceptx.data.schema import HandoffRecord
from preceptx.data.writer import write_handoffs


def test_cohens_d_recovers_known_separation() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 2000)  # one SD apart -> d ~ 1
    b = rng.normal(0.0, 1.0, 2000)
    assert 0.85 < cohens_d(a, b) < 1.15
    assert cohens_d(a, a) == 0.0  # identical groups -> no effect


def test_cliffs_delta_spans_minus_one_to_one() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert cliffs_delta(a, b) == 1.0  # a dominates b
    assert cliffs_delta(b, a) == -1.0  # antisymmetric
    assert cliffs_delta(a, a) == 0.0  # ties net to zero


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    x = np.random.default_rng(0).normal(0.0, 1.0, 500)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=0)
    assert lo <= float(np.mean(x)) <= hi
    assert hi - lo < 0.3  # n=500 -> a tight interval on the mean
    assert (lo, hi) == bootstrap_ci(x, n_boot=2000, seed=0)  # seed-reproducible


def test_correct_pvalues_single_is_unchanged_and_both_methods_only_raise() -> None:
    assert correct_pvalues(np.array([0.03]))[0] == 0.03  # nothing to correct against
    raw = np.array([0.01, 0.02, 0.03, 0.04])
    holm = correct_pvalues(raw, method="holm")
    bh = correct_pvalues(raw, method="bh")
    assert np.all(holm >= raw) and np.all(bh >= raw)
    assert np.all(holm >= bh - 1e-12)  # Holm is the more conservative family-wise control


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20).map(np.array),
    st.sampled_from(["holm", "bh"]),
)
def test_correction_never_increases_significance(pvals: np.ndarray, method: str) -> None:
    corrected = correct_pvalues(pvals, method=method)  # type: ignore[arg-type]
    assert np.all(corrected >= pvals - 1e-12)  # corrected p >= raw p, always
    assert np.all(corrected <= 1.0 + 1e-12)


def test_seed_sensitivity_aggregates() -> None:
    s = seed_sensitivity({0: 1.0, 1: 3.0})
    assert s.n_seeds == 2
    assert s.mean == 2.0
    assert abs(s.sd - np.sqrt(2.0)) < 1e-9  # std with ddof=1
    assert s.spread == 2.0


def _rec(episode: str, y: bool | None) -> HandoffRecord:
    return HandoffRecord(
        episode_id=episode,
        step=0,
        condition="C0",
        serialisation="numeric",
        difficulty="hard",
        model="m",
        seed=0,
        state={},
        state_str="s",
        message_raw="r",
        message_delivered="d",
        action={},
        pre_state={},
        post_state={},
        progress=0.0,
        success=bool(y),
        collision=False,
        stuck=False,
        y_terminal_success=y,
    )


def test_load_analysis_frame_adds_nullable_failure(tmp_path: object) -> None:
    records = [_rec("a", True), _rec("b", False), _rec("c", None)]
    write_handoffs(records, root=tmp_path, dataset_hash="h0")  # type: ignore[arg-type]
    frame = load_analysis_frame("h0", root=str(tmp_path))
    by_ep = {row.episode_id: row.failure for row in frame.itertuples()}
    assert by_ep["a"] is False  # success -> not failure
    assert by_ep["b"] is True  # failure
    assert by_ep["c"] is None  # unlabelled stays None, never a silent False
