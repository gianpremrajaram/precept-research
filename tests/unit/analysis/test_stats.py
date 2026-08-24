"""DSE-028 analysis primitives: effect sizes, bootstrap CI, corrections, seed sensitivity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import spearmanr

from preceptx.analysis.stats import (
    bootstrap_ci,
    build_provenance,
    cliffs_delta,
    cluster_bootstrap_ci,
    cohens_d,
    correct_pvalues,
    load_analysis_frame,
    overlap_restricted_contrast,
    partial_spearman,
    seed_sensitivity,
)
from preceptx.data.schema import HandoffRecord
from preceptx.data.writer import write_handoffs
from preceptx.measure.featuriser import EncoderConfig
from preceptx.measure.pvi_cpvi import ProbeConfig


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


def test_bootstrap_ci_constant_sample_collapses_to_point() -> None:
    lo, hi = bootstrap_ci(np.full(10, 2.5), n_boot=500)
    assert (lo, hi) == (2.5, 2.5)  # no spread -> BCa undefined; the interval is the point


def test_bootstrap_ci_tiny_sample_falls_back_without_error() -> None:
    lo, hi = bootstrap_ci(np.array([1.0, 3.0]), n_boot=500, seed=0)  # n<3 -> percentile fallback
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= 2.0 <= hi


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
        observation="s",
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


def test_build_provenance_captures_encoder_probe_and_code_identity() -> None:
    prov = build_provenance(EncoderConfig(revision="pinned-rev"), ProbeConfig(c=0.5))
    assert prov.encoder_name == EncoderConfig().name
    assert prov.encoder_revision == "pinned-rev"
    # Both encoder revisions ride on the artefact (DSE-033), so a DSE-022 sensitivity re-run is
    # distinguishable from the primary fit by the artefact alone.
    assert prov.second_encoder == EncoderConfig().second_encoder
    assert prov.second_encoder_revision == EncoderConfig().second_encoder_revision
    assert prov.probe.c == 0.5
    assert len(prov.git_sha) == 40  # full SHA from a real checkout
    assert prov.timestamp  # ISO timestamp present


def test_load_analysis_frame_adds_nullable_failure(tmp_path: object) -> None:
    records = [_rec("a", True), _rec("b", False), _rec("c", None)]
    write_handoffs(records, root=tmp_path, dataset_hash="h0")  # type: ignore[arg-type]
    frame = load_analysis_frame("h0", root=str(tmp_path))
    by_ep = {row.episode_id: row.failure for row in frame.itertuples()}
    assert by_ep["a"] is False  # success -> not failure
    assert by_ep["b"] is True  # failure
    assert by_ep["c"] is None  # unlabelled stays None, never a silent False


def test_cluster_bootstrap_ci_sees_the_between_cluster_variance() -> None:
    # Episode means differ; handoffs within an episode barely do. An iid handoff resample averages
    # the cluster effect away, so its interval must be narrower than the cluster-honest one.
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 25)
    x = np.repeat(rng.normal(0.0, 1.0, 8), 25) + rng.normal(0.0, 0.01, 200)
    lo_h, hi_h = bootstrap_ci(x, n_boot=400)
    lo_c, hi_c = cluster_bootstrap_ci(x, groups, n_boot=400)
    assert (hi_c - lo_c) > (hi_h - lo_h)


def test_cluster_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 10)
    x = rng.normal(0.5, 1.0, 100)
    first = cluster_bootstrap_ci(x, groups, n_boot=400, seed=3)
    assert first == cluster_bootstrap_ci(x, groups, n_boot=400, seed=3)
    assert first[0] <= float(np.mean(x)) <= first[1]


def test_cluster_bootstrap_ci_single_cluster_collapses_to_the_point() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert cluster_bootstrap_ci(x, np.zeros(3, dtype=int)) == (2.0, 2.0)


def test_cluster_bootstrap_ci_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(np.array([1.0]), np.array([0, 1]))


def test_partial_spearman_matches_the_closed_form() -> None:
    """First-order partial rank correlation: residualising must equal the textbook formula."""
    rng = np.random.default_rng(0)
    c = rng.normal(size=50)
    x = c + rng.normal(scale=0.5, size=50)
    y = c + rng.normal(scale=0.5, size=50)
    r_xy, r_xc, r_yc = (float(spearmanr(a, b).statistic) for a, b in ((x, y), (x, c), (y, c)))
    expected = (r_xy - r_xc * r_yc) / math.sqrt((1 - r_xc**2) * (1 - r_yc**2))
    assert partial_spearman(x, y, c) == pytest.approx(expected, abs=1e-9)


def test_partial_spearman_removes_a_pure_control_artefact() -> None:
    """When x and y are monotone functions of the control alone, nothing survives partialling."""
    c = np.arange(40.0)
    assert partial_spearman(2.0 * c, c**2, c) == pytest.approx(0.0, abs=1e-9)


def test_partial_spearman_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        partial_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# --- overlap-restricted length control (DSE-044, PREREGISTRATION section 5) -------------------


def test_overlap_restricted_matches_the_raw_contrast_when_lengths_are_balanced() -> None:
    """With identical length distributions there is nothing to adjust away."""
    covariate = np.array([1.0, 2, 3, 4, 1, 2, 3, 4])
    treated = np.array([False] * 4 + [True] * 4)
    value = np.array([0.0, 0, 0, 0, 1, 1, 1, 1])
    out = overlap_restricted_contrast(value, covariate, treated, n_bins=2, min_per_cell=2)
    assert out.interpretable
    assert out.n_kept == out.n_total == 8
    assert out.delta == pytest.approx(1.0)
    assert out.delta_unrestricted == pytest.approx(1.0)


def test_overlap_restriction_removes_a_difference_that_is_purely_length() -> None:
    """The case the control exists for: the raw gap is length, and nothing but length.

    ``value`` is a deterministic function of length alone, so any Ck-vs-C0 difference is a length
    artefact. The treated arm is shifted short (as C1's cap shifts it) but still overlaps. Compared
    within length strata the difference must vanish, while the unrestricted difference does not.
    """
    covariate = np.array([10.0, 10, 20, 20, 30, 30, 10, 10, 20, 20, 30, 30])
    treated = np.array([False] * 6 + [True] * 6)
    # length drives the outcome entirely; both arms cover the same three lengths
    value = covariate / 10.0
    out = overlap_restricted_contrast(value, covariate, treated, n_bins=3, min_per_cell=2)
    assert out.delta == pytest.approx(0.0)
    assert out.interpretable and out.n_bins == 3


def test_no_overlap_is_reported_rather_than_estimated() -> None:
    """Disjoint length distributions must refuse to produce a number, not extrapolate one."""
    covariate = np.array([1.0, 2, 3, 4, 50, 60, 70, 80])
    treated = np.array([False] * 4 + [True] * 4)
    out = overlap_restricted_contrast(np.arange(8.0), covariate, treated, n_bins=2)
    assert not out.interpretable
    assert np.isnan(out.delta)
    assert out.n_bins == 0 and out.n_kept == 0
    assert not np.isnan(out.delta_unrestricted)  # the unadjusted number still exists
    assert "do not overlap" in out.note


def test_a_stratum_carried_by_one_episode_is_dropped() -> None:
    """min_per_cell is the guard against a thin overlap silently carrying the whole contrast.

    The short stratum holds two reference episodes but only one treated. At the default floor it is
    dropped; at ``min_per_cell=1`` it counts. Same data, and the floor is the only thing between a
    contrast resting on one idiosyncratic episode and one that does not.
    """
    covariate = np.array([10.0, 11.0, 20.0, 21.0, 10.5, 11.5, 20.5])
    treated = np.array([False, False, False, False, True, True, True])
    value = np.arange(7.0)

    strict = overlap_restricted_contrast(value, covariate, treated, n_bins=2, min_per_cell=2)
    assert strict.n_bins == 1 and strict.n_kept == 4

    permissive = overlap_restricted_contrast(value, covariate, treated, n_bins=2, min_per_cell=1)
    assert permissive.n_bins == 2 and permissive.n_kept == 7
    assert permissive.delta != strict.delta


def test_tied_lengths_collapse_strata_and_say_so() -> None:
    """Equal-count bins cannot split identical values; the retained count reports what happened."""
    covariate = np.ones(6)
    treated = np.array([True, True, True, False, False, False])
    out = overlap_restricted_contrast(np.arange(6.0), covariate, treated, n_bins=3)
    assert out.n_bins == 1  # three bins asked for, one formable
    assert out.delta == pytest.approx(out.delta_unrestricted)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        overlap_restricted_contrast(
            np.zeros(3), np.zeros(4), np.array([True, False, True]), n_bins=2
        )
