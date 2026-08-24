from __future__ import annotations

import logging

import numpy as np
import pytest
from _synthetic import make_binary, make_continuous
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from preceptx.measure.pvi_cpvi import (
    ProbeConfig,
    _cpvi_one,
    _make_splitter,
    control_labels,
    control_task_cpvi,
    cpvi,
    cpvi_continuous,
    cpvi_with_sd,
    estimate,
    shuffled_message_cpvi,
)

CFG = ProbeConfig(n_splits=5)


def test_noise_message_cpvi_near_zero() -> None:
    e_s, e_m, y, g = make_binary("noise")
    assert abs(float(np.mean(cpvi(e_s, e_m, y, g, CFG)))) < 0.06


def test_informative_message_cpvi_positive() -> None:
    e_s, e_m, y, g = make_binary("informative")
    assert float(np.mean(cpvi(e_s, e_m, y, g, CFG))) > 0.1


def test_echo_message_pvi_exceeds_cpvi() -> None:
    e_s, e_m, y, g = make_binary("echo")
    res, _ = estimate(e_s, e_m, y, g, CFG)
    assert res.mean_pvi > res.mean_cpvi  # apparent message value was an echo of the state
    assert res.pvi_cpvi_gap > 0


def test_estimate_reports_auroc_uplift_on_informative() -> None:
    e_s, e_m, y, g = make_binary("informative")
    res, scores = estimate(e_s, e_m, y, g, CFG)
    assert res.auroc_cond is not None
    assert res.auroc_base is not None
    assert res.auroc_train_cond is not None
    assert res.auroc_cond > res.auroc_base  # the message lifts held-out AUROC
    assert res.auroc_train_cond >= res.auroc_cond - 1e-6  # in-sample >= held-out (overfit monitor)
    assert len(scores) == len(y)


def test_split_discipline_no_episode_spans_train_and_test() -> None:
    e_s, e_m, y, g = make_binary("noise")
    splitter, _ = _make_splitter(y, g, CFG, stratified=True)
    for tr, te in splitter.split(np.hstack([e_s, e_m]), y, g):
        assert set(tr.tolist()).isdisjoint(te.tolist())  # no instance scored by its own probe
        assert set(g[tr].tolist()).isdisjoint(g[te].tolist())  # no episode in both folds


def test_continuous_informative_positive_and_noise_near_zero() -> None:
    e_s, e_m, y, g = make_continuous("informative")
    assert float(np.mean(cpvi_continuous(e_s, e_m, y, g, CFG))) > 0.05
    e_s, e_m, y, g = make_continuous("noise")
    assert float(np.mean(cpvi_continuous(e_s, e_m, y, g, CFG))) < 0.05


def test_ridge_regulariser_tracks_config_c() -> None:
    e_s, e_m, y, g = make_continuous("informative")
    strong = float(np.mean(cpvi_continuous(e_s, e_m, y, g, ProbeConfig(c=0.001))))
    weak = float(np.mean(cpvi_continuous(e_s, e_m, y, g, ProbeConfig(c=1000.0))))
    assert strong < weak  # small c -> large alpha -> shrunk fit -> less recovered information


def test_heteroscedastic_continuous_is_reserved() -> None:
    e_s, e_m, y, g = make_continuous("noise", n=40)
    with pytest.raises(NotImplementedError, match="heteroscedastic"):
        cpvi_continuous(e_s, e_m, y, g, ProbeConfig(variance_model="heteroscedastic"))


def test_leave_one_group_out_when_n_splits_none() -> None:
    e_s, e_m, y, g = make_binary("informative", n=40)  # 10 episodes
    scores = cpvi(e_s, e_m, y, g, ProbeConfig(n_splits=None))
    assert len(scores) == len(y)
    assert np.all(np.isfinite(scores))


def test_ungrouped_fallback_warns(caplog: pytest.LogCaptureFixture) -> None:
    e_s, e_m, y, _ = make_binary("noise", n=80)
    with caplog.at_level(logging.WARNING):
        cpvi(e_s, e_m, y, None, CFG)
    assert any("episode groups" in r.message for r in caplog.records)


def test_shuffled_message_cpvi_collapses_to_zero() -> None:
    # RD-15 manipulation check: an informative message scores CPVI > 0, but permuting messages
    # within condition decouples them from their handoff and the mean CPVI must collapse toward 0.
    e_s, e_m, y, g = make_binary("informative")
    real = float(np.mean(cpvi(e_s, e_m, y, g, CFG)))
    conditions = np.zeros(len(y), dtype=int)  # single condition: permute across all rows
    null = shuffled_message_cpvi(
        e_s, e_m, y, g, conditions, CFG, rng=np.random.default_rng(0), n_perm=15
    )
    assert len(null) == 15
    assert real > 0.1  # the real informative message carries signal
    assert abs(float(np.mean(null))) < 0.06  # the permuted null collapses toward zero
    assert float(np.mean(null)) < real  # and sits well below the real signal


@settings(max_examples=15, deadline=None)
@given(p=st.floats(0.25, 0.75), seed=st.integers(0, 50))
def test_cpvi_finite_across_class_balance(p: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, d = 80, 6
    e_s = rng.standard_normal((n, d))
    e_m = rng.standard_normal((n, d))
    y = (rng.random(n) < p).astype(int)
    assume(len(np.unique(y)) == 2)  # both classes needed for stratified group folds
    g = np.repeat(np.arange(n // 4), 4)[:n].astype(int)
    assert np.all(np.isfinite(cpvi(e_s, e_m, y, g, ProbeConfig(n_splits=4))))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mlp_probe_path_runs_and_is_finite() -> None:
    e_s, e_m, y, g = make_binary("informative", n=60)
    cfg = ProbeConfig(probe="mlp", mlp_hidden=8, max_iter=200, n_splits=3)
    scores = cpvi(e_s, e_m, y, g, cfg)
    assert len(scores) == len(y)
    assert np.all(np.isfinite(scores))


def test_n_splits_below_two_is_rejected() -> None:
    with pytest.raises(ValidationError, match="n_splits"):
        ProbeConfig(n_splits=1)


# --- DSE-043: control tasks and probe selectivity ---------------------------------------------


def test_control_task_cpvi_near_zero_and_selectivity_positive() -> None:
    """A well-specified probe cannot manufacture information from random labels."""
    e_s, e_m, y, g = make_binary("informative")
    real = float(np.mean(cpvi(e_s, e_m, y, g, CFG)))
    control = float(np.mean(control_task_cpvi(e_s, e_m, y, g, CFG)))
    assert abs(control) < 0.06  # indistinguishable from zero at this capacity
    assert real - control > 0.1  # selectivity: the real score is not probe artefact


def test_over_capacity_probe_shows_control_cpvi_above_zero() -> None:
    """The check detects what it is meant to detect: capacity manufacturing information.

    An almost-unregularised probe on few samples at high dimension scores random labels well above
    zero, which is exactly the reading PREREGISTRATION §5's capacity ladder fires on.
    """
    rng = np.random.default_rng(0)
    n, d = 30, 64
    e_s, e_m = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    y = rng.integers(0, 2, n)
    g = np.repeat(np.arange(6), 5).astype(int)
    over = ProbeConfig(n_splits=3, c=1e4)
    assert float(np.mean(control_task_cpvi(e_s, e_m, y, g, over))) > 0.1


def test_control_labels_are_seed_reproducible_and_match_the_base_rate() -> None:
    y = np.array([0] * 30 + [1] * 70)
    a, b = control_labels(y, 7), control_labels(y, 7)
    assert np.array_equal(a, b)  # determinism
    assert not np.array_equal(a, control_labels(y, 8))  # and the seed actually varies it
    assert abs(float(a.mean()) - 0.7) < 0.15  # drawn at the observed base rate


def test_estimate_reports_control_and_selectivity() -> None:
    e_s, e_m, y, g = make_binary("informative")
    res, _ = estimate(e_s, e_m, y, g, CFG)
    assert res.control_mean_cpvi is not None
    assert res.selectivity is not None
    assert res.selectivity == pytest.approx(res.mean_cpvi - res.control_mean_cpvi)


# --- DSE-044: repeated cross-fit stabilisation -------------------------------------------------


def test_n_repeats_one_reproduces_the_unrepeated_estimator() -> None:
    """Repeat 0 is the canonical fold assignment, so the default cannot shift any recorded score."""
    e_s, e_m, y, g = make_binary("informative")
    mean, sd = cpvi_with_sd(e_s, e_m, y, g, ProbeConfig(n_splits=5, n_repeats=1))
    assert np.array_equal(mean, cpvi(e_s, e_m, y, g, ProbeConfig(n_splits=5)))
    assert np.all(sd == 0.0)  # one repeat has no spread


def test_repeated_cross_fits_reduce_fold_noise() -> None:
    """Averaging over fold assignments damps the per-handoff noise a single cross-fit carries.

    Each draw uses a disjoint set of fold seeds, so the comparison is one cross-fit against the
    mean of five - the estimator DSE-044 actually swaps in, not a reseeding of the same folds.
    """
    e_s, e_m, y, g = make_binary("noise")  # no real signal: what moves is the fold assignment
    cfg = ProbeConfig(n_splits=5)
    singles = [_cpvi_one(e_s, e_m, y, g, cfg, s) for s in (1, 2, 3)]
    fives = [
        np.mean([_cpvi_one(e_s, e_m, y, g, cfg, f) for f in block], axis=0)
        for block in ((10, 11, 12, 13, 14), (20, 21, 22, 23, 24), (30, 31, 32, 33, 34))
    ]
    assert np.std(fives, axis=0).mean() < np.std(singles, axis=0).mean()


def test_repeat_spread_is_reported_per_handoff() -> None:
    e_s, e_m, y, g = make_binary("informative")
    mean, sd = cpvi_with_sd(e_s, e_m, y, g, ProbeConfig(n_splits=5, n_repeats=4))
    assert len(sd) == len(mean) == len(y)
    assert sd.mean() > 0.0  # distinct fold assignments really do disagree
    assert np.all(np.isfinite(sd))
