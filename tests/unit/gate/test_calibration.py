from __future__ import annotations

import inspect

import numpy as np
from sklearn.metrics import roc_auc_score

from preceptx.gate.calibration import (
    CalibrationConfig,
    _choose_threshold,
    _oof_scores,
    _orient,
    _platt_reliability,
    _safe_auroc,
    calibrate,
)
from preceptx.gate.statistics import FailStatistic


def test_group_split_prevents_optimistic_auroc() -> None:
    # e_s is CONSTANT within an episode and failure is an episode property: a per-handoff random
    # split lets the probe memorise the shared state and inflates AUROC; a group split does not.
    rng = np.random.default_rng(0)
    n_ep, per, d = 40, 5, 6
    groups = np.repeat(np.arange(n_ep), per)
    e_s = rng.standard_normal((n_ep, d))[groups]  # identical within episode
    e_m = rng.standard_normal((n_ep * per, d))  # message carries nothing about failure
    fail = (rng.random(n_ep) < 0.5).astype(int)[groups]
    cfg = CalibrationConfig()
    grouped = _oof_scores(FailStatistic, e_s, e_m, fail, groups, cfg)
    singleton = np.arange(n_ep * per)  # every handoff its own group -> effectively a random split
    leaked = _oof_scores(FailStatistic, e_s, e_m, fail, singleton, cfg)
    assert roc_auc_score(fail, leaked) - roc_auc_score(fail, grouped) > 0.1


def test_safe_auroc_perfect_and_single_class() -> None:
    assert _safe_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _safe_auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_orient_flips_anti_correlated_score() -> None:
    fail = np.array([0, 0, 1, 1])
    assert _orient(np.array([0.1, 0.2, 0.8, 0.9]), fail) == 1.0
    assert _orient(np.array([0.9, 0.8, 0.2, 0.1]), fail) == -1.0


def test_choose_threshold_respects_budget_and_is_reproducible() -> None:
    score = np.linspace(0.0, 1.0, 100)
    t, firing = _choose_threshold(score, 0.2)
    assert firing <= 0.2 + 1e-9
    assert (t, firing) == _choose_threshold(score, 0.2)  # deterministic
    tied = np.concatenate([np.zeros(70), np.ones(30)])  # a tie mass blocks an intermediate rate
    _, f_tied = _choose_threshold(tied, 0.2)
    assert f_tied <= 0.2 + 1e-9  # steps above the tie rather than overshoot the budget


def test_platt_ece_low_on_informative_and_none_on_single_class() -> None:
    rng = np.random.default_rng(0)
    n = 500
    fail = (rng.random(n) < 0.5).astype(int)
    score = fail + 0.3 * rng.standard_normal(n)  # informative -> Platt recovers good probabilities
    ece, bins = _platt_reliability(score, fail, 10)
    assert ece is not None and 0.0 <= ece < 0.15
    assert sum(b.count for b in bins) == n  # the counts partition the data
    none_ece, none_bins = _platt_reliability(score, np.ones(n, dtype=int), 10)
    assert none_ece is None and none_bins == []


def test_calibrate_target_is_failure_never_cpvi() -> None:
    params = set(inspect.signature(calibrate).parameters)
    assert "cpvi" not in params and "cpvi_scores" not in params  # cannot calibrate against CPVI
