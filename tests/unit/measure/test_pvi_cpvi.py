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
    _make_splitter,
    cpvi,
    cpvi_continuous,
    estimate,
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
