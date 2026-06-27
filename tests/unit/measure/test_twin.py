from __future__ import annotations

import inspect

import numpy as np
from _synthetic import make_binary

from preceptx.measure.pvi_cpvi import ProbeConfig, predictive_distributions
from preceptx.measure.twin import (
    KL_CAP_BITS,
    prospective_twin,
    retrospective_cpvi,
    twin_agreement,
    twin_scores,
)

CFG = ProbeConfig(n_splits=5)


def test_prospective_twin_signature_has_no_y() -> None:
    assert list(inspect.signature(prospective_twin).parameters) == ["p_cond", "p_base"]


def test_prospective_is_y_free_while_retrospective_varies_with_y() -> None:
    e_s, e_m, y, g = make_binary("informative")
    p_cond, p_base, classes = predictive_distributions(e_s, e_m, y, g, CFG)
    prosp = prospective_twin(p_cond, p_base)
    retro = retrospective_cpvi(p_cond, p_base, y, classes)
    retro_flipped = retrospective_cpvi(p_cond, p_base, 1 - y, classes)
    assert not np.allclose(retro, retro_flipped)  # retrospective DOES use the realised Y
    assert np.array_equal(prosp, prospective_twin(p_cond, p_base))  # prospective is Y-free + stable


def test_twin_agreement_high_on_informative() -> None:
    e_s, e_m, y, g = make_binary("informative")
    retro, prosp = twin_scores(e_s, e_m, y, g, CFG)
    agree = twin_agreement(retro, prosp)
    assert agree.pearson_r > 0.3  # realised and expected information track
    assert abs(agree.ba_bias) < 1.0  # near-zero Bland-Altman bias on matched twins
    assert agree.n == len(y)


def test_twin_agreement_single_handoff_is_nan_not_crash() -> None:
    agree = twin_agreement(np.array([1.0]), np.array([0.5]))  # correlation undefined at n=1
    assert agree.n == 1
    assert np.isnan(agree.pearson_r)
    assert np.isnan(agree.spearman_rho)


def test_kl_cap_applied_and_counted() -> None:
    p_cond = np.array([[0.999999, 0.000001], [0.5, 0.5]])
    p_base = np.array([[0.000001, 0.999999], [0.5, 0.5]])
    val = prospective_twin(p_cond, p_base)
    assert val[0] == KL_CAP_BITS  # ~20-bit raw KL clipped to the cap
    assert val[1] == 0.0  # identical distributions -> zero divergence
    agree = twin_agreement(np.array([1.0, 2.0]), val)
    assert agree.n_kl_capped == 1
