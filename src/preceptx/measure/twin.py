"""Retrospective/prospective twin: the RQ2 measurement primitive (roadmap §2.4, §3.3).

Retrospective CPVI scores each handoff with its realised outcome Y. The prospective twin applies the
*same* trained probes at the handoff using only their predictive distributions - no realised Y at
inference - and reports the expected information ``KL(g_cond || g_base)`` in bits. The two live on
the same bits scale, so their agreement (Pearson/Spearman + Bland-Altman) is the H3 test. The no-Y
discipline is structural, not a convention: ``prospective_twin`` takes only the probe distributions,
so it cannot reach the outcome - and its output is invariant to Y by construction (the no-Y test
asserts both the signature and that call-path).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import pearsonr, spearmanr

from preceptx.measure.pvi_cpvi import (
    EPS,
    ProbeConfig,
    pointwise_logprob,
    predictive_distributions,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

# Clip pathological per-instance KL so a single mis-calibrated probe cannot dominate the
# Bland-Altman limits of agreement; the count of clipped instances is itself a calibration flag.
KL_CAP_BITS = 10.0


class TwinAgreement(BaseModel):
    """H3 agreement between the retrospective and prospective per-handoff scores."""

    model_config = ConfigDict(extra="forbid")

    n: int
    pearson_r: float
    spearman_rho: float
    ba_bias: float  # mean(retrospective - prospective)
    ba_loa_low: float  # bias - 1.96 sd
    ba_loa_high: float  # bias + 1.96 sd
    n_kl_capped: int  # prospective instances hitting KL_CAP_BITS (calibration diagnostic)


def retrospective_cpvi(
    p_cond: FloatArray, p_base: FloatArray, y: IntArray, classes: IntArray
) -> FloatArray:
    """Realised-outcome CPVI from precomputed predictive distributions (uses Y)."""
    return pointwise_logprob(p_cond, y, classes) - pointwise_logprob(p_base, y, classes)


def prospective_twin(p_cond: FloatArray, p_base: FloatArray) -> FloatArray:
    """Expected CPVI = per-instance ``KL(g_cond || g_base)`` in bits. Takes no Y."""
    ratio = np.log2((p_cond + EPS) / (p_base + EPS))
    kl: FloatArray = np.sum(p_cond * ratio, axis=1)
    return np.asarray(np.clip(kl, 0.0, KL_CAP_BITS), dtype=np.float64)


def twin_scores(
    e_s: FloatArray, e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig
) -> tuple[FloatArray, FloatArray]:
    """Paired ``(retrospective, prospective)`` per-handoff scores from one shared probe fit."""
    p_cond, p_base, classes = predictive_distributions(e_s, e_m, y, groups, cfg)
    return retrospective_cpvi(p_cond, p_base, y, classes), prospective_twin(p_cond, p_base)


def twin_agreement(retro: FloatArray, prosp: FloatArray) -> TwinAgreement:
    """Correlation and Bland-Altman agreement between the two same-scale score vectors."""
    diff = retro - prosp
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    paired = len(retro) > 1  # correlation is undefined on a single handoff
    return TwinAgreement(
        n=len(retro),
        pearson_r=float(pearsonr(retro, prosp)[0]) if paired else float("nan"),
        spearman_rho=float(spearmanr(retro, prosp)[0]) if paired else float("nan"),
        ba_bias=bias,
        ba_loa_low=bias - 1.96 * sd,
        ba_loa_high=bias + 1.96 * sd,
        n_kl_capped=int(np.sum(prosp >= KL_CAP_BITS)),
    )
