"""Shared analysis primitives (DSE-028): effect sizes, intervals, corrections, seed sensitivity.

Every RQ analysis imports these so the thesis uses one consistent statistics stack rather than each
experiment re-deriving its own (the defensibility argument in the roadmap's statistical plan). The
deliberate division of labour: this module owns the *generic* numeric helpers (stateless functions
on plain arrays - CLAUDE.md's function-over-class rule), the RQ-specific shaping and modelling lives
in ``src/preceptx/experiments``. Multiple-comparison correction wraps ``statsmodels`` rather than
reimplementing Holm/BH; effect sizes and the bootstrap interval are small enough to keep in-house.

``ANALYSIS_PROTOCOL`` documents which test backs which hypothesis (the AC's "documented analysis
protocol"); the RQ drivers cite it in their reports.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import DegenerateDataWarning
from scipy.stats import bootstrap as _scipy_bootstrap
from statsmodels.stats.multitest import multipletests

from preceptx.data.writer import load_dataset

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Which test backs which hypothesis - the report-citable protocol (roadmap §3 statistical plan).
# Frozen alongside Y/V before the RQ1 main sweep so the analysis is confirmatory, not fished.
ANALYSIS_PROTOCOL: dict[str, str] = {
    "H1": (
        "Outcome degrades C0->C4: MixedLM of outcome on condition (random effects for seed and "
        "episode); per-contrast effect size Cliff's delta with a bootstrap CI; Holm across the "
        "C0-vs-Ck contrasts."
    ),
    "H2": (
        "CPVI mediates condition->outcome: refit the MixedLM with per-handoff CPVI as a covariate "
        "and report the attenuation of the condition coefficients (Baron-Kenny step)."
    ),
    "H3": "Twin agreement (DSE-022): retrospective-vs-prospective correlation and Bland-Altman.",
    "H4": "Proxy tracking (DSE-022): rank correlation and AUROC of each runtime statistic vs CPVI.",
    "seed_sensitivity": (
        "Every headline metric is reported with its across-seed spread (LLM non-determinism, "
        "DSE-003); never a single-seed point estimate."
    ),
}


class SeedSensitivity(BaseModel):
    """Across-seed spread of one metric - the mandatory companion to any LLM-run point estimate."""

    model_config = ConfigDict(extra="forbid")

    n_seeds: int
    mean: float
    sd: float
    spread: float  # max - min across seeds
    per_seed: dict[int, float]


def load_analysis_frame(dataset_hash: str, *, root: str) -> pd.DataFrame:
    """The handoff dataset as an analysis frame: the stored columns plus a nullable ``failure``.

    Thin by design - it reuses ``data.writer.load_dataset`` (the one schema-aware reader) and only
    adds the ``failure = not y_terminal_success`` column every failure analysis needs, leaving None
    where the episode is unlabelled rather than coercing it to a silent False.
    """
    frame = load_dataset(dataset_hash, root=root)
    frame["failure"] = frame["y_terminal_success"].map(
        lambda v: None if v is None else (not bool(v))
    )
    return frame


def cohens_d(a: FloatArray, b: FloatArray) -> float:
    """Standardised mean difference ``(mean a - mean b) / pooled_sd`` (pooled, unbiased dof)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("cohens_d needs at least two observations per group")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0  # no within-group spread: the standardised effect is undefined, report 0
    return float((np.mean(a) - np.mean(b)) / pooled)


def cliffs_delta(a: FloatArray, b: FloatArray) -> float:
    """Cliff's delta in ``[-1, 1]``: ``(#(a>b) - #(a<b)) / (na*nb)`` over all cross pairs.

    Robust, distribution-free - the right effect size for the skewed steps-to-goal and CPVI
    distributions. ponytail: O(na*nb) pairwise sign; fine at analysis scale, swap to the
    sort/rank O(n log n) form only if a sample ever runs to many thousands.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cliffs_delta needs non-empty groups")
    return float(np.sign(a[:, None] - b[None, :]).mean())


def bootstrap_ci(
    x: FloatArray,
    *,
    statistic: Callable[[FloatArray], float] = lambda v: float(np.mean(v)),
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected accelerated (BCa) bootstrap CI for ``statistic``, percentile fallback.

    Distribution-free interval used everywhere uncertainty is reported (CLAUDE.md: intervals, not
    bare significance). BCa corrects the percentile method's small-sample bias and skew (DSE-028
    hardening) - it is the standard for the small, skewed pilot samples here. It is undefined on a
    degenerate sample (no spread, so the jackknife acceleration divides by zero) and unstable below
    three observations; those use the plain percentile interval instead. Deterministic via ``seed``.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("bootstrap_ci needs a non-empty sample")
    if np.ptp(x) == 0.0:
        return float(x[0]), float(x[0])  # constant sample: the interval collapses to the point
    rng = np.random.default_rng(seed)
    if len(x) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateDataWarning)  # a fall-through, not log noise
            try:
                ci = _scipy_bootstrap(
                    (x,),
                    statistic,
                    n_resamples=n_boot,
                    confidence_level=1.0 - alpha,
                    method="BCa",
                    random_state=rng,
                    vectorized=False,
                ).confidence_interval
                if np.isfinite(ci.low) and np.isfinite(ci.high):
                    return float(ci.low), float(ci.high)
            except (DegenerateDataWarning, ValueError):
                pass  # ponytail: BCa undefined here; the percentile branch below is the floor
    boot = np.array(
        [statistic(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)], dtype=np.float64
    )
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def correct_pvalues(pvals: FloatArray, *, method: Literal["holm", "bh"] = "holm") -> FloatArray:
    """Family-wise (Holm) or FDR (Benjamini-Hochberg) corrected p-values for condition contrasts.

    Wraps ``statsmodels.multipletests`` (no point reimplementing it); both methods only ever raise
    a p-value, never lower it - the property the test pins as the leakage guard against fishing.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    if len(pvals) == 0:
        return pvals
    sm_method = "holm" if method == "holm" else "fdr_bh"
    corrected: FloatArray = multipletests(pvals, method=sm_method)[1].astype(np.float64)
    return corrected


def seed_sensitivity(by_seed: Mapping[int, float]) -> SeedSensitivity:
    """Aggregate one metric across seeds into its spread (the LLM-non-determinism companion)."""
    if not by_seed:
        raise ValueError("seed_sensitivity needs at least one seed")
    vals = np.array(list(by_seed.values()), dtype=np.float64)
    return SeedSensitivity(
        n_seeds=len(by_seed),
        mean=float(np.mean(vals)),
        sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        spread=float(np.max(vals) - np.min(vals)),
        per_seed=dict(by_seed),
    )
