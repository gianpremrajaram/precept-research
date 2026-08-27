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

import datetime as dt
import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import DegenerateDataWarning, rankdata
from scipy.stats import bootstrap as _scipy_bootstrap
from statsmodels.stats.multitest import multipletests

from preceptx.data.writer import load_dataset
from preceptx.manifest import git_sha
from preceptx.measure.featuriser import EncoderConfig
from preceptx.measure.pvi_cpvi import ProbeConfig

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
    "H1_efficiency": (
        "Step-efficiency degrades C0->C4: Cliff's delta on steps-to-goal per C0-vs-Ck contrast "
        "with a bootstrap CI. Failed episodes sit at the step budget (episodes terminate on "
        "success or budget), so the rank-based delta treats them as the censored-at-budget mass."
    ),
    "H2": (
        "CPVI mediates condition->outcome: episode-level Baron-Kenny mediation of terminal "
        "success through episode-mean CPVI - path a (condition->CPVI), path b/c' "
        "(success ~ condition + CPVI), total c, and the indirect effect a*b per condition with a "
        "percentile-bootstrap CI over episodes. The handoff-level CPVI attenuation of the "
        "condition coefficients is reported as a within-episode diagnostic, not the H2 test."
    ),
    "control_task": (
        "Probe selectivity (DSE-043): CPVI re-estimated against random labels drawn at the "
        "observed base rate, through the same splitter and probe family. Pre-registered is "
        "control CPVI <= 0 - out of fold neither probe generalises and g_cond carries twice the "
        "features, so it overfits the noise harder. Selectivity = mean CPVI - control CPVI is "
        "reported alongside every CPVI summary; PREREGISTRATION section 5's capacity ladder fires "
        "when control CPVI exceeds +0.02 bits or its episode-cluster interval excludes zero above."
    ),
    "length_control": (
        "Message length is confounded with condition by construction (C1 caps it), so CPVI's "
        "relation to outcome is reported with delivered-message token length partialled out "
        "(partial Spearman), and the H2 mediation reports path b both uncontrolled and with "
        "episode-mean message length as a covariate (DSE-044). The second pre-registered control "
        "is the overlap-restricted comparison: episodes are stratified into quantile bins of "
        "episode-mean message length and Ck-vs-C0 differences are taken only inside bins holding "
        "at least min_per_cell episodes of BOTH conditions, size-weighted across bins. It is a "
        "sensitivity analysis, not a clean length-free estimate of the channel effect - C1 shifts "
        "the length distribution by construction, so the overlap region is a non-random subset of "
        "both arms and the restricted contrast generalises only to lengths both arms reach."
    ),
    "signal_decomposition": (
        "Pre-registered secondary analysis (DSE-046): handoffs are split on CPVI at the "
        "**within-condition** median (low = cpvi <= median) and crossed with realised progress, "
        "giving a 2x2 per condition. Two rates over that condition's handoffs are reported with "
        "episode-cluster bootstrap intervals - the absent-signal rate (low CPVI and no progress: "
        "the sender failed to encode) and the unused-signal rate (high CPVI and no progress: the "
        "information was there and the receiver did not act). They are an additive decomposition "
        "of the condition's no-progress rate, which is what separates the two failures a single "
        "correlational number conflates (Eccles et al. 2019, positive signalling vs positive "
        "listening). The median split is within condition so it is not a restatement of the "
        "condition effect, and it is fixed at the observed sample rather than recomputed inside "
        "the bootstrap: the interval covers the rates *given* the pre-registered split rule."
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


class AnalysisProvenance(BaseModel):
    """Who/what produced an analysis artefact (P1-8): encoder, probe family, and code identity.

    Embedded in every persisted analysis result (``RQ1Result``, ``CalibrationReport``) so each
    artefact is self-describing - CLAUDE.md: a result with an unrecorded revision is not a result.
    """

    model_config = ConfigDict(extra="forbid")

    encoder_name: str
    encoder_revision: str
    # The sensitivity-check encoder is recorded even when unused, so a re-run under DSE-022 can be
    # told apart from the primary fit by the artefact alone (DSE-033).
    second_encoder: str
    second_encoder_revision: str
    probe: ProbeConfig
    git_sha: str
    timestamp: str


def build_provenance(encoder: EncoderConfig, probe: ProbeConfig) -> AnalysisProvenance:
    """Assemble the provenance block from the live environment (the one shared constructor)."""
    return AnalysisProvenance(
        encoder_name=encoder.name,
        encoder_revision=encoder.revision,
        second_encoder=encoder.second_encoder,
        second_encoder_revision=encoder.second_encoder_revision,
        probe=probe,
        git_sha=git_sha(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )


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


def cluster_bootstrap_ci(
    x: FloatArray,
    groups: NDArray[np.int_],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``, resampling whole clusters (episodes).

    Handoffs within an episode share a start pose, a trajectory and overlapping next-k label
    windows, so resampling them as if independent understates uncertainty - on E3-local the iid
    handoff interval read roughly half the honest width. The episode is the sampling unit: clusters
    are resampled with replacement and their handoffs pooled per draw. Percentile rather than BCa:
    at pilot cluster counts the draw space is too discrete for the jackknife acceleration.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or len(x) != len(groups):
        raise ValueError("cluster_bootstrap_ci needs a non-empty x aligned to groups")
    ids = np.unique(groups)
    if len(ids) < 2:
        m = float(np.mean(x))
        return m, m  # one cluster: no between-cluster variance to estimate
    by_cluster = [x[groups == g] for g in ids]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        boots[b] = float(np.mean(np.concatenate([by_cluster[i] for i in pick])))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def partial_spearman(x: FloatArray, y: FloatArray, control: FloatArray) -> float:
    """Spearman correlation of ``x`` and ``y`` with ``control`` partialled out (DSE-044).

    "CPVI is just message length" is the first sceptical question the design invites, and C1
    manipulates length directly, so the control belongs in the protocol rather than in a rebuttal.
    Rank-transform all three, residualise ``x`` and ``y`` on the ranked control by least squares,
    and correlate the residuals - the standard partial-rank construction. Returns 0.0 when a
    residual is constant (no variance left to correlate), rather than a NaN that would propagate.
    """
    rx, ry, rc = (rankdata(np.asarray(v, dtype=np.float64)) for v in (x, y, control))
    if len(rx) < 3:
        raise ValueError("partial_spearman needs at least 3 observations")
    design = np.column_stack([np.ones_like(rc), rc])
    ex, ey = (v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (rx, ry))
    # A control that fully explains a rank leaves floating-point dust, not exact zeros; correlating
    # that dust would report a spurious +-1. Compare against the pre-residual spread, not against 0.
    if ex.std() <= 1e-9 * rx.std() or ey.std() <= 1e-9 * ry.std():
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


class OverlapRestrictedContrast(BaseModel):
    """A two-group difference taken only inside covariate strata where both groups are present.

    The second pre-registered length control (PREREGISTRATION section 5, DSE-044). C1 caps message
    length, so length and condition are confounded by construction and the raw Ck-vs-C0 difference
    cannot separate "the channel did this" from "shorter messages did this".

    Read it as an **overlap-restricted, length-adjusted sensitivity analysis**, never as a clean
    estimate of the channel effect with length removed. Restricting to the overlap is exactly what
    makes it informative and exactly what makes it partial: the retained episodes are a non-random
    subset of both arms, so the contrast generalises only to the lengths both arms actually reach,
    and at pilot N the overlap can be thin enough that a couple of idiosyncratic episodes carry it.
    ``n_kept`` against ``n_total`` is therefore part of the result, not diagnostics: a delta read
    without them is unreadable.
    """

    model_config = ConfigDict(extra="forbid")

    n_bins: int  # strata retained (holding min_per_cell of BOTH groups), not strata attempted
    n_kept: int  # episodes inside those strata
    n_total: int  # episodes offered
    delta: float  # size-weighted within-stratum mean(treated) - mean(reference); nan if none
    delta_unrestricted: float  # the same difference over every episode, for comparison
    interpretable: bool  # False when no stratum held both groups
    note: str


def overlap_restricted_contrast(
    value: FloatArray,
    covariate: FloatArray,
    treated: NDArray[np.bool_],
    *,
    n_bins: int = 3,
    min_per_cell: int = 2,
) -> OverlapRestrictedContrast:
    """Stratify on ``covariate``, difference ``value`` within strata holding both groups.

    Quantile bins over the pooled covariate rather than nearest-neighbour matching on purpose: at
    six episodes per condition a caliper match has too little support and can silently collapse the
    comparison to one or two pairs, while coarse strata degrade visibly (``n_bins`` falls) instead
    of quietly. ``min_per_cell`` is the floor per group per stratum, so a stratum carried by a
    single episode is dropped rather than weighted.

    ponytail: equal-count quantile bins, size-weighted. Propensity weighting only if the length
    distributions turn out to overlap so little that binning keeps nothing.
    """
    value = np.asarray(value, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    treated = np.asarray(treated, dtype=bool)
    if not (len(value) == len(covariate) == len(treated)):
        raise ValueError("value, covariate and treated must be the same length")
    if n_bins < 1 or min_per_cell < 1:
        raise ValueError("n_bins and min_per_cell must be >= 1")

    n_total = len(value)
    ref = ~treated
    unrestricted = (
        float(value[treated].mean() - value[ref].mean())
        if treated.any() and ref.any()
        else float("nan")
    )

    # Interior quantile edges only; ties collapse bins, which is why n_bins is reported from what
    # was retained rather than from what was asked for.
    cuts = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges: FloatArray = np.quantile(covariate, cuts).astype(np.float64)
    idx = np.digitize(covariate, edges)

    kept_deltas: list[float] = []
    kept_weights: list[int] = []
    for b in np.unique(idx):
        in_bin = idx == b
        t, r = in_bin & treated, in_bin & ref
        if int(t.sum()) >= min_per_cell and int(r.sum()) >= min_per_cell:
            kept_deltas.append(float(value[t].mean() - value[r].mean()))
            kept_weights.append(int(in_bin.sum()))

    if not kept_deltas:
        return OverlapRestrictedContrast(
            n_bins=0,
            n_kept=0,
            n_total=n_total,
            delta=float("nan"),
            delta_unrestricted=unrestricted,
            interpretable=False,
            note=(
                f"no length stratum held >= {min_per_cell} episodes of both conditions; the two "
                "length distributions do not overlap enough to compare at this N"
            ),
        )

    weights = np.asarray(kept_weights, dtype=np.float64)
    delta = float(np.average(np.asarray(kept_deltas, dtype=np.float64), weights=weights))
    n_kept = int(weights.sum())
    return OverlapRestrictedContrast(
        n_bins=len(kept_deltas),
        n_kept=n_kept,
        n_total=n_total,
        delta=delta,
        delta_unrestricted=unrestricted,
        interpretable=True,
        note=(
            f"overlap-restricted, length-adjusted sensitivity analysis: {n_kept}/{n_total} "
            f"episodes across {len(kept_deltas)} length stratum/strata holding both conditions"
        ),
    )


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
