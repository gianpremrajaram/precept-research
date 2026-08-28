"""RQ2 measurement analysis (DSE-022): is the primitive trustworthy, and on which label?

Four arms over the RQ1 episodes, all offline - this module runs no grid and issues no model call.

- **H3, twin agreement.** Retrospective CPVI (scored against the realised Y) and its prospective
  twin (the same probes' expected information, taking no Y at inference) sit on one bits scale, so
  their agreement is the test that the runtime-computable half of the primitive tracks the offline
  half. Reported pooled and per condition, from **one** shared probe fit - refitting within a
  condition at pilot N would measure the fold split, not the agreement.
- **H4, proxy tracking.** Each target-free runtime statistic (`info`, `fail`, `cosine`) scored
  out-of-fold, then its rank correlation with CPVI and its AUROC for low CPVI and for failure.
- **Encoder sensitivity.** The whole score pipeline re-run under the pre-registered second encoder;
  what is reported is the rank correlation of the per-handoff CPVI across encoders.
- **Label comparison.** The four Y labels scored side by side under one pre-declared rule.

**The leakage null runs on every tracking and label number (RD-15, PREREGISTRATION section 8).**
Permuting messages *within condition* leaves each handoff its own state, its own Y and its own
condition, and takes only the message it actually received. Whatever CPVI survives that is the
identity component: condition-level message style plus per-condition progress base rates make the
condition tag alone predictive. On the three datasets recorded to date the identity component
carried 78-96% of the C0-C4 CPVI contrast, so a proxy that "tracks CPVI" may be tracking the tag.
Every rho and every mean CPVI here is therefore reported three ways - real, null, and the
leakage-corrected difference with an episode-cluster interval. RQ1 applies this null to the pooled
mean only; applying it to a *correlation* is what makes H4 falsifiable.

**This measures the proxy-CPVI relationship; it does not calibrate on it.** The gate's threshold is
chosen against realised outcomes in `gate/calibration.py` and never against CPVI (CLAUDE.md, the
circularity guard). H4 asks a descriptive question - does the cheap runtime score order handoffs the
way the expensive offline one does - and its answer is an input to the write-up, not to a threshold.

**Y is frozen and this analysis does not re-open it.** RQ1 ran on `y_binary_progress`
(PREREGISTRATION section 4), and the register records that re-choosing it "would rescue the gate
without touching the defect, which is the forbidden move". The label comparison here is therefore a
**pre-declared robustness check** on that frozen choice plus the selection input for the RQ3b gate
target, which is not yet frozen. It cannot retroactively re-point RQ1, and the rule it applies is
fixed in this module's constants rather than chosen after the numbers land. Same for the encoder:
the primary is pinned, and the rule below can only ever *flag* a re-freeze, never perform one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from preceptx.analysis.figures import ci_plot, scatter_plot, series_plot
from preceptx.analysis.stats import AnalysisProvenance, build_provenance, cluster_bootstrap_ci
from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord
from preceptx.gate.calibration import CalibrationConfig, _oof_scores
from preceptx.gate.statistics import (
    CosineStatistic,
    FailStatistic,
    InfoStatistic,
    Statistic,
    episode_groups,
    failure_label,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, cpvi, cpvi_continuous
from preceptx.measure.twin import TwinAgreement, twin_agreement, twin_scores

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

YKind = Literal["binary", "multiclass", "continuous"]
YStatus = Literal["ok", "unavailable", "degenerate"]

#: The four labels of PREREGISTRATION section 4, in register order. ``y_binary_progress`` is the
#: frozen primary; the other three are secondaries the register already defines, so this comparison
#: introduces no new outcome variable - it scores the ones already declared.
PRIMARY_Y = "y_binary_progress"
Y_LABELS: dict[str, YKind] = {
    "y_binary_progress": "binary",
    "y_continuous_displacement": "continuous",
    "y_discrete_config": "multiclass",
    "y_terminal_success": "binary",
}

#: Expected sign of each statistic's correlation with CPVI, **declared rather than fitted**. All
#: three are risk-shaped: entropy up means the probe is less certain about Y, P(fail) up means the
#: handoff looks doomed, cosine up means the message merely echoes the state - each should move
#: opposite to information. `gate/calibration.py` derives an orientation from the failure label
#: because a gate must act; a descriptive analysis that picked the sign making its own AUROC exceed
#: 0.5 would be fitting the direction to the data. An AUROC below 0.5 here means the declared
#: direction was wrong, which is the finding, not a bug. ``fail`` is the least certain of the three:
#: it is a failure probe, and CPVI is about progress, so its sign is a design intent, not a theorem.
DECLARED_ORIENTATION: dict[str, float] = {"info": -1.0, "fail": -1.0, "cosine": -1.0}

# --- The decision rule, fixed here before any RQ1 outcome was read (methodology D24) ------------
_MIN_COVERAGE = 0.95  # a label defined on fewer handoffs than this is not a headline candidate
_MIN_MINORITY_SHARE = 0.10  # 0/375 terminal successes is a label no probe can be scored on
_TIE_BAND = 0.05  # rho differences inside this band go to the next criterion, not to the leader
_ENCODER_INSTABILITY_RHO = 0.50  # below this the two encoders disagree enough to flag a re-freeze


class RQ2Config(BaseModel):
    """Analysis knobs. Defaults track PREREGISTRATION section 5 so RQ2 scores what RQ1 scored."""

    model_config = ConfigDict(extra="forbid")

    probe: ProbeConfig = Field(default_factory=lambda: ProbeConfig(n_repeats=5))
    n_boot: int = Field(default=2000, ge=100)  # episode-cluster interval on per-handoff means
    # Each draw of the tracking bootstrap recomputes two rank correlations rather than one mean, so
    # it is the costly one; 500 draws is enough for a two-decimal interval on a correlation.
    n_boot_track: int = Field(default=500, ge=50)
    n_shuffle: int = Field(default=20, ge=0)  # within-condition permutations; 0 disables the null
    alpha: float = Field(default=0.05, gt=0, lt=1)
    low_cpvi_quantile: float = Field(default=0.25, gt=0, lt=1)  # "low CPVI" = this bottom quantile
    seed: int = Field(default=0, ge=0)


class TwinReport(BaseModel):
    """H3 agreement over one slice of the handoffs (``scope`` is ``pooled`` or a condition)."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    agreement: TwinAgreement


class ProxyTracking(BaseModel):
    """H4 for one runtime statistic: how well it orders handoffs the way CPVI does.

    ``spearman_cpvi`` is the headline number the ticket asks for and is *not* interpretable alone:
    ``spearman_shuffled`` is the same correlation against CPVI recomputed from within-condition
    permuted messages, so it is what the statistic scores by tracking condition identity. The
    corrected difference and its episode-cluster interval are the message-content claim.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    n: int
    declared_orientation: float  # +1/-1, from DECLARED_ORIENTATION; never fitted
    spearman_cpvi: float
    spearman_ci: tuple[float, float]
    spearman_shuffled: float
    spearman_shuffled_ci: tuple[float, float]
    spearman_corrected: float  # real - null: the part not explained by the condition tag
    spearman_corrected_ci: tuple[float, float]
    auroc_low_cpvi: float | None  # under the declared orientation; None if one class
    auroc_failure: float | None


class YComparison(BaseModel):
    """One label scored under the pre-declared rule. A label that cannot be scored keeps its row.

    ``status`` mirrors the RQ3a convention: an unscoreable label reports ``unavailable`` or
    ``degenerate`` with a ``reason``, never ``0.0`` and never a dropped row, because a missing row
    reads as "not considered" when the truth is "considered and unusable".
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    kind: YKind
    status: YStatus
    reason: str | None = None
    n: int = 0
    coverage: float = 0.0  # share of handoffs where the label is populated
    minority_share: float | None = None  # categorical only; None for the continuous label
    mean_cpvi: float = float("nan")
    mean_cpvi_shuffled: float = float("nan")
    corrected_mean_cpvi: float = float("nan")
    corrected_ci: tuple[float, float] = (float("nan"), float("nan"))
    encoder_rho: float | None = None  # per-handoff CPVI rank correlation across the two encoders
    # H3 agreement for THIS label: the gate has to act with no realised Y, so a label whose
    # prospective twin does not track its retrospective score cannot be gated on. ``None`` for the
    # continuous label, which has no implemented twin - a fact that ranks it last, correctly.
    twin_rho: float | None = None
    admissible: bool = False


class EncoderComparison(BaseModel):
    """The pre-registered sensitivity encoder against the primary, on the frozen label."""

    model_config = ConfigDict(extra="forbid")

    primary_name: str
    primary_revision: str
    second_name: str
    second_revision: str
    ran: bool  # False when no second featuriser was available (the sensitivity arm is optional)
    primary_mean_cpvi: float = float("nan")
    second_mean_cpvi: float = float("nan")
    rho_primary_second: float = float("nan")  # per-handoff CPVI agreement across encoders
    label_ranking_invariant: bool = False  # do both encoders admit the same set of labels?


class RQ2Result(BaseModel):
    """The full RQ2 analysis, ready to persist and to drive the figures/tables."""

    model_config = ConfigDict(extra="forbid")

    dataset_hash: str
    n_handoffs: int
    primary_y: str  # the frozen label the twin and proxy arms were scored on
    provenance: AnalysisProvenance
    twin: list[TwinReport]
    proxies: list[ProxyTracking]
    labels: list[YComparison]
    encoders: EncoderComparison
    recommended_y: str | None  # None is a reportable outcome, not a failure
    recommended_encoder: str
    recommendation_note: str
    figures: dict[str, str] = Field(default_factory=dict)


def _statistics() -> dict[str, Callable[[], Statistic]]:
    """Fresh factories for the three runtime statistics, keyed by their stable ``Statistic.key``."""
    return {"info": InfoStatistic, "fail": FailStatistic, "cosine": CosineStatistic}


def _rho(a: FloatArray, b: FloatArray) -> float:
    """Spearman rho, 0.0 on degenerate input - the convention ``partial_spearman`` already uses."""
    if len(a) < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return 0.0
    value = float(spearmanr(a, b)[0])
    return 0.0 if not np.isfinite(value) else value


def _percentile_ci(values: FloatArray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _cluster_resamples(groups: IntArray, *, n_boot: int, seed: int) -> Iterator[IntArray]:
    """Row indices for each episode-cluster bootstrap draw (whole episodes, with replacement).

    Handoffs inside an episode share a start pose and overlapping label windows, so the episode is
    the sampling unit here exactly as it is in ``cluster_bootstrap_ci``; this yields the indices
    instead of a mean so a *paired* statistic (real minus null on the same draw) can be built.
    """
    ids = np.unique(groups)
    by_cluster = [np.flatnonzero(groups == g) for g in ids]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        yield np.concatenate([by_cluster[i] for i in pick])


def _y_values(
    records: list[HandoffRecord], label: str
) -> tuple[np.ndarray[Any, Any], IntArray] | None:
    """``(values, row mask)`` for one label over ``records``, or ``None`` if never populated.

    The mask is returned rather than applied because ``e_s``/``e_m``/groups/conditions all have to
    be subset the same way; a label with partial coverage is scored on the rows it has.
    """
    raw = [getattr(r, label) for r in records]
    mask = np.array([v is not None for v in raw], dtype=bool)
    if not mask.any():
        return None
    kept = [v for v in raw if v is not None]
    kind = Y_LABELS[label]
    dtype = np.float64 if kind == "continuous" else np.int_
    return np.array(kept, dtype=dtype), np.flatnonzero(mask)


def _cpvi_scores(
    e_s: FloatArray,
    e_m: FloatArray,
    values: np.ndarray[Any, Any],
    kind: YKind,
    groups: IntArray,
    probe: ProbeConfig,
) -> FloatArray:
    """Per-handoff CPVI for any of the four labels: Gaussian if continuous, else discrete."""
    if kind == "continuous":
        return cpvi_continuous(e_s, e_m, values.astype(np.float64), groups, probe)
    return cpvi(e_s, e_m, values.astype(int), groups, probe)


def _null_cpvi(
    e_s: FloatArray,
    e_m: FloatArray,
    values: np.ndarray[Any, Any],
    kind: YKind,
    groups: IntArray,
    conditions: NDArray[Any],
    cfg: RQ2Config,
) -> FloatArray:
    """Per-handoff CPVI under within-condition message permutation, averaged over permutations.

    ``pvi_cpvi.shuffled_message_cpvi`` returns per-permutation *means*, which cannot be paired back
    to a handoff; the corrected interval and every corrected correlation here need the per-handoff
    null, so the permutation is rebuilt (and a unit test pins it against that function's null mean).
    Cross-fit repeats are dropped inside the null: averaging over ``n_shuffle`` permutations already
    damps the fold noise that ``n_repeats`` exists to damp, so repeating both multiplies cost for
    nothing.
    """
    probe = cfg.probe.model_copy(update={"n_repeats": 1})
    rng = np.random.default_rng(cfg.seed)
    base = np.arange(len(values))
    unique_conditions = np.unique(conditions)
    draws = np.empty((cfg.n_shuffle, len(values)), dtype=np.float64)
    for p in range(cfg.n_shuffle):
        perm = base.copy()
        for c in unique_conditions:
            block = np.flatnonzero(conditions == c)
            perm[block] = rng.permutation(block)
        draws[p] = _cpvi_scores(e_s, e_m[perm], values, kind, groups, probe)
    return np.asarray(draws.mean(axis=0), dtype=np.float64)


def twin_reports(
    e_s: FloatArray,
    e_m: FloatArray,
    y: IntArray,
    groups: IntArray,
    conditions: NDArray[Any],
    cfg: RQ2Config,
) -> tuple[list[TwinReport], FloatArray, FloatArray]:
    """H3 pooled and per condition, plus the paired score vectors the figure is drawn from."""
    retro, prosp = twin_scores(e_s, e_m, y, groups, cfg.probe)
    reports = [TwinReport(scope="pooled", agreement=twin_agreement(retro, prosp))]
    for c in sorted(np.unique(conditions).tolist()):
        idx = np.flatnonzero(conditions == c)
        if len(idx) > 1:  # a single handoff has no correlation to report
            reports.append(
                TwinReport(scope=str(c), agreement=twin_agreement(retro[idx], prosp[idx]))
            )
    return reports, retro, prosp


def _auroc(label: IntArray, score: FloatArray) -> float | None:
    """AUROC, or ``None`` when the label is single-class (the honest answer, not 0.5)."""
    if len(np.unique(label)) < 2:
        return None
    return float(roc_auc_score(label, score))


def proxy_tracking(
    records: list[HandoffRecord],
    e_s: FloatArray,
    e_m: FloatArray,
    groups: IntArray,
    real: FloatArray,
    null: FloatArray | None,
    cfg: RQ2Config,
) -> list[ProxyTracking]:
    """H4 for all three statistics, each scored out of fold and corrected by the leakage null.

    Scored against the frozen primary label only. A per-label sweep would be 3x4 scorings to answer
    a question nobody asked: these statistics are properties of the *statistic*, and which of them
    tracks CPVI does not change which label CPVI should be computed on.
    """
    cal = CalibrationConfig(probe=cfg.probe)
    fail = failure_label(records)
    low = (real <= float(np.quantile(real, cfg.low_cpvi_quantile))).astype(int)
    null_scores = np.zeros_like(real) if null is None else null
    out: list[ProxyTracking] = []
    for key, factory in _statistics().items():
        stat_label = factory().label(records)
        raw = _oof_scores(factory, e_s, e_m, stat_label, groups, cal)
        # The declared orientation turns each statistic into a *risk* (higher = less information),
        # which is the direction both AUROCs are asked in. It is applied, never chosen.
        risk = -DECLARED_ORIENTATION[key] * raw
        rho_real, rho_null = _rho(raw, real), _rho(raw, null_scores)
        boots = np.array(
            [
                [_rho(raw[i], real[i]), _rho(raw[i], null_scores[i])]
                for i in _cluster_resamples(groups, n_boot=cfg.n_boot_track, seed=cfg.seed)
            ],
            dtype=np.float64,
        )
        out.append(
            ProxyTracking(
                key=key,
                n=len(raw),
                declared_orientation=DECLARED_ORIENTATION[key],
                spearman_cpvi=rho_real,
                spearman_ci=_percentile_ci(boots[:, 0], cfg.alpha),
                spearman_shuffled=rho_null,
                spearman_shuffled_ci=_percentile_ci(boots[:, 1], cfg.alpha),
                spearman_corrected=rho_real - rho_null,
                spearman_corrected_ci=_percentile_ci(boots[:, 0] - boots[:, 1], cfg.alpha),
                auroc_low_cpvi=_auroc(low, risk),
                auroc_failure=_auroc(fail, risk),
            )
        )
    return out


def _minority_share(values: np.ndarray[Any, Any], kind: YKind) -> float | None:
    if kind == "continuous":
        return None
    counts = np.unique(values, return_counts=True)[1]
    return float(counts.min() / counts.sum())


def compare_labels(
    records: list[HandoffRecord],
    e_s: FloatArray,
    e_m: FloatArray,
    conditions: NDArray[Any],
    cfg: RQ2Config,
    *,
    second: tuple[FloatArray, FloatArray] | None = None,
) -> tuple[list[YComparison], dict[str, tuple[FloatArray, FloatArray | None]]]:
    """Score all four labels under the pre-declared rule; every label keeps a row (see YComparison).

    ``second`` is the second encoder's ``(e_s, e_m)`` when the sensitivity arm ran; it supplies
    ``encoder_rho``, the criterion a label is ranked on, so a label whose per-handoff CPVI ordering
    flips with the encoder cannot win on a larger effect.

    Also returns the per-handoff ``(real, null)`` CPVI vectors per scored label, so the caller does
    not pay for the primary label's permutation null a second time - it is the single most expensive
    computation in this module (``n_shuffle`` full cross-fits).
    """
    groups_all = episode_groups(records)
    out: list[YComparison] = []
    scored: dict[str, tuple[FloatArray, FloatArray | None]] = {}
    for label, kind in Y_LABELS.items():
        found = _y_values(records, label)
        if found is None:
            out.append(
                YComparison(
                    label=label, kind=kind, status="unavailable", reason="label never populated"
                )
            )
            continue
        values, rows = found
        coverage = len(rows) / len(records)
        minority = _minority_share(values, kind)
        degenerate = (
            "constant label"
            if (np.ptp(values) == 0.0 if kind == "continuous" else len(np.unique(values)) < 2)
            else None
        )
        if degenerate is not None:
            out.append(
                YComparison(
                    label=label,
                    kind=kind,
                    status="degenerate",
                    reason=degenerate,
                    n=len(rows),
                    coverage=coverage,
                    minority_share=minority,
                )
            )
            continue
        e_s_k, e_m_k, g_k, c_k = e_s[rows], e_m[rows], groups_all[rows], conditions[rows]
        real = _cpvi_scores(e_s_k, e_m_k, values, kind, g_k, cfg.probe)
        null = _null_cpvi(e_s_k, e_m_k, values, kind, g_k, c_k, cfg) if cfg.n_shuffle else None
        corrected = real if null is None else real - null
        lo, hi = cluster_bootstrap_ci(
            corrected, g_k, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed
        )
        rho_enc = (
            None
            if second is None
            else _rho(
                real,
                _cpvi_scores(second[0][rows], second[1][rows], values, kind, g_k, cfg.probe),
            )
        )
        # The reported mean uses the register's repeated cross-fit (PREREGISTRATION section 5); the
        # twin pair must come from ONE shared fit, so it is estimated separately rather than reusing
        # ``real``. Two estimators of the same quantity, each doing the thing its contract requires.
        twin_rho = None
        if kind != "continuous":
            retro_k, prosp_k = twin_scores(e_s_k, e_m_k, values.astype(int), g_k, cfg.probe)
            twin_rho = _rho(retro_k, prosp_k)
        scored[label] = (real, null)
        out.append(
            YComparison(
                label=label,
                kind=kind,
                status="ok",
                n=len(rows),
                coverage=coverage,
                minority_share=minority,
                mean_cpvi=float(np.mean(real)),
                mean_cpvi_shuffled=float("nan") if null is None else float(np.mean(null)),
                corrected_mean_cpvi=float(np.mean(corrected)),
                corrected_ci=(lo, hi),
                encoder_rho=rho_enc,
                twin_rho=twin_rho,
                admissible=(
                    coverage >= _MIN_COVERAGE
                    and (minority is None or minority >= _MIN_MINORITY_SHARE)
                    and lo > 0.0
                ),
            )
        )
    return out, scored


def _rank_labels(labels: list[YComparison]) -> str | None:
    """The pre-declared lexicographic ranking over admissible labels (see the module docstring).

    Criteria in order: encoder-invariance of the per-handoff CPVI ordering, then the label's own
    twin agreement, then - only inside the ``_TIE_BAND``, where the two above cannot separate - the
    corrected mean CPVI. Effect size is deliberately *last*: leading with it would select the label
    carrying the largest number, which is the move the freeze exists to prevent. The two criteria
    ahead of it are both about whether the measurement is *the same measurement* under a different
    encoder and without the realised outcome - which is what a gate target has to be.
    """
    admissible = [y for y in labels if y.admissible]
    if not admissible:
        return None
    ranked = sorted(
        admissible,
        key=lambda y: (
            round((y.encoder_rho if y.encoder_rho is not None else 0.0) / _TIE_BAND),
            round((y.twin_rho if y.twin_rho is not None else -1.0) / _TIE_BAND),
            y.corrected_mean_cpvi,
        ),
        reverse=True,
    )
    return ranked[0].label


def recommend(
    labels: list[YComparison], encoders: EncoderComparison
) -> tuple[str | None, str, str]:
    """``(recommended label, recommended encoder, note)`` under the rules fixed in this module."""
    winner = _rank_labels(labels)
    unstable = encoders.ran and encoders.rho_primary_second < _ENCODER_INSTABILITY_RHO
    encoder = encoders.primary_name
    if winner is None:
        note = (
            "No label is admissible: none clears coverage >= "
            f"{_MIN_COVERAGE:.2f}, minority share >= {_MIN_MINORITY_SHARE:.2f} and a "
            "leakage-corrected mean CPVI whose episode-cluster interval excludes zero. This is a "
            "reportable outcome - it says the message content the probes recover is not separable "
            f"from condition identity on this dataset - and it leaves the frozen {PRIMARY_Y} in "
            "place for RQ1 and leaves the RQ3b gate target unselected."
        )
    else:
        note = (
            f"Recommended gate target for RQ3b: {winner}. Chosen by the rule declared in "
            "experiments/rq2.py before any RQ1 outcome was read - admissibility first, then "
            "encoder-invariance, then the label's own twin agreement, and corrected effect size "
            f"only as a tie-break inside {_TIE_BAND}. RQ1's frozen primary Y ({PRIMARY_Y}) is "
            "unchanged by this either way; the register forbids re-pointing it after results."
        )
    if unstable:
        note += (
            f" ENCODER FLAG: per-handoff CPVI agrees across encoders at rho="
            f"{encoders.rho_primary_second:.2f}, below the declared "
            f"{_ENCODER_INSTABILITY_RHO:.2f}. "
            "The primary encoder still stands - this rule can only flag a re-freeze, never perform "
            "one - but the sensitivity result needs reporting as a limitation."
        )
    return winner, encoder, note


def analyse_rq2(
    records: list[HandoffRecord],
    featuriser: Featuriser,
    *,
    dataset_hash: str,
    second_featuriser: Featuriser | None = None,
    cfg: RQ2Config | None = None,
) -> tuple[RQ2Result, pd.DataFrame]:
    """Score H3, H4, the label comparison and the encoder check on one RQ1 dataset.

    Returns the result plus the per-handoff score frame (the join key downstream analyses need:
    retrospective CPVI, its prospective twin, the leakage null and each runtime statistic, all row-
    aligned to ``records``). ``second_featuriser`` omitted means the sensitivity arm is skipped and
    said so in ``EncoderComparison.ran`` - it is not silently reported as agreement.
    """
    cfg = cfg or RQ2Config()
    if not records:
        raise ConfigError("analyse_rq2 called with no records")
    primary = _y_values(records, PRIMARY_Y)
    if primary is None or len(primary[1]) != len(records):
        raise ConfigError(
            f"RQ2 needs {PRIMARY_Y} on every handoff (run the DSE-009 labeller before analysing)"
        )
    y = primary[0].astype(int)
    if len(np.unique(y)) < 2:
        raise ConfigError(f"RQ2 needs both classes of {PRIMARY_Y} to fit probes")

    e_s, e_m = featuriser.featurise(records)
    groups = episode_groups(records)
    conditions = np.array([r.condition for r in records])

    twin, retro, prosp = twin_reports(e_s, e_m, y, groups, conditions, cfg)
    second = None if second_featuriser is None else second_featuriser.featurise(records)
    labels, scored = compare_labels(records, e_s, e_m, conditions, cfg, second=second)
    # The primary label is guaranteed present and "ok" by the guards above, and its permutation
    # null is the most expensive thing this module computes - so H4 reads it rather than repeating.
    real, null = scored[PRIMARY_Y]
    proxies = proxy_tracking(records, e_s, e_m, groups, real, null, cfg)
    encoders = _encoder_comparison(featuriser.cfg, second, y, groups, labels, cfg, real)
    winner, encoder, note = recommend(labels, encoders)

    frame = pd.DataFrame(
        {
            "episode_id": [r.episode_id for r in records],
            "step": [r.step for r in records],
            "condition": conditions,
            "seed": [r.seed for r in records],
            # The reported CPVI and its leakage null (repeated cross-fit, PREREGISTRATION section 5)
            "cpvi": real,
            "cpvi_shuffled_null": np.full(len(records), np.nan) if null is None else null,
            # The H3 pair, from one shared fit - the same quantity, a different estimator, kept in
            # its own columns so nothing downstream silently mixes the two.
            "twin_retrospective": retro,
            "twin_prospective": prosp,
        }
    )
    result = RQ2Result(
        dataset_hash=dataset_hash,
        n_handoffs=len(records),
        primary_y=PRIMARY_Y,
        provenance=build_provenance(featuriser.cfg, cfg.probe),
        twin=twin,
        proxies=proxies,
        labels=labels,
        encoders=encoders,
        recommended_y=winner,
        recommended_encoder=encoder,
        recommendation_note=note,
    )
    logger.info(
        "rq2: %d handoffs, twin rho=%.3f pooled, recommended Y=%s",
        len(records),
        twin[0].agreement.spearman_rho,
        winner,
    )
    return result, frame


def _encoder_comparison(
    encoder_cfg: EncoderConfig,
    second: tuple[FloatArray, FloatArray] | None,
    y: IntArray,
    groups: IntArray,
    labels: list[YComparison],
    cfg: RQ2Config,
    real: FloatArray,
) -> EncoderComparison:
    """The sensitivity encoder against the primary on the frozen label; skipped arm says so."""
    base = EncoderComparison(
        primary_name=encoder_cfg.name,
        primary_revision=encoder_cfg.revision,
        second_name=encoder_cfg.second_encoder,
        second_revision=encoder_cfg.second_encoder_revision,
        ran=second is not None,
    )
    if second is None:
        return base
    base.primary_mean_cpvi = float(np.mean(real))
    base.second_mean_cpvi = float(np.mean(cpvi(second[0], second[1], y, groups, cfg.probe)))
    # Read off the primary label's row rather than recomputing: two independently derived copies of
    # one correlation in two tables is a table that can disagree with itself.
    primary_row = next(
        (r for r in labels if r.label == PRIMARY_Y and r.encoder_rho is not None), None
    )
    base.rho_primary_second = 0.0 if primary_row is None else float(primary_row.encoder_rho or 0.0)
    # Invariance of the *admissible set*, not of the effect sizes: the rule ranks on ordering, so
    # what matters is whether the second encoder would have admitted a different set of labels.
    base.label_ranking_invariant = all(
        (y_row.encoder_rho is None) or (y_row.encoder_rho >= _ENCODER_INSTABILITY_RHO)
        for y_row in labels
        if y_row.admissible
    )
    return base


def write_rq2(result: RQ2Result, dir: Path | str, *, scores: pd.DataFrame) -> Path:
    """Persist the analysis JSON, the per-handoff scores, the three tables and the figures."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "rq2.json").write_text(result.model_dump_json(indent=2))
    scores.to_parquet(dir / "rq2_scores.parquet", index=False)
    pd.DataFrame([t.agreement.model_dump() | {"scope": t.scope} for t in result.twin]).to_csv(
        dir / "twin_agreement.csv", index=False
    )
    pd.DataFrame([p.model_dump() for p in result.proxies]).to_csv(
        dir / "proxy_tracking.csv", index=False
    )
    pd.DataFrame([y.model_dump() for y in result.labels]).to_csv(
        dir / "label_comparison.csv", index=False
    )
    (dir / "recommendation.md").write_text(
        f"# RQ2 recommendation\n\n- **Frozen primary Y (RQ1):** `{result.primary_y}`\n"
        f"- **Recommended RQ3b gate target:** "
        f"{'none' if result.recommended_y is None else '`' + result.recommended_y + '`'}\n"
        f"- **Encoder:** `{result.recommended_encoder}`\n\n{result.recommendation_note}\n"
    )

    pooled = next(t for t in result.twin if t.scope == "pooled").agreement
    figs = {
        "twin_agreement": scatter_plot(
            (scores["twin_retrospective"] + scores["twin_prospective"]).to_numpy() / 2.0,
            (scores["twin_retrospective"] - scores["twin_prospective"]).to_numpy(),
            xlabel="mean of the two scores (bits)",
            ylabel="retrospective - prospective (bits)",
            title=f"RQ2 H3: Bland-Altman (rho={pooled.spearman_rho:.2f}, n={pooled.n})",
            hlines=[pooled.ba_bias, pooled.ba_loa_low, pooled.ba_loa_high],
            path=dir / "twin_agreement.png",
        ),
        "proxy_tracking": series_plot(
            [p.key for p in result.proxies],
            {
                "rho(proxy, CPVI)": (
                    [p.spearman_cpvi for p in result.proxies],
                    [p.spearman_ci for p in result.proxies],
                ),
                "rho(proxy, shuffled CPVI)": (
                    [p.spearman_shuffled for p in result.proxies],
                    [p.spearman_shuffled_ci for p in result.proxies],
                ),
            },
            ylabel="Spearman rho",
            title="RQ2 H4: proxy tracking against its leakage null",
            path=dir / "proxy_tracking.png",
        ),
        "label_comparison": ci_plot(
            [y.label for y in result.labels if y.status == "ok"],
            [y.corrected_mean_cpvi for y in result.labels if y.status == "ok"],
            [y.corrected_ci for y in result.labels if y.status == "ok"],
            ylabel="leakage-corrected mean CPVI (bits)",
            title="RQ2: corrected message content by candidate label",
            path=dir / "label_comparison.png",
        ),
    }
    if all(v is not None for v in figs.values()):  # all render or none (the viz extra is optional)
        result.figures = {k: str(v) for k, v in figs.items() if v is not None}
        (dir / "rq2.json").write_text(result.model_dump_json(indent=2))
    return dir
