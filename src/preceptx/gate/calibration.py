"""Offline gate calibration (DSE-017).

Pick each runtime statistic's operating point by validating it against *realised outcomes* (episode
failure), never against CPVI - the D10 circularity fix (roadmap §2.5, R5). For each statistic we
cross-fit by episode (honest held-out scores), orient it so higher = more failure-risk, measure
AUROC plus a Platt-calibrated ECE and reliability curve against failure, and choose the most
aggressive threshold whose firing rate stays within a budget. The threshold sits on the raw oriented
score so the gate (DSE-018) applies it with one comparison; the Platt map is report-only. Output: a
persisted ``CalibrationReport`` (per statistic: threshold, orientation, firing rate, AUROC, ECE).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from preceptx.data.schema import HandoffRecord
from preceptx.gate.statistics import (
    CosineStatistic,
    FailStatistic,
    InfoStatistic,
    Statistic,
    episode_groups,
    failure_label,
)
from preceptx.manifest import git_sha
from preceptx.measure.featuriser import Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

_ECE_RELIABLE_MIN = 200  # below this, ECE is high-variance; flagged in the report and a log warning


class CalibrationConfig(BaseModel):
    """Operating-point search: the firing-rate budget, ECE binning, and the cross-fit probe."""

    model_config = ConfigDict(extra="forbid")

    firing_rate_budget: float = Field(default=0.2, gt=0, le=1)
    n_bins: int = Field(default=10, ge=2)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)  # ``n_splits`` sets the cross-fit folds
    seed: int = Field(default=0, ge=0)


class ReliabilityBin(BaseModel):
    """One reliability-curve bin; ``count`` is surfaced so small-N bins are not over-read."""

    model_config = ConfigDict(extra="forbid")

    p_mean: float
    y_rate: float
    count: int


class StatisticCalibration(BaseModel):
    """The calibrated operating point for one statistic, keyed by a stable string for DSE-018."""

    model_config = ConfigDict(extra="forbid")

    key: str
    threshold: float
    orientation: float  # +1/-1: oriented score = orientation * raw, so higher = more failure-risk
    firing_rate: float
    auroc: float | None
    ece: float | None
    n_classes: int
    reliability: list[ReliabilityBin]


class CalibrationReport(BaseModel):
    """The persisted calibration: thresholds + diagnostics, explicitly against realised failure."""

    model_config = ConfigDict(extra="forbid")

    target: Literal["realised_failure"] = "realised_failure"  # never CPVI (R5 circularity guard)
    dataset_hash: str
    git_sha: str
    n: int
    n_bins: int
    ece_reliable: bool
    statistics: list[StatisticCalibration]


def _n_folds(groups: IntArray, cfg: CalibrationConfig) -> int:
    return max(2, min(cfg.probe.n_splits or 5, len(np.unique(groups))))


def _oof_scores(
    factory: Callable[[], Statistic],
    e_s: FloatArray,
    e_m: FloatArray,
    y: IntArray,
    groups: IntArray,
    cfg: CalibrationConfig,
) -> FloatArray:
    """Held-out scores: a fresh statistic per fold trains on train and scores test (group split).

    The group split (no episode in both folds) is what keeps the AUROC honest: a per-handoff random
    split would let the probe memorise an episode's shared state and inflate the score (R6 leakage).
    """
    scores = np.zeros(len(groups), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=_n_folds(groups, cfg)).split(e_s, y, groups):
        stat = factory()
        stat.fit(e_s[tr], e_m[tr], y[tr])
        scores[te] = stat.score(e_s[te], e_m[te])
    return scores


def _safe_auroc(fail: IntArray, score: FloatArray) -> float | None:
    """AUROC of ``score`` for predicting failure; ``None`` when failure has a single class."""
    if len(np.unique(fail)) < 2:
        return None
    return float(roc_auc_score(fail, score))


def _orient(raw: FloatArray, fail: IntArray) -> float:
    """+1, or -1 when the raw score anti-correlates with failure (AUROC < 0.5)."""
    auroc = _safe_auroc(fail, raw)
    return -1.0 if auroc is not None and auroc < 0.5 else 1.0


def _choose_threshold(score: FloatArray, budget: float) -> tuple[float, float]:
    """Most aggressive threshold (block ``score >= t``) with firing rate <= budget. Deterministic.

    The score is oriented to failure, so a lower threshold blocks strictly more handoffs; the lowest
    within-budget threshold therefore maximises failures caught subject to the budget. A tie mass at
    the budget quantile can overshoot - we then step just above the tie (more conservative), never
    over budget; if that empties, the no-op threshold fires nothing.
    """
    t = float(np.quantile(score, 1.0 - budget))
    firing = float(np.mean(score >= t))
    if firing > budget:
        above = score[score > t]
        if above.size:
            t = float(np.min(above))
        else:  # degenerate/constant scores: no within-budget cut exists, so block nothing
            logger.warning(
                "firing_rate_budget %.2f infeasible (degenerate scores) - threshold set to no-op",
                budget,
            )
            t = float(np.max(score)) + 1.0
        firing = float(np.mean(score >= t))
    return t, firing


def _platt_reliability(
    score: FloatArray, fail: IntArray, n_bins: int
) -> tuple[float | None, list[ReliabilityBin]]:
    """Platt-map the held-out score to ``P(fail)``, then ECE and reliability bins (with counts).

    The Platt fit is on the held-out (cross-fit) scores, not in-sample statistic scores, so the ECE
    is honest. The residual optimism of evaluating a 2-parameter map on its own fit is second order.
    The map is a fixed 1-parameter logistic (default ``C``), deliberately not wired to
    ``ProbeConfig.c``, which governs the statistic's probe rather than this calibration map.
    """
    if len(np.unique(fail)) < 2:
        return None, []
    lr = LogisticRegression().fit(score.reshape(-1, 1), fail)
    col = list(lr.classes_).index(1)
    p: FloatArray = lr.predict_proba(score.reshape(-1, 1))[:, col]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        m = idx == b
        c = int(np.sum(m))
        if c == 0:
            continue  # skip empty bins: keeps the JSON nan-free and round-trippable
        p_mean = float(np.mean(p[m]))
        y_rate = float(np.mean(fail[m]))
        ece += c / len(p) * abs(p_mean - y_rate)
        bins.append(ReliabilityBin(p_mean=p_mean, y_rate=y_rate, count=c))
    return float(ece), bins


def _calibrate_one(
    factory: Callable[[], Statistic],
    records: list[HandoffRecord],
    e_s: FloatArray,
    e_m: FloatArray,
    cfg: CalibrationConfig,
) -> StatisticCalibration:
    stat = factory()
    y = stat.label(records)  # the label THIS statistic predicts
    fail = failure_label(records)  # the calibration TARGET (realised failure, never CPVI)
    groups = episode_groups(records)
    raw = _oof_scores(factory, e_s, e_m, y, groups, cfg)
    orientation = _orient(raw, fail)
    oriented = orientation * raw
    threshold, firing = _choose_threshold(oriented, cfg.firing_rate_budget)
    ece, reliability = _platt_reliability(oriented, fail, cfg.n_bins)
    return StatisticCalibration(
        key=stat.key,
        threshold=threshold,
        orientation=orientation,
        firing_rate=firing,
        auroc=_safe_auroc(fail, oriented),
        ece=ece,
        n_classes=len(np.unique(y)),
        reliability=reliability,
    )


def calibrate(
    records: list[HandoffRecord],
    featuriser: Featuriser,
    *,
    dataset_hash: str,
    cfg: CalibrationConfig | None = None,
) -> CalibrationReport:
    """Calibrate all three runtime statistics against realised failure; return the report."""
    cfg = cfg or CalibrationConfig()
    failure_label(records)  # fail loud on unlabelled data before any compute
    if len(records) < _ECE_RELIABLE_MIN:
        logger.warning(
            "calibrating on N=%d < %d handoffs; ECE is high-variance, treat it as indicative",
            len(records),
            _ECE_RELIABLE_MIN,
        )
    e_s, e_m = featuriser.featurise(records)
    factories: list[Callable[[], Statistic]] = [
        lambda: InfoStatistic(cfg.probe),
        lambda: FailStatistic(cfg.probe),
        CosineStatistic,
    ]
    return CalibrationReport(
        dataset_hash=dataset_hash,
        git_sha=git_sha(),
        n=len(records),
        n_bins=cfg.n_bins,
        ece_reliable=len(records) >= _ECE_RELIABLE_MIN,
        statistics=[_calibrate_one(f, records, e_s, e_m, cfg) for f in factories],
    )


def write_report(report: CalibrationReport, dir: Path | str) -> Path:
    """Persist the report JSON (always) and a reliability figure (only with the viz extra)."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / "calibration.json"
    path.write_text(report.model_dump_json(indent=2))
    _render_figure(report, dir)
    return path


def _render_figure(report: CalibrationReport, dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib absent (install the 'viz' extra) - skipping the calibration figure")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
    for sc in report.statistics:
        pts = [(b.p_mean, b.y_rate) for b in sc.reliability if b.count > 0]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=f"{sc.key} (AUROC={sc.auroc})")
    ax.set_xlabel("predicted P(fail)")
    ax.set_ylabel("observed failure rate")
    ax.set_title(f"Gate calibration (N={report.n})")
    ax.legend()
    fig.savefig(dir / "calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
