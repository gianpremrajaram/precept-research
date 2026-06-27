"""PVI/CPVI estimator: the conditional V-information at a handoff.

Fit a state-only baseline probe ``g_base`` and a state-plus-message probe ``g_cond`` and take the
per-instance log2-likelihood difference of the true label (roadmap §2.4). CPVI conditions on the
shared state - the novelty-defining move - so we always surface the ``PVI - CPVI`` gap (how much
apparent message value was just an echo of the state). Probes are cross-fitted with strict group
discipline: every instance is scored out-of-fold by a probe it did not train, and a fold never
splits an episode across train/test - a random handoff split would leak the shared trajectory and
inflate CPVI (the R6 leakage failure). Synthetic ground-truth tests pin the behaviour: a noise
message gives CPVI ~ 0, an informative message gives CPVI > 0, and a state-echo message gives
PVI > CPVI.

Per-instance scores are returned row-aligned to the input arrays (and hence to the source
``HandoffRecord``s they came from), the join key for downstream analysis.

Deviation logged for audit: roadmap §2.4 pins a *heteroscedastic* regressor as the continuous
default; we ship homoscedastic-per-probe Gaussian-NLL and record the choice in
``ProbeConfig.variance_model`` (and thus the run manifest), with the heteroscedastic path reserved.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

logger = logging.getLogger(__name__)

EPS = 1e-9  # probability floor before the log (roadmap pseudocode)
_DEFAULT_K = 5
_WARN_UNGROUPED_N = 50  # above this, a missing episode-group split is a likely silent leak

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class ProbeConfig(BaseModel):
    """Probe family V and cross-fit discipline. ``n_splits=None`` selects leave-one-episode-out for
    small pilots; ``variance_model`` records the continuous-path deviation from roadmap §2.4."""

    model_config = ConfigDict(extra="forbid")

    probe: Literal["logistic", "mlp"] = "logistic"
    variance_model: Literal["homoscedastic", "heteroscedastic"] = "homoscedastic"
    c: float = Field(default=1.0, gt=0)  # L2 inverse-strength (roadmap default)
    max_iter: int = Field(default=1000, ge=1)
    n_splits: int | None = 5
    mlp_hidden: int = Field(default=64, ge=1)
    seed: int = Field(default=0, ge=0)

    @field_validator("n_splits")
    @classmethod
    def _at_least_two(cls, v: int | None) -> int | None:
        if v is not None and v < 2:
            raise ValueError("n_splits must be >= 2, or None for leave-one-group-out")
        return v


class CpviResult(BaseModel):
    """Summary metrics for metrics.json; per-instance scores are returned alongside.

    ``auroc_train_cond`` is the in-sample AUROC of ``g_cond``; its gap to ``auroc_cond`` (held-out)
    is the overfit monitor for the dimensionality-vs-N risk the encoder choice carries.
    """

    model_config = ConfigDict(extra="forbid")

    n: int
    n_splits: int
    mean_cpvi: float
    mean_pvi: float
    pvi_cpvi_gap: float
    auroc_cond: float | None = None
    auroc_base: float | None = None
    auroc_train_cond: float | None = None


def _warn_ungrouped(y: np.ndarray[Any, Any], groups: IntArray | None) -> None:
    if groups is None and len(y) >= _WARN_UNGROUPED_N:
        logger.warning(
            "cross-fitting %d instances with no episode groups; a random split can leak "
            "shared-trajectory state across folds (R6). Pass groups=episode_id.",
            len(y),
        )


def _make_splitter(
    y: np.ndarray[Any, Any], groups: IntArray | None, cfg: ProbeConfig, *, stratified: bool
) -> tuple[Any, int]:
    """A CV splitter and its effective fold count, group- and stratify-aware (pure)."""
    if groups is None:
        cap = int(min(np.unique(y, return_counts=True)[1])) if stratified else len(y)
        k = max(2, min(cfg.n_splits or _DEFAULT_K, cap))
        ctor = StratifiedKFold if stratified else KFold
        return ctor(n_splits=k, shuffle=True, random_state=cfg.seed), k
    n_groups = len(np.unique(groups))
    if cfg.n_splits is None:
        return LeaveOneGroupOut(), n_groups
    k = max(2, min(cfg.n_splits, n_groups))
    return (StratifiedGroupKFold(n_splits=k) if stratified else GroupKFold(n_splits=k)), k


def _fit_classifier(X: FloatArray, y: IntArray, cfg: ProbeConfig) -> Any:
    if cfg.probe == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(cfg.mlp_hidden, cfg.mlp_hidden),
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
        ).fit(X, y)
    return LogisticRegression(C=cfg.c, max_iter=cfg.max_iter).fit(X, y)


def _fit_regressor(X: FloatArray, y: FloatArray, cfg: ProbeConfig) -> Any:
    if cfg.probe == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(cfg.mlp_hidden, cfg.mlp_hidden),
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
        ).fit(X, y)
    return Ridge(alpha=1.0 / cfg.c).fit(X, y)  # alpha is direct-strength; C is its inverse


def _proba_aligned(clf: Any, X: FloatArray, classes: IntArray) -> FloatArray:
    """Predicted class probabilities, columns aligned to ``classes`` (a fold may miss a class)."""
    p: FloatArray = clf.predict_proba(X)
    if len(clf.classes_) == len(classes) and np.array_equal(clf.classes_, classes):
        return p
    out = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    col = {c: i for i, c in enumerate(classes.tolist())}
    for j, c in enumerate(clf.classes_.tolist()):
        out[:, col[c]] = p[:, j]
    return out


def _oof_proba(
    X: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig, classes: IntArray
) -> FloatArray:
    proba = np.zeros((len(y), len(classes)), dtype=np.float64)
    splitter, _ = _make_splitter(y, groups, cfg, stratified=True)
    for tr, te in splitter.split(X, y, groups):
        proba[te] = _proba_aligned(_fit_classifier(X[tr], y[tr], cfg), X[te], classes)
    return proba


def _oof_prior(
    y: IntArray, groups: IntArray | None, cfg: ProbeConfig, classes: IntArray
) -> FloatArray:
    """The cross-fitted class marginal - the null model for unconditional PVI."""
    proba = np.zeros((len(y), len(classes)), dtype=np.float64)
    splitter, _ = _make_splitter(y, groups, cfg, stratified=True)
    for tr, te in splitter.split(np.zeros((len(y), 1)), y, groups):
        counts = np.array([(y[tr] == c).sum() for c in classes], dtype=np.float64)
        proba[te] = counts / counts.sum()
    return proba


def pointwise_logprob(proba: FloatArray, y: IntArray, classes: IntArray) -> FloatArray:
    """``log2 p[true class]`` per instance, with the roadmap epsilon floor."""
    col = {c: i for i, c in enumerate(classes.tolist())}
    idx = np.array([col[v] for v in y.tolist()])
    chosen: FloatArray = proba[np.arange(len(y)), idx]
    return np.log2(chosen + EPS)


def predictive_distributions(
    e_s: FloatArray, e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Out-of-fold ``(p_cond, p_base, classes)`` - the shared substrate of CPVI and its twin."""
    classes = np.unique(y)
    p_cond = _oof_proba(np.hstack([e_s, e_m]), y, groups, cfg, classes)
    p_base = _oof_proba(e_s, y, groups, cfg, classes)
    return p_cond, p_base, classes


def cpvi(
    e_s: FloatArray, e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig
) -> FloatArray:
    """Per-instance conditional V-information: ``log2 g_cond[y] - log2 g_base[y]`` (held-out)."""
    _warn_ungrouped(y, groups)
    p_cond, p_base, classes = predictive_distributions(e_s, e_m, y, groups, cfg)
    return pointwise_logprob(p_cond, y, classes) - pointwise_logprob(p_base, y, classes)


def _pvi_unconditional(
    e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig, classes: IntArray
) -> FloatArray:
    p_full = _oof_proba(e_m, y, groups, cfg, classes)
    p_null = _oof_prior(y, groups, cfg, classes)
    return pointwise_logprob(p_full, y, classes) - pointwise_logprob(p_null, y, classes)


def pvi(e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig) -> FloatArray:
    """Per-instance *unconditional* message value: message-probe vs the class-prior null."""
    _warn_ungrouped(y, groups)
    return _pvi_unconditional(e_m, y, groups, cfg, np.unique(y))


def cpvi_continuous(
    e_s: FloatArray, e_m: FloatArray, y: FloatArray, groups: IntArray | None, cfg: ProbeConfig
) -> FloatArray:
    """Per-instance CPVI for the continuous twin (Gaussian log2-likelihood difference)."""
    _warn_ungrouped(y, groups)
    if cfg.variance_model == "heteroscedastic":
        raise NotImplementedError(
            "heteroscedastic continuous CPVI is reserved (roadmap §2.4); use "
            "variance_model='homoscedastic'"
        )  # ponytail: add a 2-output Gaussian-NLL MLP if homoscedastic underfits the pilot
    return _oof_gaussian_ll(np.hstack([e_s, e_m]), y, groups, cfg) - _oof_gaussian_ll(
        e_s, y, groups, cfg
    )


def _oof_gaussian_ll(
    X: FloatArray, y: FloatArray, groups: IntArray | None, cfg: ProbeConfig
) -> FloatArray:
    yhat = np.zeros(len(y), dtype=np.float64)
    sig2 = np.zeros(len(y), dtype=np.float64)
    splitter, _ = _make_splitter(y, groups, cfg, stratified=False)
    for tr, te in splitter.split(X, y, groups):
        reg = _fit_regressor(X[tr], y[tr], cfg)
        sig2[te] = float(np.var(y[tr] - reg.predict(X[tr]))) + EPS  # homoscedastic, per fold
        yhat[te] = reg.predict(X[te])
    ln2 = math.log(2.0)
    return -0.5 * np.log2(2.0 * math.pi * sig2) - (y - yhat) ** 2 / (2.0 * sig2 * ln2)


def estimate(
    e_s: FloatArray, e_m: FloatArray, y: IntArray, groups: IntArray | None, cfg: ProbeConfig
) -> tuple[CpviResult, FloatArray]:
    """Full report (summary + per-instance CPVI) for a binary outcome; AUROCs only when binary."""
    _warn_ungrouped(y, groups)
    p_cond, p_base, classes = predictive_distributions(e_s, e_m, y, groups, cfg)
    cpvi_scores = pointwise_logprob(p_cond, y, classes) - pointwise_logprob(p_base, y, classes)
    pvi_scores = _pvi_unconditional(e_m, y, groups, cfg, classes)

    auroc_cond = auroc_base = auroc_train = None
    if len(classes) == 2:  # positive class is classes[1] (np.unique sorts ascending)
        auroc_cond = float(roc_auc_score(y, p_cond[:, 1]))
        auroc_base = float(roc_auc_score(y, p_base[:, 1]))
        x_cond = np.hstack([e_s, e_m])
        clf = _fit_classifier(x_cond, y, cfg)
        auroc_train = float(roc_auc_score(y, _proba_aligned(clf, x_cond, classes)[:, 1]))

    _, k = _make_splitter(y, groups, cfg, stratified=True)
    result = CpviResult(
        n=len(y),
        n_splits=k,
        mean_cpvi=float(np.mean(cpvi_scores)),
        mean_pvi=float(np.mean(pvi_scores)),
        pvi_cpvi_gap=float(np.mean(pvi_scores) - np.mean(cpvi_scores)),
        auroc_cond=auroc_cond,
        auroc_base=auroc_base,
        auroc_train_cond=auroc_train,
    )
    return result, cpvi_scores
