"""Target-free runtime statistics for the gate (DSE-016).

Three per-handoff scores computable at the A->B handoff with no access to the realised outcome: the
gate cannot threshold CPVI (it needs Y), so it thresholds one of these instead (roadmap §2.5). The
no-Y guarantee is structural - ``score(e_s, e_m)`` never takes Y; only ``fit`` does, and only the
probe-backed statistics fit at all. ``CosineStatistic`` is probe-independent and so is the statistic
that answers the circularity objection. Each statistic owns the label it predicts (``label``), so a
caller (the DSE-017 calibration) can never feed the wrong Y to the wrong statistic.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import joblib  # transitive via scikit-learn; the CLAUDE.md-sanctioned probe/array persister
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord
from preceptx.manifest import git_sha
from preceptx.measure.divergence import embedding_cosine
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import EPS, ProbeConfig, _fit_classifier

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class GateError(RuntimeError):
    """A runtime statistic was used incorrectly (e.g. scored before being fit, or not found)."""


def _require_labelled(records: list[HandoffRecord]) -> None:
    """Fail loud on unlabelled episodes - calibrating on a None outcome is a silent bug (R5)."""
    if any(r.y_terminal_success is None for r in records):
        raise ConfigError(
            "runtime statistics need labelled episodes; found y_terminal_success=None "
            "(run the DSE-009 labeller before calibration)"
        )


def failure_label(records: list[HandoffRecord]) -> IntArray:
    """The realised-failure calibration target: 1 = the episode did not reach the goal."""
    _require_labelled(records)
    return np.array([0 if r.y_terminal_success else 1 for r in records], dtype=int)


def outcome_label(records: list[HandoffRecord]) -> IntArray:
    """The frozen binary outcome the info probe predicts (terminal success in the pilot)."""
    _require_labelled(records)
    # ponytail: binary pilot Y = terminal success; extend to y_discrete_config when V widens.
    return np.array([1 if r.y_terminal_success else 0 for r in records], dtype=int)


def episode_groups(records: list[HandoffRecord]) -> IntArray:
    """Integer episode ids for group-aware cross-fitting (no episode spans train and test)."""
    seen: dict[str, int] = {}
    return np.array([seen.setdefault(r.episode_id, len(seen)) for r in records], dtype=int)


def _entropy_bits(p: FloatArray) -> FloatArray:
    """Row-wise Shannon entropy in bits; a point mass (single class) gives 0."""
    return np.asarray(-np.sum(p * np.log2(p + EPS), axis=1), dtype=np.float64)


def _fit_probe(e_s: FloatArray, e_m: FloatArray, y: IntArray, cfg: ProbeConfig) -> Any:
    """The probe on ``[e_s;e_m]``, or None for a one-class fold (then scored as a constant)."""
    if len(np.unique(y)) < 2:
        return None
    return _fit_classifier(np.hstack([e_s, e_m]), y, cfg)


class Statistic(ABC):
    """A target-free per-handoff score. ``fit`` may use Y; ``score`` never does."""

    key: ClassVar[str]

    @abstractmethod
    def label(self, records: list[HandoffRecord]) -> IntArray:
        """The integer label THIS statistic predicts (so callers cannot pass the wrong Y)."""

    @abstractmethod
    def fit(self, e_s: FloatArray, e_m: FloatArray, y: IntArray) -> None:
        """Train on the state+message embeddings and the statistic's own label."""

    @abstractmethod
    def score(self, e_s: FloatArray, e_m: FloatArray) -> FloatArray:
        """Per-handoff score at the handoff - no access to Y."""


class InfoStatistic(Statistic):
    """``s_info``: predictive entropy ``H(g_cond)`` about the outcome given ``[e_s;e_m]``.

    Uses the offline-trained probe (the same family as the CPVI estimator's ``g_cond``) but never
    the realised outcome at score time - the distinction the thesis relies on.
    """

    key: ClassVar[str] = "info"

    def __init__(self, cfg: ProbeConfig | None = None) -> None:
        self.cfg = cfg or ProbeConfig()
        self._clf: Any = None
        self._fitted = False
        self.n_classes = 0

    def label(self, records: list[HandoffRecord]) -> IntArray:
        return outcome_label(records)

    def fit(self, e_s: FloatArray, e_m: FloatArray, y: IntArray) -> None:
        self.n_classes = len(np.unique(y))  # a one-class fold -> None probe -> entropy 0, no crash
        self._clf = _fit_probe(e_s, e_m, y, self.cfg)
        self._fitted = True

    def score(self, e_s: FloatArray, e_m: FloatArray) -> FloatArray:
        if not self._fitted:
            raise GateError("InfoStatistic scored before fit")
        if self._clf is None:
            return np.zeros(len(e_s), dtype=np.float64)
        proba: FloatArray = self._clf.predict_proba(np.hstack([e_s, e_m]))
        return _entropy_bits(proba)


class FailStatistic(Statistic):
    """``s_fail``: a dedicated failure-risk probe; outputs ``P(fail)`` at the handoff."""

    key: ClassVar[str] = "fail"

    def __init__(self, cfg: ProbeConfig | None = None) -> None:
        self.cfg = cfg or ProbeConfig()
        self._clf: Any = None
        self._fitted = False
        self._base_rate = 0.0

    def label(self, records: list[HandoffRecord]) -> IntArray:
        return failure_label(records)

    def fit(self, e_s: FloatArray, e_m: FloatArray, y: IntArray) -> None:
        # A one-class fold gives a None probe; score then falls back to this base rate.
        self._base_rate = float(np.mean(y)) if len(y) else 0.0
        self._clf = _fit_probe(e_s, e_m, y, self.cfg)
        self._fitted = True

    def score(self, e_s: FloatArray, e_m: FloatArray) -> FloatArray:
        if not self._fitted:
            raise GateError("FailStatistic scored before fit")
        if self._clf is None:
            return np.full(len(e_s), self._base_rate, dtype=np.float64)
        proba: FloatArray = self._clf.predict_proba(np.hstack([e_s, e_m]))
        col = list(self._clf.classes_).index(1)  # P(class 1 == failure)
        return np.asarray(proba[:, col], dtype=np.float64)


class CosineStatistic(Statistic):
    """``s_cos``: cosine(e_m, e_s) - message vs pre-handoff state. Probe-independent (no fit, no Y).

    A message highly redundant with the state has high cosine; calibration (DSE-017) decides which
    side gates. Being independent of any fitted probe, this is the statistic that defeats the
    circularity objection.
    """

    key: ClassVar[str] = "cosine"

    def label(self, records: list[HandoffRecord]) -> IntArray:
        return np.zeros(len(records), dtype=int)  # probe-independent: no label is used

    def fit(self, e_s: FloatArray, e_m: FloatArray, y: IntArray) -> None:
        return None  # nothing to train - this is the circularity-proof statistic

    def score(self, e_s: FloatArray, e_m: FloatArray) -> FloatArray:
        return embedding_cosine(e_m, e_s)  # reuse the DSE-015 state-echo bridge (zero-norm safe)


def score_records(
    stat: Statistic, records: list[HandoffRecord], featuriser: Featuriser
) -> tuple[FloatArray, IntArray]:
    """Score a fitted ``stat`` over ``records``; returns ``(scores, episode_groups)`` row-aligned.

    Keeping the group derivation here makes the calibration cross-fit (DSE-017) a one-liner and the
    join key (record order) explicit in one place.
    """
    e_s, e_m = featuriser.featurise(records)
    return stat.score(e_s, e_m), episode_groups(records)


class StatisticManifest(BaseModel):
    """Provenance for a persisted fitted statistic; links to the producing run via ``git_sha``."""

    model_config = ConfigDict(extra="forbid")

    key: str
    encoder_name: str
    encoder_revision: str
    probe_config: ProbeConfig
    train_dataset_hash: str
    n_classes: int
    git_sha: str
    timestamp: str


def save_statistic(
    stat: Statistic, *, encoder: EncoderConfig, train_dataset_hash: str, dir: Path | str
) -> Path:
    """Persist a fitted statistic (joblib) and its provenance manifest; gitignored. Returns dir."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stat, dir / f"{stat.key}.joblib")
    manifest = StatisticManifest(
        key=stat.key,
        encoder_name=encoder.name,
        encoder_revision=encoder.revision,
        probe_config=getattr(stat, "cfg", ProbeConfig()),
        train_dataset_hash=train_dataset_hash,
        n_classes=getattr(stat, "n_classes", 0),
        git_sha=git_sha(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )
    (dir / f"{stat.key}.manifest.json").write_text(manifest.model_dump_json(indent=2))
    return dir


def load_statistic(key: str, *, dir: Path | str) -> Statistic:
    """Load a persisted statistic by its stable string key, validating the key on load (DSE-018)."""
    dir = Path(dir)
    if not (dir / f"{key}.manifest.json").exists():
        raise GateError(f"no persisted statistic {key!r} under {dir}")
    stat: Statistic = joblib.load(dir / f"{key}.joblib")
    if stat.key != key:
        raise GateError(f"persisted statistic key mismatch: file is {stat.key!r}, asked {key!r}")
    return stat
