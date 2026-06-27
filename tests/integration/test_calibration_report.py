"""Integration: the gate calibration closes over a fixture dataset and emits a persisted report.

Torch-free - a stub encoder embeds the record texts. Failure is encoded in ``state_str`` so the
probe-backed statistics have signal to calibrate against. Exercises ``calibrate`` ->
``write_report`` -> reload end to end (DSE-017 "emits a report").
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from preceptx.data.schema import HandoffRecord
from preceptx.gate.calibration import (
    CalibrationConfig,
    CalibrationReport,
    calibrate,
    write_report,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser


class _HashEncoder:
    """Deterministic, content-varied embeddings: a 'risk' flag plus stable sha256-seeded noise."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        rows = []
        for s in sentences:
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            noise = np.random.default_rng(seed).standard_normal(4)
            rows.append([1.0 if "risk" in s else -1.0, *noise.tolist()])
        return np.array(rows, dtype=np.float64)


def _dataset(n_episodes: int = 24, per: int = 5) -> list[HandoffRecord]:
    rng = np.random.default_rng(0)
    records: list[HandoffRecord] = []
    for ep in range(n_episodes):
        failed = bool(rng.random() < 0.5)
        tag = "risk" if failed else "calm"
        for step in range(per):
            records.append(
                HandoffRecord(
                    episode_id=f"ep{ep}",
                    step=step,
                    condition="C0",
                    serialisation="numeric",
                    difficulty="hard",
                    model="m",
                    seed=ep,
                    state={},
                    state_str=f"{tag} state ep{ep} step{step}",
                    message_raw="raw",
                    message_delivered=f"{tag} message ep{ep}",
                    action={},
                    pre_state={},
                    post_state={},
                    progress=0.0,
                    success=not failed,
                    collision=False,
                    stuck=False,
                    y_terminal_success=not failed,
                )
            )
    return records


def test_calibration_runs_and_persists_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = _dataset()
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "cache"), encoder=_HashEncoder())
    report = calibrate(records, feat, dataset_hash="d0", cfg=CalibrationConfig(n_bins=5))

    assert report.target == "realised_failure"  # never CPVI
    assert {s.key for s in report.statistics} == {"info", "fail", "cosine"}
    for sc in report.statistics:
        assert np.isfinite(sc.threshold)
        assert sc.firing_rate <= 0.2 + 1e-9  # within the default budget
        assert sc.orientation in (1.0, -1.0)

    path = write_report(report, tmp_path / "run")
    assert path.exists()
    reloaded = CalibrationReport.model_validate_json(path.read_text())
    assert reloaded == report  # round-trips through the persisted JSON
