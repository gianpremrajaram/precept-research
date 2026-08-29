"""The console entry points' pre-flight behaviour (DSE-031).

``--dry-run`` is the guard that lets a sweep be costed before a single token is spent, so the two
things worth pinning are that it prints the right plan and that it touches no endpoint at all.
``preceptx-rq2`` is the offline analysis entry point, so what is pinned there is the opposite: that
none of the serving pre-flight applies to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

from preceptx.analysis.stats import build_provenance
from preceptx.config import ConfigError
from preceptx.data.writer import DatasetError
from preceptx.experiments.cli import _transfer_config, pilot, rq1, rq2
from preceptx.gate.calibration import (
    CalibrationReport,
    StatisticCalibration,
    write_report,
)
from preceptx.measure.featuriser import EncoderConfig
from preceptx.measure.pvi_cpvi import ProbeConfig
from preceptx.sim.feasibility import STEP_BUDGETS


@respx.mock
def test_pilot_dry_run_prints_the_plan_and_issues_no_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pilot(["--dry-run", "--model", "qwen8b", "--conditions", "C0,C4"]) == 0
    out = capsys.readouterr().out
    # 2 conditions x 1 serialisation x 2 difficulties (easy, hard) x 10 seeds (PREREGISTRATION §6,
    # widened for attempt 2: at 5 seeds G1 rested on five easy-C0 episodes)
    assert "cells:            40" in out
    assert "Qwen/Qwen3-8B@" in out
    assert not respx.calls  # no endpoint was constructed, let alone called


@respx.mock
def test_dry_run_projects_the_upper_bound_call_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pilot(["--dry-run", "--conditions", "C0", "--difficulties", "easy", "--seeds", "0,1"])
    out = capsys.readouterr().out
    assert f"model calls:      {2 * 2 * STEP_BUDGETS['easy']}" in out  # 2 cells x 2 calls per step


@respx.mock
def test_dry_run_needs_no_endpoint_for_a_heterogeneous_pair(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rq1(["--dry-run", "--model", "qwen8b", "--model-b", "qwen14b", "--seeds", "0"])
    out = capsys.readouterr().out
    assert "model (A):        Qwen/Qwen3-8B@" in out
    assert "model (B):        Qwen/Qwen3-14B@" in out


def test_unlabelled_substrate_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unlabelled dataset cannot be told apart from a Myriad one after the fact, so this is an
    # exception at start-up rather than a warning mid-run.
    monkeypatch.delenv("PRECEPTX_SERVING_SUBSTRATE", raising=False)
    with pytest.raises(ConfigError, match="PRECEPTX_SERVING_SUBSTRATE"):
        pilot(["--seeds", "0"])


@respx.mock
def test_rq2_is_offline_and_needs_no_serving_pre_flight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # It analyses episodes already on disk, so neither the substrate label nor an endpoint applies:
    # the only thing it can fail on is the dataset itself.
    monkeypatch.delenv("PRECEPTX_SERVING_SUBSTRATE", raising=False)
    with pytest.raises(DatasetError, match="no parquet parts"):
        rq2(["--dataset-hash", "nosuchhash", "--root", str(tmp_path)])
    assert not respx.calls


def test_rq2_requires_the_dataset_hash() -> None:
    with pytest.raises(SystemExit):  # argparse exits 2; the hash is the only handle it has
        rq2([])


@respx.mock
def test_max_steps_broadcasts_over_the_certified_budgets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # STEP_BUDGETS bounds an OPTIMAL policy; run 232980 saturated every one of 96 episodes, so the
    # budget is a live constraint and the flag has to reach the sweep, not just the parser.
    rq1(
        [
            "--dry-run",
            "--conditions",
            "C0",
            "--difficulties",
            "easy",
            "--seeds",
            "0,1",
            "--max-steps",
            "50",
        ]
    )
    assert f"model calls:      {2 * 2 * 50}" in capsys.readouterr().out


@respx.mock
def test_omitting_max_steps_keeps_the_certified_budgets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rq1(["--dry-run", "--conditions", "C0", "--difficulties", "easy", "--seeds", "0,1"])
    assert f"model calls:      {2 * 2 * STEP_BUDGETS['easy']}" in capsys.readouterr().out


# ------------------------------------------------------- the RQ3a transfer arm's wiring (DSE-024)


def _calibration(dir: Path, *, keys: dict[str, float]) -> Path:
    """A calibration.json from the real models, so the fixture cannot drift from the schema."""
    dir.mkdir(parents=True, exist_ok=True)
    report = CalibrationReport(
        dataset_hash="d0",
        provenance=build_provenance(EncoderConfig(), ProbeConfig()),
        n=100,
        n_bins=10,
        ece_reliable=False,
        statistics=[
            StatisticCalibration(
                key=key,
                threshold=0.5,
                orientation=orientation,
                firing_rate=0.2,
                auroc=0.6,
                ece=0.0,
                n_classes=2,
                reliability=[],
            )
            for key, orientation in keys.items()
        ],
    )
    for key in keys:
        # `_transfer_config` and `load_statistic` both check presence only, so presence is enough.
        (dir / f"{key}.manifest.json").touch()
    return write_report(report, dir)


def test_transfer_arm_is_off_unless_a_calibration_is_named() -> None:
    cfg, train_hash = _transfer_config(None, "fail")
    assert cfg.transfer_dir is None and cfg.transfer_orientation is None and train_hash is None


def test_transfer_orientation_is_read_from_the_report_never_typed(tmp_path: Path) -> None:
    """A hand-entered sign would silently invert every localisation number in the table."""
    _calibration(tmp_path, keys={"fail": -1.0})
    cfg, train_hash = _transfer_config(tmp_path, "fail")
    assert cfg.transfer_orientation == -1.0 and cfg.transfer_key == "fail"
    assert cfg.transfer_dir == tmp_path and train_hash == "d0"


def test_transfer_resolves_a_retired_key_against_the_report(tmp_path: Path) -> None:
    _calibration(tmp_path, keys={"fail": 1.0})
    cfg, _ = _transfer_config(tmp_path, "info")  # DSE-061 retired "info"
    assert cfg.transfer_key == "fail"


def test_a_transfer_dir_that_cannot_supply_the_arm_fails_loud(tmp_path: Path) -> None:
    """Wiring errors raise; they do not degrade into an unavailable row that reads as "not run"."""
    with pytest.raises(ConfigError, match="preceptx-calibrate"):
        _transfer_config(tmp_path, "fail")

    _calibration(tmp_path / "cal", keys={"cosine": 1.0})
    with pytest.raises(ConfigError, match="has no statistic 'fail'"):
        _transfer_config(tmp_path / "cal", "fail")


def test_a_frozen_run_dir_without_the_joblib_fails_loud(tmp_path: Path) -> None:
    """A frozen run dir has the report but not the probe: catch it here, not 8h into a job."""
    _calibration(tmp_path, keys={"fail": 1.0})
    (tmp_path / "fail.manifest.json").unlink()
    with pytest.raises(ConfigError, match="no persisted 'fail' statistic"):
        _transfer_config(tmp_path, "fail")
