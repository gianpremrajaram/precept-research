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

from preceptx.config import ConfigError
from preceptx.data.writer import DatasetError
from preceptx.experiments.cli import pilot, rq1, rq2
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
