"""The console entry points' pre-flight behaviour (DSE-031).

``--dry-run`` is the guard that lets a sweep be costed before a single token is spent, so the two
things worth pinning are that it prints the right plan and that it touches no endpoint at all.
"""

from __future__ import annotations

import pytest
import respx

from preceptx.config import ConfigError
from preceptx.experiments.cli import pilot, rq1
from preceptx.sim.feasibility import STEP_BUDGETS


@respx.mock
def test_pilot_dry_run_prints_the_plan_and_issues_no_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert pilot(["--dry-run", "--model", "qwen8b", "--conditions", "C0,C4"]) == 0
    out = capsys.readouterr().out
    # 2 conditions x 1 serialisation x 2 difficulties (easy, hard) x 5 seeds (PREREGISTRATION §6)
    assert "cells:            20" in out
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
