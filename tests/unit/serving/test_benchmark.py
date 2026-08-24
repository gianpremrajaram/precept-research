"""Ladder-benchmark metric parsing, measurement and table generation (DSE-005).

No GPU and no served model: `nvidia-smi` output is a captured fixture, the endpoint is mocked, and
the capability smoke (which drives the real loop) is covered by the runner's own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from preceptx.serving.benchmark import (
    BenchmarkInvocation,
    TierResult,
    append_result,
    begin_invocation,
    measure_schema_adherence,
    measure_throughput,
    measure_ttft,
    parse_nvidia_smi,
    recommend,
    render_table,
    write_invocation,
    write_report,
)
from preceptx.serving.client import LLMClient, ServingConfig, ServingError
from preceptx.sim.fingerprint import simulation_fingerprint

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"

# Captured `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` output, two GPUs.
_SMI_FIXTURE = "12043\n28911\n"


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _row(tier: str, *, smoke: float, schema: float, tok_s: float) -> TierResult:
    return TierResult(
        tier=tier,
        model=f"vendor/{tier}",
        revision="a" * 40,
        base_url=BASE_URL,
        substrate="myriad-a100",
        structured_mode="guided_json",
        tokens_per_s=tok_s,
        ttft_s=0.4,
        peak_memory_mb=28911.0,
        schema_adherence=schema,
        smoke_success_rate=smoke,
        n_smoke_episodes=10,
        timestamp="2026-08-24T00:00:00+00:00",
    )


def test_parse_nvidia_smi_takes_the_peak() -> None:
    assert parse_nvidia_smi(_SMI_FIXTURE) == 28911.0


def test_parse_nvidia_smi_returns_none_off_gpu() -> None:
    # No GPU, or a driver error string: the figure is absent, never a fabricated zero.
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("NVIDIA-SMI has failed\n") is None


@respx.mock
def test_schema_adherence_counts_invalid_responses_as_misses() -> None:
    responses = [
        httpx.Response(200, json=_completion(json.dumps({"action": "N"}))),
        httpx.Response(200, json=_completion(json.dumps({"action": "N"}))),
        httpx.Response(200, json=_completion(json.dumps({"action": "NOT_AN_ACTION"}))),
        httpx.Response(200, json=_completion("sure, I'll move north")),  # not JSON at all
    ]
    respx.post(CHAT).mock(side_effect=responses)
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))
    assert measure_schema_adherence(client, n_calls=4) == 0.5  # 2 of 4, no retries


def test_render_table_has_one_row_per_tier() -> None:
    table = render_table(
        [_row("8b", smoke=0.6, schema=1.0, tok_s=90), _row("14b", smoke=0.8, schema=1.0, tok_s=40)]
    )
    assert table.count("\n") == 3  # header, separator, two rows
    assert "`vendor/8b`" in table and "`vendor/14b`" in table
    assert "60% (10 ep)" in table


def test_recommendation_picks_the_fastest_tier_clearing_both_floors() -> None:
    rows = [
        _row("8b", smoke=0.2, schema=1.0, tok_s=120),  # fast but cannot do the task
        _row("14b", smoke=0.8, schema=1.0, tok_s=40),
        _row("32b", smoke=0.9, schema=0.5, tok_s=15),  # capable but misses the schema
    ]
    note = recommend(rows)
    assert "`vendor/14b`" in note
    assert "vendor/8b" not in note.split("Tiers clearing the floors:")[1]


def test_recommendation_refuses_to_pick_when_nothing_clears() -> None:
    note = recommend([_row("8b", smoke=0.1, schema=0.2, tok_s=120)])
    assert "No tier clears both floors" in note


def test_rows_accumulate_across_runs_and_the_report_rewrites(tmp_path: Path) -> None:
    append_result(_row("8b", smoke=0.6, schema=1.0, tok_s=90), tmp_path)
    rows = append_result(_row("14b", smoke=0.8, schema=1.0, tok_s=40), tmp_path)
    assert [r.tier for r in rows] == ["8b", "14b"]  # one endpoint at a time, one table

    out = write_report(rows, tmp_path)
    assert (out / "ladder.md").read_text().count("\n") == 4
    assert "vendor/14b" in (out / "recommendation.md").read_text()
    csv = (out / "ladder.csv").read_text().splitlines()
    assert csv[0].startswith("tier,model,revision") and len(csv) == 3


@respx.mock
def test_latency_and_throughput_probes_survive_a_terse_endpoint() -> None:
    # Both probes go through LLMClient.chat, which rejects empty content, so neither may ask for
    # so few tokens that a live endpoint returns nothing - as the one-token TTFT probe did.
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("ready")))
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))
    assert measure_ttft(client, n_calls=2) >= 0.0
    assert measure_throughput(client, n_calls=2) > 0.0  # 1 word per call over a finite elapsed time


@respx.mock
def test_probes_propagate_an_empty_completion_rather_than_scoring_it() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("")))
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))
    with pytest.raises(ServingError):
        measure_ttft(client, n_calls=1)


# --- the invocation record: provenance for a hand-launched ladder row ---------------------------


def _invocation() -> BenchmarkInvocation:
    return begin_invocation(
        tier="14b",
        model="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        substrate="myriad-a100",
        args={"episodes": "10", "base_url": "http://localhost:8000/v1"},
    )


def test_the_invocation_record_pins_the_served_identity() -> None:
    """A ladder row is launched by hand, so ``--model``/``--revision`` cannot rest on memory: the
    served checkpoint is exactly what no later check recovers, since /v1/models carries no
    revision."""
    inv = _invocation()
    assert inv.model == "Qwen/Qwen3-14B"
    assert inv.revision == "40c069824f4251a91eefaf281ebe4c544efd3e18"
    assert inv.substrate == "myriad-a100"
    assert inv.git_sha and inv.simulation_digest and inv.host
    assert inv.args["episodes"] == "10"


def test_the_record_is_complete_before_the_run_and_says_it_has_not_finished() -> None:
    """Written before the first model call, so persistence is a precondition of serving. An
    unfinished record is the honest reading of a crashed run - and one fewer broad except clause
    than recording the failure explicitly would have cost."""
    inv = _invocation()
    assert inv.started_at
    assert inv.ended_at is None
    assert inv.exit_status is None
    assert inv.artefacts == []


def test_the_record_carries_the_world_the_smoke_episodes_ran_in() -> None:
    """So two ladder rows cannot be compared across a geometry retune by accident."""
    assert _invocation().simulation_digest == simulation_fingerprint().digest()


def test_the_record_names_the_serving_capture_it_ran_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "serve_env.json"
    sidecar.write_text(json.dumps({"vllm": "0.18.1", "gpu": "NVIDIA A100-40GB"}))
    monkeypatch.setenv("PRECEPTX_SERVE_ENV", str(sidecar))
    captured = _invocation().serve_env
    assert captured is not None
    assert captured.values["gpu"] == "NVIDIA A100-40GB"


def test_write_invocation_round_trips_and_rewrites_in_place(tmp_path: Path) -> None:
    """Two writes, one file: the second is the same invocation with its outcome filled in, not a
    second event."""
    inv = _invocation()
    first = write_invocation(inv, tmp_path)
    assert first == tmp_path / inv.run_id / "benchmark-invocation.json"
    assert json.loads(first.read_text())["exit_status"] is None

    second = write_invocation(inv.model_copy(update={"exit_status": 0}), tmp_path)
    assert second == first
    assert json.loads(second.read_text())["exit_status"] == 0
