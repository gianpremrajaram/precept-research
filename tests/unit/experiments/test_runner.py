from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from preceptx.config import ModelConfig
from preceptx.data.writer import dataset_hash, load_records
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, sweep_hash
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


def _client() -> LLMClient:
    return LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))


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


def _wait_script(request: httpx.Request) -> httpx.Response:
    if b"guided_json" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    return httpx.Response(200, json=_completion("hold position"))


def _sweep(concurrency: int = 4) -> SweepConfig:
    return SweepConfig(
        conditions=["C0", "C4"],
        serialisations=["numeric"],
        difficulties=["easy"],
        seeds=[1, 2, 3],
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        max_steps=2,
        concurrency=concurrency,
    )


def _data_dir(sweep: SweepConfig, root: Path) -> Path:
    return root / dataset_hash(sweep_hash(sweep))


@respx.mock
def test_run_grid_writes_one_record_set_per_cell(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    summary = run_grid(sweep, _client(), root=tmp_path)
    assert summary.n_cells == 6  # 2 conditions x 3 seeds
    assert summary.n_episodes == 6
    assert summary.n_handoffs == 6 * 2  # max_steps=2 per episode
    run_dir = tmp_path / f"{dataset_hash(sweep_hash(sweep))}-run"
    assert (run_dir / "manifest.json").exists()  # run manifest persisted beside the dataset dir
    records = load_records(dataset_hash(sweep_hash(sweep)), root=tmp_path)
    assert len({r.episode_id for r in records}) == 6  # no dropped/duplicated episodes


@respx.mock
def test_run_grid_is_concurrency_safe(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep(concurrency=4)  # the serialised-write lock must survive 4 concurrent episodes
    run_grid(sweep, _client(), root=tmp_path)
    records = load_records(dataset_hash(sweep_hash(sweep)), root=tmp_path)
    assert len(records) == 6 * 2  # every cell's records land exactly once (no part-index race)
    assert len({r.episode_id for r in records}) == 6


@respx.mock
def test_run_grid_resume_skips_completed_cells(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    parts_first = len(list(_data_dir(sweep, tmp_path).glob("part-*.parquet")))
    summary = run_grid(sweep, _client(), root=tmp_path)  # rerun: everything already complete
    parts_second = len(list(_data_dir(sweep, tmp_path).glob("part-*.parquet")))
    assert parts_second == parts_first  # no new parts written on resume
    assert summary.n_episodes == 6  # summary still reports the full grid
    records = load_records(dataset_hash(sweep_hash(sweep)), root=tmp_path)
    assert len(records) == 6 * 2  # not duplicated
