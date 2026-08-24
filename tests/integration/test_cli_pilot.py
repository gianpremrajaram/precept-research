"""Integration: ``preceptx-pilot`` closes the loop from a shell argv to a written report (DSE-031).

Torch-free and offline: a mocked OpenAI-compatible endpoint stands in for the served model and a
stub encoder for the featuriser, so the test exercises the wiring - config resolution, health check,
grid run, manifest, analysis, report - rather than any model behaviour. The second test covers the
per-role client path (DSE-049) end to end on the same two-cell grid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.experiments import cli
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
BASE_URL_B = "http://localhost:8001/v1"


class _StubEncoder:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        out = np.zeros((len(sentences), 16), dtype=np.float64)
        for i, s in enumerate(sentences):
            seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")
            out[i] = np.random.default_rng(seed).standard_normal(16)
        return out


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
    if b"guided_json" in request.content or b"response_format" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    return httpx.Response(200, json=_completion("hold position"))


def _mock_endpoint(base_url: str, model: str = "Qwen/Qwen3-8B") -> None:
    # The served id has to be the configured one: health_check compares them, so that a leftover
    # job serving another tier cannot be recorded as this one (DSE-002).
    respx.get(f"{base_url}/models").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"id": model, "object": "model"}]}
        )
    )
    respx.post(f"{base_url}/chat/completions").mock(side_effect=_wait_script)


def _stub_featuriser(monkeypatch: pytest.MonkeyPatch, cache_dir: Path) -> None:
    """Swap the CLI's real (torch-backed) featuriser for the stub, leaving the wiring intact."""
    monkeypatch.setattr(
        cli,
        "Featuriser",
        lambda cfg: Featuriser(EncoderConfig(cache_dir=cache_dir), encoder=_StubEncoder()),
    )


@respx.mock
def test_pilot_cli_runs_a_two_cell_grid_and_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_endpoint(BASE_URL)
    _stub_featuriser(monkeypatch, tmp_path / "embed")
    monkeypatch.setenv("PRECEPTX_SERVING_SUBSTRATE", "local-lmstudio")

    exit_code = cli.pilot(
        [
            "--model",
            "qwen8b",
            "--conditions",
            "C0,C4",
            "--difficulties",
            "easy",
            "--seeds",
            "1",
            "--root",
            str(tmp_path),
            "--structured-mode",
            "response_format",
        ]
    )

    assert exit_code == 0
    report = tmp_path / f"{dataset_hash_for(_expected_sweep())}-report"
    assert (report / "pilot.md").exists()
    assert json.loads((report / "pilot.json").read_text())["recommendation"] in {
        "proceed",
        "retune_once",
        "fallback",
    }

    manifest = json.loads(
        (tmp_path / f"{dataset_hash_for(_expected_sweep())}-run" / "manifest.json").read_text()
    )
    assert manifest["serving_substrate"] == "local-lmstudio"
    assert manifest["structured_mode"] == "response_format"
    assert manifest["endpoint_base_url"] == BASE_URL


def _expected_sweep() -> SweepConfig:
    """The sweep the argv above resolves to - the dataset hash keys the run and report dirs."""
    return SweepConfig(
        conditions=["C0", "C4"],
        serialisations=["numeric"],
        difficulties=["easy"],
        seeds=[1],
        model=ModelConfig(
            name="Qwen/Qwen3-8B",
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            tier="8b",
        ),
    )


@respx.mock
def test_two_distinct_clients_run_a_two_cell_grid_end_to_end(tmp_path: Path) -> None:
    _mock_endpoint(BASE_URL)
    _mock_endpoint(BASE_URL_B)
    sweep = _expected_sweep().model_copy(
        update={"model_b": ModelConfig(name="mb", revision="rev-b", tier="14b")}
    )
    summary = run_grid(
        sweep,
        LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0)),
        LLMClient(ServingConfig(model="mb", base_url=BASE_URL_B, max_retries=0)),
        root=tmp_path,
    )
    assert summary.n_cells == 2 and summary.n_episodes == 2
