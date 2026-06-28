"""Integration: a small mock RQ1 grid runs and the analysis emits a table + JSON (DSE-020).

Torch-free and offline - a mocked vLLM endpoint drives ``run_grid`` over a C0+C4 grid with the
east-push script (the hard difficulty moves then jams, so per-step progress carries both classes),
then ``run_rq1`` analyses it with a stub encoder. The check is that the headline driver closes the
loop and writes its artefacts; the full-scale run is gated on the resolved compute (DSE-005).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import respx
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.data.writer import dataset_hash
from preceptx.experiments.rq1 import RQ1Config, rq1_sweep, run_rq1, write_rq1
from preceptx.experiments.sweep import sweep_hash
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


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


def _east_script(request: httpx.Request) -> httpx.Response:
    if b"guided_json" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": "E"})))
    return httpx.Response(200, json=_completion("push the load east toward the goal"))


@respx.mock
def test_rq1_runs_on_a_small_grid_and_writes_artefacts(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_east_script)
    sweep = rq1_sweep(
        ModelConfig(name="m", revision="rev", tier="8b"),
        seeds=[1, 2, 3, 4],
        conditions=["C0", "C4"],  # a 2-point grid keeps the mock run small
        difficulties=["hard"],
        max_steps=8,
    )
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))
    result = run_rq1(
        sweep, client, feat, root=tmp_path, cfg=RQ1Config(probe=ProbeConfig(n_splits=2))
    )

    assert result.dataset_hash == dataset_hash(sweep_hash(sweep))
    assert {c.condition for c in result.conditions} == {"C0", "C4"}
    out = write_rq1(result, tmp_path / "rq1")
    assert (out / "rq1.json").exists() and (out / "rq1_results.csv").exists()
