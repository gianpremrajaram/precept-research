"""Integration: the pilot harness runs on a small stub sweep and emits a go/no-go report (DSE-019).

Torch-free and offline - a mocked vLLM endpoint drives ``run_grid`` over a C0+C4 grid, a stub
encoder feeds G2's CPVI. The always-WAIT script makes every episode fail, so the report exercises a
mixed verdict (G1 fail, G2 single-class guard, G3 pass) rather than a happy path; the check is that
the harness closes the loop and writes a report, not the specific verdicts.
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
from preceptx.data.writer import dataset_hash, load_records
from preceptx.experiments.pilot import run_pilot, write_pilot_report
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, sweep_hash
from preceptx.measure.featuriser import EncoderConfig, Featuriser
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


def _wait_script(request: httpx.Request) -> httpx.Response:
    if b"guided_json" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    return httpx.Response(200, json=_completion("hold position"))


@respx.mock
def test_pilot_runs_on_a_small_sweep_and_writes_report(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = SweepConfig(
        conditions=["C0", "C4"],  # G2 needs a degraded condition to contrast C0 against
        serialisations=["numeric"],
        difficulties=["easy"],
        seeds=[1, 2, 3],
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        max_steps=2,
    )
    run_grid(
        sweep, LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0)), root=tmp_path
    )
    d_hash = dataset_hash(sweep_hash(sweep))
    records = load_records(d_hash, root=tmp_path)

    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    report = run_pilot(records, feat, dataset_hash=d_hash)

    assert {g.name for g in report.gates} == {"G1 capability", "G2 signal", "G3 groundedness"}
    assert report.recommendation in {"proceed", "retune_once", "fallback"}
    out = write_pilot_report(report, tmp_path / "pilot")
    assert (out / "pilot.md").exists() and (out / "pilot.json").exists()
