"""The agent spine (DSE-010/011/012) feeds the measure stack (DSE-013/014) end to end.

A small mock-LLM grid is run, its handoff records are loaded back, embedded with a stub encoder, and
scored by the CPVI estimator. This is the close-the-loop check: real runner output flows through the
featuriser and estimator without a schema or wiring break. The hard difficulty with an east push
moves the load then jams it at the first slit, so ``y_binary_progress`` carries both classes (needed
for the stratified group folds).
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
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, sweep_hash
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, cpvi
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


class _StubEncoder:
    """Deterministic content-addressed embedder: torch-free, distinct vector per distinct text."""

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
def test_spine_output_feeds_cpvi(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_east_script)
    sweep = SweepConfig(
        conditions=["C0"],
        serialisations=["numeric"],
        difficulties=["hard"],  # moves then jams -> mixed y_binary_progress
        seeds=[1, 2, 3, 4],
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        max_steps=8,
        concurrency=2,
    )
    run_grid(
        sweep, LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0)), root=tmp_path
    )

    records = load_records(dataset_hash(sweep_hash(sweep)), root=tmp_path)
    assert records, "the spine produced no records"

    featuriser = Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())
    e_s, e_m = featuriser.featurise(records)
    assert e_s.shape == e_m.shape == (len(records), 16)  # aligned, row-for-row

    y = np.array([int(bool(r.y_binary_progress)) for r in records])
    groups = np.unique([r.episode_id for r in records], return_inverse=True)[1]
    assert len(np.unique(y)) == 2, "fixture must carry both outcome classes for the group folds"

    scores = cpvi(e_s, e_m, y, groups, ProbeConfig(n_splits=2))
    assert scores.shape == (len(records),)
    assert np.all(np.isfinite(scores))  # the loop closes: every handoff gets a finite CPVI score
