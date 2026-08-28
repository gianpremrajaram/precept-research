"""Integration: RQ2 analyses a real RQ1 dataset and emits its figures and decision note (DSE-022).

Torch-free and offline. A mocked vLLM endpoint drives ``run_grid`` over the same C0+C4 hard grid the
RQ1 integration uses (the east-push script moves then jams, so per-step progress carries both
classes), the episodes go to disk through the normal writer, and ``analyse_rq2`` then reads them
back exactly as ``preceptx-rq2`` would. The point is that the offline entry point closes the loop on
artefacts another driver produced - no in-memory hand-off, no fixture shortcut.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import httpx
import numpy as np
import respx
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.data.writer import load_records
from preceptx.experiments.rq1 import rq1_sweep
from preceptx.experiments.rq2 import PRIMARY_Y, RQ2Config, analyse_rq2, write_rq2
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import dataset_hash_for
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


class _StubEncoder:
    def __init__(self, salt: str = "") -> None:
        self.salt = salt

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
            seed = int.from_bytes(hashlib.sha256((self.salt + s).encode()).digest()[:4], "big")
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


# E,E,W rather than pure E (DSE-059). The progress label is "net geodesic gain over the next k
# steps > 0", and a load pressed against a wall still creeps forward ~0.012 units per step inside
# the contact solver's slop - so a pure-east script now labels EVERY handoff as progress and the
# analysis has one class to fit. Backing off every third step makes the label genuinely two-class
# (26/32 positive) without depending on a jam that the corrected physics no longer produces.
_ACTION_CYCLE = itertools.cycle(("E", "E", "W"))


def _east_script(request: httpx.Request) -> httpx.Response:
    if b"structured_outputs" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": next(_ACTION_CYCLE)})))
    return httpx.Response(200, json=_completion("push the load east toward the goal"))


@respx.mock
def test_rq2_analyses_an_rq1_dataset_and_writes_its_report(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_east_script)
    sweep = rq1_sweep(
        ModelConfig(name="m", revision="rev", tier="8b"),
        seeds=[1, 2, 3, 4],
        conditions=["C0", "C4"],
        difficulties=["hard"],
        max_steps=8,
    )
    client = LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))
    run_grid(sweep, client, None, root=tmp_path)
    d_hash = dataset_hash_for(sweep)
    records = load_records(d_hash, root=tmp_path)

    cfg = RQ2Config(
        probe=ProbeConfig(n_splits=2, n_repeats=1), n_boot=100, n_boot_track=50, n_shuffle=2
    )
    result, scores = analyse_rq2(
        records,
        Featuriser(EncoderConfig(cache_dir=tmp_path / "e1"), encoder=_StubEncoder()),
        dataset_hash=d_hash,
        second_featuriser=Featuriser(
            EncoderConfig(cache_dir=tmp_path / "e2"), encoder=_StubEncoder(salt="other")
        ),
        cfg=cfg,
    )

    assert result.dataset_hash == d_hash and result.n_handoffs == len(records)
    assert result.primary_y == PRIMARY_Y
    assert len(scores) == result.n_handoffs  # row-aligned to the dataset it was scored on
    assert result.provenance.encoder_name == EncoderConfig().name  # P1-8 rides along
    assert result.twin[0].scope == "pooled"
    assert {p.key for p in result.proxies} == {"info", "fail", "cosine"}
    assert len(result.labels) == 4  # every candidate label keeps a row
    assert result.encoders.ran is True

    out = write_rq2(result, tmp_path / "rq2", scores=scores)
    assert (out / "rq2.json").exists() and (out / "rq2_scores.parquet").exists()
    assert (out / "recommendation.md").exists()  # the decision note the AC asks for
    for table in ("twin_agreement.csv", "proxy_tracking.csv", "label_comparison.csv"):
        assert (out / table).exists(), table

    try:  # figures are the optional viz extra: assert them only where matplotlib is installed
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert set(result.figures) == {"twin_agreement", "proxy_tracking", "label_comparison"}
    for name in ("twin_agreement.png", "proxy_tracking.png", "label_comparison.png"):
        assert (out / name).exists(), name
