from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from preceptx.data.schema import HandoffRecord
from preceptx.measure.featuriser import EncoderConfig, Featuriser


class _StubEncoder:
    """A deterministic, hash-seeded stand-in for SentenceTransformer; counts texts it encodes."""

    dim = 16

    def __init__(self) -> None:
        self.n_encoded = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.n_encoded += len(sentences)
        out = np.zeros((len(sentences), self.dim), dtype=np.float64)
        for i, s in enumerate(sentences):
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out[i] = v / np.linalg.norm(v) if normalize_embeddings else v
        return out


def _cfg(cache_dir: Path) -> EncoderConfig:
    return EncoderConfig(revision="testrev", cache_dir=cache_dir)


def _record(step: int, state_str: str, message: str) -> HandoffRecord:
    return HandoffRecord(
        episode_id="e0",
        step=step,
        condition="C0",
        serialisation="numeric",
        difficulty="hard",
        model="stub",
        seed=0,
        state={},
        state_str=state_str,
        message_raw=message,
        message_delivered=message,
        action={},
        pre_state={},
        post_state={},
        progress=0.0,
        success=False,
        collision=False,
        stuck=False,
    )


def test_embeddings_deterministic_per_text(tmp_path: Path) -> None:
    a = Featuriser(_cfg(tmp_path / "a"), _StubEncoder()).embed_texts(["alpha", "beta"])
    b = Featuriser(_cfg(tmp_path / "b"), _StubEncoder()).embed_texts(["alpha", "beta"])
    assert np.array_equal(a, b)


def test_cache_hit_returns_identical_without_reencoding(tmp_path: Path) -> None:
    stub = _StubEncoder()
    f = Featuriser(_cfg(tmp_path), stub)
    first = f.embed_texts(["x", "y"])
    assert stub.n_encoded == 2
    second = f.embed_texts(["x", "y"])
    assert stub.n_encoded == 2  # both served from cache, no re-encode
    assert np.array_equal(first, second)


def test_partial_cache_only_encodes_misses(tmp_path: Path) -> None:
    stub = _StubEncoder()
    f = Featuriser(_cfg(tmp_path), stub)
    f.embed_texts(["a", "b"])
    assert stub.n_encoded == 2
    f.embed_texts(["a", "b", "c"])
    assert stub.n_encoded == 3  # only "c" is new


def test_featurise_shapes_align_to_records(tmp_path: Path) -> None:
    recs = [_record(i, f"state {i}", f"msg {i}") for i in range(5)]
    e_s, e_m = Featuriser(_cfg(tmp_path), _StubEncoder()).featurise(recs)
    assert e_s.shape == (5, 16)
    assert e_m.shape == (5, 16)


def test_batch_encoding_completes_quickly(tmp_path: Path) -> None:
    recs = [_record(i, f"s{i}", f"m{i}") for i in range(200)]
    start = time.perf_counter()
    Featuriser(_cfg(tmp_path), _StubEncoder()).featurise(recs)
    assert time.perf_counter() - start < 5.0
