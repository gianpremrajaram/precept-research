from __future__ import annotations

import hashlib
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord
from preceptx.measure.featuriser import EncoderConfig, Featuriser, second_encoder_config


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


def _record(
    step: int, state_str: str, message: str, *, observation: str | None = None
) -> HandoffRecord:
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
        observation=state_str if observation is None else observation,
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


def test_featurise_embeds_receiver_observation_not_full_state(tmp_path: Path) -> None:
    # The P0-1 conditioning semantics: e_s is the receiver's (possibly C3-restricted) view.
    f = Featuriser(_cfg(tmp_path), _StubEncoder())
    rec = _record(0, "full state with goal", "msg", observation="windowed view")
    e_s, _ = f.featurise([rec])
    assert np.array_equal(e_s, f.embed_texts(["windowed view"]))
    assert not np.array_equal(e_s, f.embed_texts(["full state with goal"]))


def test_cache_key_separates_encoders_by_name(tmp_path: Path) -> None:
    # P1-16: two encoders at the same revision sharing one cache dir must never serve each other's
    # vectors - that would fabricate a near-zero encoder-sensitivity result in DSE-022.
    shared = tmp_path / "shared_cache"
    stub_a, stub_b = _StubEncoder(), _StubEncoder()
    Featuriser(EncoderConfig(name="enc/A", revision="main", cache_dir=shared), stub_a).embed_texts(
        ["same text"]
    )
    Featuriser(EncoderConfig(name="enc/B", revision="main", cache_dir=shared), stub_b).embed_texts(
        ["same text"]
    )
    assert stub_a.n_encoded == 1
    assert stub_b.n_encoded == 1  # a cache hit here would mean enc/B was served enc/A's vector


def test_batch_encoding_completes_quickly(tmp_path: Path) -> None:
    recs = [_record(i, f"s{i}", f"m{i}") for i in range(200)]
    start = time.perf_counter()
    Featuriser(_cfg(tmp_path), _StubEncoder()).featurise(recs)
    assert time.perf_counter() - start < 5.0


# --- DSE-033: the encoder is pinned, and an unpinned one cannot reach a recorded run ------------


def test_default_revisions_are_resolved_commit_shas() -> None:
    cfg = EncoderConfig()
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.revision)
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.second_encoder_revision)


def test_unpinned_revision_raises_on_the_real_load_path(tmp_path: Path) -> None:
    # No injected encoder, so embedding falls through to the real load path - which must refuse a
    # branch name rather than warn and carry on into a recorded run.
    f = Featuriser(EncoderConfig(revision="main", cache_dir=tmp_path))
    with pytest.raises(ConfigError, match="unpinned revision"):
        f.embed_texts(["anything"])


def test_stub_path_is_unaffected_by_the_pin_check(tmp_path: Path) -> None:
    f = Featuriser(EncoderConfig(revision="main", cache_dir=tmp_path), _StubEncoder())
    assert f.embed_texts(["anything"]).shape == (1, 16)


def test_second_encoder_config_promotes_the_sensitivity_encoder(tmp_path: Path) -> None:
    # DSE-022 rescores everything under the second encoder; promoting it must swap BOTH the name and
    # the revision, or the sensitivity arm silently re-runs the primary and reports perfect
    # agreement. The pair is carried through unchanged so the config still records where it came
    # from, and the shared cache dir is safe because keys include the name (P1-16, tested above).
    primary = EncoderConfig(cache_dir=tmp_path / "c")
    second = second_encoder_config(primary)
    assert second.name == primary.second_encoder != primary.name
    assert second.revision == primary.second_encoder_revision != primary.revision
    assert second.second_encoder == primary.second_encoder  # the pair survives the promotion
    assert second.cache_dir == primary.cache_dir


# ------------------------------------------------------------------ device determinism (DSE-064)


def test_the_pinned_device_encodes_a_repeated_text_identically() -> None:
    """The regression guard for the MPS divergence found while freezing the first RQ3a result.

    sentence-transformers auto-selects a backend, and on Apple Silicon that is MPS, which returns
    *substantively different* vectors for the same string depending on which batch it lands in -
    cosine 0.543 to the first row, 62 of 64 rows below 0.999, on torch 2.10.0. Not float jitter,
    and upstream of every CPVI number. The corpus makes it reachable rather than theoretical:
    TraceElephant's messages are 1,166 unique strings over 2,488 slots, one of them repeated 316
    times, so duplicates straddle batch boundaries constantly.

    The assertion is on cosine, not bit-equality: CPU still carries ~1.8e-07 of elementwise float32
    batch-order jitter, which is harmless and moves cosine by ~1e-12. The threshold sits far above
    that and far below the MPS failure, so it separates "float noise" from "different vector"
    rather than pinning an exact float. Loads the real encoder on purpose - a stub cannot exhibit a
    backend's bug.
    """
    pytest.importorskip("sentence_transformers")
    cfg = EncoderConfig(cache_dir=Path(tempfile.mkdtemp()))
    assert cfg.device == "cpu", "the default device must stay the one verified deterministic"
    rows = Featuriser(cfg).embed_texts(["InformationExtraction_Expert"] * (2 * cfg.batch_size))
    cosine = rows @ rows[0] / (np.linalg.norm(rows, axis=1) * np.linalg.norm(rows[0]))
    assert cosine.min() > 1.0 - 1e-6, (
        f"the same string encoded to different vectors across a batch boundary "
        f"(min cosine {cosine.min():.6f}); on MPS this reads ~0.54"
    )


def test_a_repeated_text_is_encoded_once_not_once_per_occurrence() -> None:
    """Deduplication is a determinism property, not just a saving.

    Encoding a string once per occurrence puts its occurrences in different batch slots, and
    float32 batch-order jitter (~1.8e-07 even on CPU) then gives them slightly different vectors,
    while the content-addressed cache stores exactly one. That is what made a cold run and every
    later warm run disagree, by enough to reorder near-ties and move MRR in the fourth decimal.
    """
    seen: list[list[str]] = []

    class _Counting:
        def encode(
            self,
            sentences: list[str],
            *,
            batch_size: int,
            normalize_embeddings: bool,
            convert_to_numpy: bool,
        ) -> NDArray[np.float64]:
            seen.append(list(sentences))
            return np.array([[float(len(s)), 1.0] for s in sentences], dtype=np.float64)

    cfg = EncoderConfig(cache_dir=Path(tempfile.mkdtemp()))
    texts = ["a", "bb", "a", "a", "bb"]
    rows = Featuriser(cfg, encoder=_Counting()).embed_texts(texts)

    assert seen == [["a", "bb"]], "the encoder saw a repeated text more than once"
    assert rows.shape == (len(texts), 2), "the fan-out must restore one row per input"
    assert [r[0] for r in rows] == [1.0, 2.0, 1.0, 1.0, 2.0], "rows are misaligned to their inputs"
