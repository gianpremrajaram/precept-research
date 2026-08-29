"""Embedding featuriser: pinned, cached, swappable sentence-embeddings for the CPVI stack.

CPVI is computed on *frozen* embeddings of the receiver-observed state (``observation``) and the
delivered message (``message_delivered``); this module turns ``HandoffRecord``s into the aligned
arrays ``e_s`` and ``e_m`` the estimator (DSE-014) consumes, row-for-row in record order. The
conditioning state s is the state observable to the RECEIVER at the handoff - under C3 that is the
windowed view, by design (P0-1): conditioning on A's full ``state_str`` would hand ``g_base`` the
goal/global information the C3 message uniquely carries and floor its CPVI at ~0. The encoder is
revision-pinned and content-hash cached, so the whole sweep re-fits probes on identical vectors
without re-encoding (DEPENDENCIES.md: the encoder is frozen before probes fit). ``sentence-
transformers`` is the optional ``embed`` extra (the only torch puller), so it is imported lazily -
this module loads, and its unit tests run, with an injected stub encoder and no torch installed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord

logger = logging.getLogger(__name__)

# Both encoders are pinned to resolved commit SHAs (DSE-033), verified against the HuggingFace API
# on 24 August 2026. The encoder sits upstream of every CPVI number in the dissertation, so a moving
# branch here would be a moving target underneath a manifest that claims to pin everything; the real
# load path now REFUSES an unpinned revision rather than warning (stub-backed tests never reach it).
_BGE_BASE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"  # BAAI/bge-base-en-v1.5
_MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"  # all-mpnet-base-v2
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")  # a branch or tag name is not a pin


class EncoderConfig(BaseModel):
    """Which sentence-transformer to embed with, pinned by revision, plus the cache location.

    The default is a strong 768-dim retrieval embedder (roadmap §2.4); ``second_encoder`` is a
    different training family (paraphrase/NLI vs retrieval-contrastive) at matching dim, reserved
    for the DSE-022 encoder-sensitivity check. Both carry their own pinned revision. Not yet nested
    into ``ExperimentConfig`` - it is threaded in with the sweep driver (DSE-020), mirroring
    ``GridConfig``/``OutcomeConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="BAAI/bge-base-en-v1.5", min_length=1)
    revision: str = Field(default=_BGE_BASE_REVISION, min_length=1)
    second_encoder: str = Field(default="sentence-transformers/all-mpnet-base-v2", min_length=1)
    second_encoder_revision: str = Field(default=_MPNET_REVISION, min_length=1)
    batch_size: int = Field(default=32, gt=0)
    normalize: bool = True
    cache_dir: Path = Field(default=Path(".embed_cache"))
    # Pinned, not auto-selected. sentence-transformers picks the "best" backend it finds, which on
    # Apple Silicon is MPS - and MPS returns *substantively different* vectors for the same string
    # depending on which batch it lands in: measured on torch 2.10.0 / sentence-transformers 5.6.0,
    # a text repeated 64 times across a 32-wide batch boundary encoded to vectors whose cosine to
    # the first fell to 0.543, with 62 of 64 rows below 0.999. CPU on the identical call holds
    # cosine at 0.999999999999 (~1.8e-07 of elementwise float32 jitter, which is the real floor).
    # A 0.46 cosine gap is a different vector, not noise, and it sits upstream of every CPVI so
    # the device is a reproducibility parameter and defaults to the backend verified deterministic.
    # Override for a CUDA node only after running the determinism check in tests/unit/measure/.
    device: str = Field(default="cpu", min_length=1)


class EncoderBackend(Protocol):
    """The slice of ``SentenceTransformer`` the featuriser uses; lets a stub stand in for tests."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]: ...


class Featuriser:
    """Embeds ``observation``/``message_delivered`` with a pinned encoder, caching by content hash.

    The encoder is injected (a stub in tests) or lazily constructed from ``cfg`` on first use, so
    importing this module never requires ``sentence-transformers``/torch. The cache is content-
    addressed by ``(encoder name, revision, text)``, so one cache dir is safe to share across the
    whole sweep AND across encoders (P1-16: omitting the name would serve one encoder's vectors to
    another whose revision string matches, fabricating a near-zero encoder-sensitivity result).
    """

    def __init__(
        self, cfg: EncoderConfig | None = None, encoder: EncoderBackend | None = None
    ) -> None:
        self.cfg = cfg or EncoderConfig()
        self._encoder = encoder

    def _backend(self) -> EncoderBackend:
        if self._encoder is None:
            self._encoder = self._load()
        return self._encoder

    def _load(self) -> EncoderBackend:
        if not _COMMIT_SHA.match(self.cfg.revision):
            raise ConfigError(
                f"encoder {self.cfg.name!r} would load at unpinned revision "
                f"{self.cfg.revision!r}; a result with an unrecorded revision is not a result - "
                "pin the 40-character commit SHA"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # the embed extra is optional, kept out of core/CI
            raise ImportError(
                "the featuriser needs the 'embed' extra: install with "
                "`uv pip install -e '.[embed]'`"
            ) from exc
        model = SentenceTransformer(
            self.cfg.name, revision=self.cfg.revision, device=self.cfg.device
        )
        return model  # type: ignore[no-any-return]  # untyped import duck-types EncoderBackend

    def _cache_path(self, text: str) -> Path:
        key = f"{self.cfg.name}\x00{self.cfg.revision}\x00{text}"
        return self.cfg.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.npy"

    def embed_texts(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed ``texts`` to ``(len(texts), dim)``, serving cache hits and encoding only misses.

        Deduplicated before encoding, then fanned back out. This is a determinism property, not an
        optimisation: encoding is float32 and batch-order sensitive at ~1.8e-07, so encoding one
        string once per occurrence gives its occurrences *slightly different* vectors, while the
        content-addressed cache stores exactly one - which made a cold run and every later warm run
        disagree, enough to reorder near-ties and move MRR in the fourth decimal. Encoding each
        unique text once makes cold and warm identical by construction, on any backend. It is also
        markedly cheaper on real corpora: TraceElephant's 2,488 handoff messages are 1,166 unique
        strings, one of them repeated 316 times.

        Indexing the output by text rather than by position also removes a latent misalignment: the
        previous ``[v for v in vectors if v is not None]`` would have silently returned *fewer rows
        than inputs*, shifting every downstream pairing, had any slot gone unfilled.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        vectors: dict[str, NDArray[np.float64]] = {}
        misses: list[str] = []
        for text in dict.fromkeys(texts):  # unique, first-seen order; one stat per unique text
            path = self._cache_path(text)
            if path.exists():
                vectors[text] = np.load(path)
            else:
                misses.append(text)
        if misses:
            encoded: NDArray[np.float64] = (
                self._backend()
                .encode(
                    misses,
                    batch_size=self.cfg.batch_size,
                    normalize_embeddings=self.cfg.normalize,
                    convert_to_numpy=True,
                )
                .astype(np.float64)
            )
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            for text, vector in zip(misses, encoded, strict=True):
                vectors[text] = vector
                np.save(self._cache_path(text), vector)
        return np.vstack([vectors[t] for t in texts]).astype(np.float64)

    def featurise(
        self, records: list[HandoffRecord]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(e_s, e_m)`` - state/message embeddings, row-aligned to ``records``.

        ``e_s`` embeds the RECEIVER's observation (P0-1 conditioning semantics), not A's full
        ``state_str``; the two coincide except under C3, where the restricted window is the point.
        Every downstream consumer (estimator, twin, runtime statistics, calibration, G2) inherits
        the semantics through this one choke point.
        """
        e_s = self.embed_texts([r.observation for r in records])
        e_m = self.embed_texts([r.message_delivered for r in records])
        return e_s, e_m


def second_encoder_config(cfg: EncoderConfig) -> EncoderConfig:
    """The sensitivity encoder promoted to primary, for the DSE-022 side-by-side rescore.

    Same cache directory on purpose: cache keys already include the encoder *name* as well as its
    revision (P1-16), so the two encoders' vectors cannot collide, and sharing the directory keeps
    one cache to warm. ``second_encoder``/``second_encoder_revision`` are carried through unchanged
    so the returned config still records which pair it came from.
    """
    return cfg.model_copy(
        update={"name": cfg.second_encoder, "revision": cfg.second_encoder_revision}
    )
