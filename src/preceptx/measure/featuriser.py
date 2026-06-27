"""Embedding featuriser: pinned, cached, swappable sentence-embeddings for the CPVI stack.

CPVI is computed on *frozen* embeddings of the serialised state (``state_str``) and the delivered
message (``message_delivered``); this module turns ``HandoffRecord``s into the aligned arrays
``e_s`` and ``e_m`` the estimator (DSE-014) consumes, row-for-row in record order. The encoder is
revision-pinned and content-hash cached, so the whole sweep re-fits probes on identical vectors
without re-encoding (DEPENDENCIES.md: the encoder is frozen before probes fit). ``sentence-
transformers`` is the optional ``embed`` extra (the only torch puller), so it is imported lazily -
this module loads, and its unit tests run, with an injected stub encoder and no torch installed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import HandoffRecord

logger = logging.getLogger(__name__)

# Revisions default to a moving branch and MUST be pinned to a commit SHA before the Phase-2 freeze;
# the real-encoder load path warns until they are (the stub-backed unit tests never trip it).
_UNPINNED = "main"


class EncoderConfig(BaseModel):
    """Which sentence-transformer to embed with, pinned by revision, plus the cache location.

    The default is a strong 768-dim retrieval embedder (roadmap §2.4); ``second_encoder`` is a
    different training family (paraphrase/NLI vs retrieval-contrastive) at matching dim, reserved
    for the DSE-022 encoder-sensitivity check. Not yet nested into ``ExperimentConfig`` - it is
    threaded in with the sweep driver (DSE-020), mirroring ``GridConfig``/``OutcomeConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="BAAI/bge-base-en-v1.5", min_length=1)
    revision: str = Field(default=_UNPINNED, min_length=1)
    second_encoder: str = Field(default="sentence-transformers/all-mpnet-base-v2", min_length=1)
    batch_size: int = Field(default=32, gt=0)
    normalize: bool = True
    cache_dir: Path = Field(default=Path(".embed_cache"))


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
    """Embeds ``state_str``/``message_delivered`` with a pinned encoder, caching by content hash.

    The encoder is injected (a stub in tests) or lazily constructed from ``cfg`` on first use, so
    importing this module never requires ``sentence-transformers``/torch. The cache is content-
    addressed by ``(revision, text)``, so one cache dir is safe to share across the whole sweep.
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
        if self.cfg.revision == _UNPINNED:
            logger.warning(
                "encoder %s loaded at unpinned revision %r; pin the commit SHA before the freeze",
                self.cfg.name,
                self.cfg.revision,
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # the embed extra is optional, kept out of core/CI
            raise ImportError(
                "the featuriser needs the 'embed' extra: install with "
                "`uv pip install -e '.[embed]'`"
            ) from exc
        model = SentenceTransformer(self.cfg.name, revision=self.cfg.revision)
        return model  # type: ignore[no-any-return]  # untyped import duck-types EncoderBackend

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.cfg.revision}\x00{text}".encode()).hexdigest()
        return self.cfg.cache_dir / f"{digest}.npy"

    def embed_texts(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed ``texts`` to ``(len(texts), dim)``, serving cache hits and encoding only misses."""
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        vectors: list[NDArray[np.float64] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        for i, text in enumerate(texts):  # one stat per text; load hits inline, collect misses
            path = self._cache_path(text)
            if path.exists():
                vectors[i] = np.load(path)
            else:
                miss_idx.append(i)
        if miss_idx:
            encoded: NDArray[np.float64] = (
                self._backend()
                .encode(
                    [texts[i] for i in miss_idx],
                    batch_size=self.cfg.batch_size,
                    normalize_embeddings=self.cfg.normalize,
                    convert_to_numpy=True,
                )
                .astype(np.float64)
            )
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            for j, i in enumerate(miss_idx):
                vectors[i] = encoded[j]
                np.save(self._cache_path(texts[i]), encoded[j])
        return np.vstack([v for v in vectors if v is not None]).astype(np.float64)

    def featurise(
        self, records: list[HandoffRecord]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(e_s, e_m)`` - state/message embeddings, row-aligned to ``records``."""
        e_s = self.embed_texts([r.state_str for r in records])
        e_m = self.embed_texts([r.message_delivered for r in records])
        return e_s, e_m
