"""Centralised seeding for Python, NumPy, and (if installed) torch.

Call ``set_global_seed`` once at run start. This pins the host-side RNGs; it does NOT make LLM
inference bit-exact. Batched vLLM inference is not reproducible across runs even at a fixed seed
(non-associative float reductions, kernel non-determinism), so determinism in this repo means
"low-variance, seed-pinned, revision-pinned", never "exactly reproducible" - the determinism harness
quantifies the residual variance honestly rather than hiding it.

Note: ``PYTHONHASHSEED`` governs hash randomisation and can only be set before interpreter start, so
it is out of scope here; set it in the run entry point's environment if hash-stable runs are needed.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and torch (if present). torch absence is logged, not fatal."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding (core env is torch-free)")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
