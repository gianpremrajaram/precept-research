"""Divergence proxies between the two probes' predictive distributions.

The Jensen-Shannon divergence between ``g_cond`` and ``g_base`` per handoff is the cheap, *bounded*
bridge to the runtime proxy (DSE-016) - symmetric and in ``[0, 1]`` bits for two classes, unlike the
asymmetric, unbounded KL the prospective twin uses. The message-vs-state embedding cosine is its
probe-independent cousin (low cosine = the message points somewhere the state does not). Both are
returned row-for-row aligned to the handoffs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import jensenshannon

FloatArray = NDArray[np.float64]


def jsd(p_cond: FloatArray, p_base: FloatArray) -> FloatArray:
    """Per-row Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint for two classes)."""
    # scipy returns the JS *distance* (sqrt of the divergence); square it back to the divergence.
    dist: FloatArray = np.array(
        [jensenshannon(a, b, base=2.0) for a, b in zip(p_cond, p_base, strict=True)],
        dtype=np.float64,
    )
    return np.nan_to_num(dist) ** 2


def embedding_cosine(e_m: FloatArray, e_s: FloatArray) -> FloatArray:
    """Per-row cosine between message and state embeddings - the state-echo bridge statistic."""
    num: FloatArray = np.sum(e_m * e_s, axis=1)
    den: FloatArray = np.linalg.norm(e_m, axis=1) * np.linalg.norm(e_s, axis=1) + 1e-12
    return num / den
