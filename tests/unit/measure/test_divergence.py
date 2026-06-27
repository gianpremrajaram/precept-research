from __future__ import annotations

import numpy as np

from preceptx.measure.divergence import embedding_cosine, jsd


def test_jsd_zero_for_identical_distributions() -> None:
    p = np.array([[0.3, 0.7], [0.5, 0.5]])
    assert np.allclose(jsd(p, p), 0.0, atol=1e-9)


def test_jsd_one_bit_for_disjoint_binary() -> None:
    assert np.allclose(jsd(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])), 1.0, atol=1e-6)


def test_jsd_is_symmetric() -> None:
    p = np.array([[0.2, 0.8]])
    q = np.array([[0.6, 0.4]])
    assert np.allclose(jsd(p, q), jsd(q, p))


def test_embedding_cosine_known_vectors() -> None:
    e_m = np.array([[1.0, 0.0], [1.0, 1.0]])
    e_s = np.array([[1.0, 0.0], [0.0, 1.0]])  # identical, then 45 degrees
    cos = embedding_cosine(e_m, e_s)
    assert np.isclose(cos[0], 1.0)
    assert np.isclose(cos[1], 1.0 / np.sqrt(2.0), atol=1e-6)
