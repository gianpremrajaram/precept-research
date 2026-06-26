from __future__ import annotations

import random

import numpy as np
import pytest

from preceptx.seeding import set_global_seed


def test_same_seed_gives_identical_draws() -> None:
    set_global_seed(42)
    a = (random.random(), np.random.rand(3).tolist())
    set_global_seed(42)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_different_seed_gives_different_draws() -> None:
    set_global_seed(1)
    a = np.random.rand(3).tolist()
    set_global_seed(2)
    b = np.random.rand(3).tolist()
    assert a != b


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)
