"""Synthetic CPVI ground truth: messages built to carry / not carry information beyond the state.

The mandated determinism / known-answer fixture (CLAUDE.md): the estimator and twin tests assert
CPVI ~ 0 on a noise message, CPVI > 0 on an informative one, and PVI > CPVI on a state-echo message.
Four handoffs share an episode so the group-aware cross-fit is exercised.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Case = Literal["noise", "informative", "echo"]

Arrays = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]
ContArrays = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]


def _groups(n: int) -> NDArray[np.int_]:
    return np.repeat(np.arange(n // 4), 4)[:n].astype(int)


def make_binary(case: Case, *, n: int = 400, d: int = 8, seed: int = 0) -> Arrays:
    rng = np.random.default_rng(seed)
    e_s = rng.standard_normal((n, d))
    logit = e_s @ rng.standard_normal(d)  # the state always carries some signal
    if case == "noise":
        e_m = rng.standard_normal((n, d))  # message carries nothing
    elif case == "informative":
        e_m = rng.standard_normal((n, d))
        logit = logit + 3.0 * (e_m @ rng.standard_normal(d))  # signal only in the message
    else:  # echo: message is the state plus a little noise
        e_m = e_s + 0.05 * rng.standard_normal((n, d))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return e_s.astype(np.float64), e_m.astype(np.float64), y, _groups(n)


def make_continuous(case: Case, *, n: int = 400, d: int = 8, seed: int = 0) -> ContArrays:
    rng = np.random.default_rng(seed)
    e_s = rng.standard_normal((n, d))
    y = e_s @ rng.standard_normal(d)
    if case == "noise":
        e_m = rng.standard_normal((n, d))
    elif case == "informative":
        e_m = rng.standard_normal((n, d))
        y = y + 3.0 * (e_m @ rng.standard_normal(d))
    else:  # echo
        e_m = e_s + 0.05 * rng.standard_normal((n, d))
    y = y + 0.3 * rng.standard_normal(n)  # observation noise
    return e_s.astype(np.float64), e_m.astype(np.float64), y.astype(np.float64), _groups(n)
