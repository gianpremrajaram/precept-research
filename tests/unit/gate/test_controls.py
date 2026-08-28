from __future__ import annotations

import numpy as np

from preceptx.gate.controls import matched_random, random_trigger


def _rate(fn, rate: float, *, seed: int = 0, gate_seed: int = 0, n: int = 4000) -> float:  # type: ignore[no-untyped-def]
    return float(np.mean([fn(rate, seed=seed, gate_seed=gate_seed, step=s) for s in range(n)]))


def test_draws_are_reproducible_per_handoff() -> None:
    # A control arm must be as re-runnable as the episode it runs in: same key, same block.
    keys = {"seed": 7, "gate_seed": 1, "step": 3}
    assert matched_random(0.5, **keys) == matched_random(0.5, **keys)
    assert random_trigger(0.5, **keys) == random_trigger(0.5, **keys)


def test_degenerate_rates_are_exact() -> None:
    for step in range(50):
        assert not matched_random(0.0, seed=0, gate_seed=0, step=step)
        assert random_trigger(1.0, seed=0, gate_seed=0, step=step)  # the "always-retry" arm


def test_empirical_rate_matches_the_configured_rate() -> None:
    # The whole point of the matched control: it blocks the gate's COUNT, not its handoffs.
    for rate in (0.1, 0.2, 0.5):
        assert abs(_rate(matched_random, rate) - rate) < 0.02
        assert abs(_rate(random_trigger, rate) - rate) < 0.02


def test_the_two_controls_do_not_share_a_stream() -> None:
    # Distinct salts: at equal rates a shared stream would block byte-identical handoffs and
    # collapse the matched-firing-rate arm and the random-trigger arm into one arm.
    keys = [{"seed": 0, "gate_seed": 0, "step": s} for s in range(200)]
    m = [matched_random(0.5, **k) for k in keys]
    r = [random_trigger(0.5, **k) for k in keys]
    assert m != r


def test_gate_seed_re_realises_the_arm_without_moving_the_episode() -> None:
    # Re-drawing the blocked positions must not require changing the episode seed, which would
    # also move the physics scenario (graph.py keys the jitter on it).
    a = [matched_random(0.5, seed=3, gate_seed=0, step=s) for s in range(200)]
    b = [matched_random(0.5, seed=3, gate_seed=1, step=s) for s in range(200)]
    assert a != b


def test_step_indexes_the_stream() -> None:
    draws = [random_trigger(0.5, seed=0, gate_seed=0, step=s) for s in range(60)]
    assert len(set(draws)) == 2  # not a constant per episode
