"""Score-blind blocking controls for the causal arm (DSE-018).

The causal claim needs null arms that block *without looking at the handoff*, so an improvement in
the gate-active arm cannot be explained by blocking (and the extra A turn it buys) per se - this is
the Lowe et al. (2019) demand made concrete (roadmap §3.5). Both controls are seeded Bernoulli
draws keyed on the episode seed and the step, so a control arm is exactly as reproducible as the
episode it runs in and blocks the same handoffs on a re-run.

``matched_random`` draws at the gate's own calibrated firing rate (DSE-017), so it blocks the same
count as the real gate on the same episodes; ``random_trigger`` draws at a fixed configured rate,
and at rate 1.0 it is the preregistration's "always-retry" arm.
"""

from __future__ import annotations

import numpy as np

# Distinct salts keep the two controls off each other's stream: at equal rates a shared salt would
# block byte-identical handoffs and collapse the two arms into one. Both sit above graph.py's
# jitter salt (2**16) and outside the channel's two-element [seed, step] keys, so no gate draw can
# alias a scenario pose or a C4 dropout mask.
_MATCHED_SALT = 2**17
_RANDOM_SALT = 2**18


def _draw(rate: float, salt: int, seed: int, gate_seed: int, step: int) -> bool:
    """Block iff a seeded uniform falls below ``rate``. Never sees the message or the state."""
    return bool(np.random.default_rng([seed, salt, gate_seed, step]).random() < rate)


def matched_random(firing_rate: float, *, seed: int, gate_seed: int, step: int) -> bool:
    """Block at the gate's calibrated firing rate - the same count, the wrong handoffs.

    ``firing_rate`` is ``StatisticCalibration.firing_rate``, i.e. the rate the real gate was
    measured to fire at on the calibration set, so the counts match in expectation over the same
    episodes.

    ponytail: rate-matched per handoff, not count-matched per episode. An exact per-episode count
    needs the episode length up front, which early termination on success denies. Draw the block
    positions from the realised gate-active episodes if H6 ever turns on the difference.
    """
    return _draw(firing_rate, _MATCHED_SALT, seed, gate_seed, step)


def random_trigger(rate: float, *, seed: int, gate_seed: int, step: int) -> bool:
    """Block at a fixed rate, independent of both the score and the gate's own firing rate."""
    return _draw(rate, _RANDOM_SALT, seed, gate_seed, step)
