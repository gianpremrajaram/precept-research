"""The runtime gate at the A->B boundary (DSE-018).

``RuntimeGate`` turns a calibrated statistic (DSE-016/DSE-017) into an intervention: it scores the
pair the measurement stack scores - the receiver's ``observation`` and the *post-channel*
``message_delivered`` - and blocks the handoff when the oriented score crosses the calibrated
threshold. The runner re-prompts A on a block (bounded retries) and records the block, so the gate
changes the run rather than merely annotating it. One config flag selects the four arms of the H6
contrast: ``off``, ``active``, and the two score-blind controls in ``gate.controls``.

**This module fails OPEN, and that is deliberate.** Everywhere else in this repo a missing input is
an exception, because a passing-looking broken run is the worst outcome (CLAUDE.md). Here the
inverse holds and the ticket says so: a gate with no statistic, no calibration or no encoder must
degrade to the ungated loop and log a WARNING, never crash the episode. The reason is that an
absent gate is a *valid arm of the experiment* (it is literally the ``off`` control), so failing
open lands the run in a defined cell of the design rather than destroying it. What does NOT fail
open is a wiring mistake: a statistic or calibration whose key disagrees with the configured one is
raised at construction time, outside the episode loop, because applying ``cosine``'s threshold to
``info``'s score is a wrong number rather than a missing one.

Orientation is taken from ``StatisticCalibration``, never assumed. ``oriented = orientation * raw``
(higher = more failure-risk), and the calibration object carries the sign, so the threshold and
the sign that makes it mean anything cannot be separated - the rule ``rq3a.transfer_scores``
enforces, here made structural by injecting the calibration rather than a bare float.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from preceptx.gate.calibration import StatisticCalibration
from preceptx.gate.controls import matched_random, random_trigger
from preceptx.gate.statistics import GateError, Statistic, resolve_statistic_key
from preceptx.measure.featuriser import Featuriser

logger = logging.getLogger(__name__)

GateMode = Literal["off", "active", "matched_random", "random_trigger"]
"""The four arms of H6: no gate, the calibrated gate, and the two score-blind controls."""


class GateConfig(BaseModel):
    """Which arm to run and how hard to push back; the calibrated numbers come from DSE-017."""

    model_config = ConfigDict(extra="forbid")

    mode: GateMode = "off"
    statistic_key: str = Field(default="cosine", min_length=1)

    @field_validator("statistic_key")
    @classmethod
    def _resolve_retired_key(cls, v: str) -> str:
        """Accept a retired key and resolve it, so pre-DSE-061 configs still load (DSE-061)."""
        return resolve_statistic_key(v)

    # One re-prompt by default: the cheapest bound that makes the treatment non-vacuous, and it
    # caps the arm's extra A calls at the firing rate rather than a multiple of it. 0 is a valid
    # and useful setting - it records every block without ever re-prompting (observation-only).
    max_retries: int = Field(default=1, ge=0)
    random_rate: float = Field(default=0.2, ge=0.0, le=1.0)  # 1.0 = the "always-retry" arm
    # Salted into the control draws so a control arm can be re-realised (different blocked handoffs)
    # without touching the episode seed, which would also move the physics scenario.
    seed: int = Field(default=0, ge=0)


class GateDecision(BaseModel):
    """One gate verdict; ``score``/``threshold`` are None in the modes that never score."""

    model_config = ConfigDict(extra="forbid")

    blocked: bool
    reason: str
    score: float | None = None
    threshold: float | None = None


class RuntimeGate:
    """Scores a handoff and decides whether to block it. Holds the injected measurement stack.

    ``statistic``/``calibration``/``featuriser`` are optional so every degraded wiring lands on the
    fail-open path rather than at a construction site that cannot recover; ``mode="active"`` is the
    only arm that needs all three.
    """

    def __init__(
        self,
        cfg: GateConfig,
        *,
        statistic: Statistic | None = None,
        calibration: StatisticCalibration | None = None,
        featuriser: Featuriser | None = None,
    ) -> None:
        for name, key in (
            ("statistic", getattr(statistic, "key", None)),
            ("calibration", None if calibration is None else calibration.key),
        ):
            if key is not None and key != cfg.statistic_key:
                raise GateError(
                    f"gate {name} is {key!r} but GateConfig.statistic_key is "
                    f"{cfg.statistic_key!r}; a threshold calibrated for one statistic applied to "
                    "another is a wrong number, not a missing one"
                )
        self.cfg = cfg
        self._stat = statistic
        self._cal = calibration
        self._featuriser = featuriser
        self._warned = False

    @property
    def max_retries(self) -> int:
        """Re-prompt budget the runner honours on a block."""
        return self.cfg.max_retries

    def _fail_open(self, reason: str) -> GateDecision:
        """Pass the handoff through, warning once per gate rather than once per handoff."""
        if not self._warned:
            logger.warning("runtime gate failing open in mode %r: %s", self.cfg.mode, reason)
            self._warned = True
        return GateDecision(blocked=False, reason=reason)

    def decide(self, observation: str, message: str, *, seed: int, step: int) -> GateDecision:
        """Verdict for one handoff, on the RECEIVER's post-channel ``observation``/``message``.

        ``seed`` is the episode's seed and ``step`` its index; together they key the control draws,
        so the control arms are reproducible per handoff. Neither reaches the active path, which is
        a pure function of the scored pair.
        """
        mode = self.cfg.mode
        if mode == "off":
            return GateDecision(blocked=False, reason="gate off")
        if mode == "random_trigger":
            blocked = random_trigger(
                self.cfg.random_rate, seed=seed, gate_seed=self.cfg.seed, step=step
            )
            return GateDecision(blocked=blocked, reason="random-trigger control")
        if mode == "matched_random":
            if self._cal is None:
                return self._fail_open("matched-random control has no calibrated firing rate")
            blocked = matched_random(
                self._cal.firing_rate, seed=seed, gate_seed=self.cfg.seed, step=step
            )
            return GateDecision(blocked=blocked, reason="matched-firing-rate control")
        # mode == "active"
        if self._stat is None or self._cal is None or self._featuriser is None:
            return self._fail_open(
                "active gate needs a fitted statistic, its calibration and an encoder; "
                f"missing {'statistic ' if self._stat is None else ''}"
                f"{'calibration ' if self._cal is None else ''}"
                f"{'featuriser' if self._featuriser is None else ''}".strip()
            )
        # One batched encode: the two texts share a cache and a forward pass.
        vecs = self._featuriser.embed_texts([observation, message])
        score = float(self._cal.orientation * self._stat.score(vecs[:1], vecs[1:])[0])
        return GateDecision(
            blocked=score >= self._cal.threshold,  # matches calibration's "block score >= t"
            reason="active gate",
            score=score,
            threshold=self._cal.threshold,
        )
