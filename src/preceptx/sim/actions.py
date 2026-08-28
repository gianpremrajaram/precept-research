"""Macro-action interface, the settle-step, state read-back, and stuck/collision detection.

Agents act through a discrete macro-action (the primary design); a two-grip force-handle interface
sits behind a flag. Each macro action applies one impulse and then settles the space under damping.
In the default quasi-static regime residual velocity is zeroed after settling, so the load is nudged
and comes to rest each turn (matching the damped top-down task) and inverse actions cancel cleanly.
"""

from __future__ import annotations

import math
from typing import Literal

import pymunk
from pydantic import BaseModel, ConfigDict, Field

from preceptx.sim.arena import DAMPING, LOAD_MASS
from preceptx.sim.load import BAR_LEN, add_load

MacroAction = Literal["N", "S", "E", "W", "ROT+", "ROT-", "WAIT"]

# The rotation quantum, in degrees per ROT action. This is the AUTHORED quantity; the impulse below
# is derived from it. That direction matters: the previous default was an impulse authored directly
# ("0.5, sized for ~34 deg against the T"), and when DSE-057 swapped the T for a bar of 1.71x
# smaller
# moment the same impulse silently became 57.8 deg. A comment cannot notice that; a test can.
#
# Sizing is NOT "step < window". The bar is symmetric under a half-turn and both rotate directions
# are available, so the set reachable in k actions is the lattice `theta0 + m*step (mod 180)` for
# |m| <= k, and what matters is whether that ORBIT enters the passing window - which is not monotone
# in the step (9.25 deg leaves the hard rung unreachable where 11.7 deg reaches it). At 12.0 deg
# every jittered start reaches every rung's window, in a median of 4/6/7 rotations, and every round
# step from 8 to 20 deg also works - so this sits inside a broad basin, not on a resonance.
ROTATION_STEP_DEG = 12.0

# Realised degrees of rotation per unit of angular impulse, for the active load under the shipped
# damping and settle schedule. MEASURED, not derived: the continuum integral predicts 115.2 and the
# discrete solver realises 115.582, and it is the solver that runs the experiment. Free rotation is
# exactly deterministic (sd 0.000000 across 37 start angles at seven impulses), so this constant is
# exact rather than an average. ``measure_rotation_step`` re-measures it; the unit test asserts the
# shipped impulse still realises ROTATION_STEP_DEG, which is what makes this safe to hard-code.
DEG_PER_UNIT_IMPULSE = 115.582
ANGULAR_IMPULSE = ROTATION_STEP_DEG / DEG_PER_UNIT_IMPULSE

# Local-frame grip points for the force-handle interface: the two ends of the bar.
_GRIP_LEFT = (-BAR_LEN / 2.0, 0.0)
_GRIP_RIGHT = (BAR_LEN / 2.0, 0.0)


class StepConfig(BaseModel):
    """Stepping and impulse parameters. Defaults chosen for stability under thin walls and damping.

    ``dt`` is split into ``substeps`` substeps per settle step to avoid tunnelling; ``settle_steps``
    settle steps run after each impulse. ``quasi_static`` zeroes residual velocity once settled, and
    ``hold_orientation`` restores the pre-action angle after any non-rotate action.
    """

    model_config = ConfigDict(extra="forbid")

    dt: float = Field(default=1.0 / 60.0, gt=0)
    substeps: int = Field(default=4, ge=1)
    settle_steps: int = Field(default=30, ge=1)
    # Two grips holding a rigid load hold its ORIENTATION too: pushing does not spin it, and the
    # angle changes only when the pair deliberately rotates it. Implemented by giving the body an
    # INFINITE MOMENT for the duration of a non-rotate action, so no contact torque can spin it.
    #
    # It closes the third degeneracy (DSE-058): macro impulses are applied at the COM and carry no
    # torque, but contact at the channel mouth rotated the load anyway, so a translation-only ACTION
    # sequence produced a large STATE rotation and threaded the channel with no rotate command ever
    # issued - up to 114 deg, leaking on ~27% of jittered starts, and CHAOTICALLY in both aperture
    # and start pose, so no width could tune it away.
    #
    # DSE-059: it did not close that degeneracy. The original implementation RESTORED the angle
    # after
    # the settle, which prevents the rotation from being *recorded* but not from *happening*. From a
    # 30 deg start on medium (geometric window +/-17.2 deg) the load reached 0.48 deg mid-action,
    # slipped through the channel, and was written back as 30.00 deg. Two consequences: apertures
    # were softer than certified (medium's realised window was +/-32.6 deg, not +/-17.2), and the
    # recorded angle was not the angle at which the load passed the gap - so the state the messages
    # described was not the state that decided the outcome.
    #
    # Infinite moment prevents it instead of masking it. Measured against geometric fit across seven
    # apertures: restore-after errs by up to 20.03 deg and is NON-MONOTONE in aperture (so
    # difficulty
    # did not grade cleanly); zeroing angular velocity each substep errs by 1.12 deg; infinite
    # moment
    # errs by 0.01 deg and is monotone. Angle drift over ten pushes is exactly 0.0.
    hold_orientation: bool = True
    linear_impulse: float = Field(default=3.0, gt=0)
    # Derived from ROTATION_STEP_DEG, not authored directly - see that constant for the rationale
    # and
    # ``tests/unit/sim/test_actions.py`` for the assertion that this value still realises it.
    angular_impulse: float = Field(default=ANGULAR_IMPULSE, gt=0)
    quasi_static: bool = True


class BodyState(BaseModel):
    """Read-back of the load's pose and motion; ``model_dump`` feeds ``HandoffRecord.state``."""

    model_config = ConfigDict(extra="forbid")

    com_x: float
    com_y: float
    angle: float
    vx: float
    vy: float
    omega: float
    in_contact: bool


def _settle(space: pymunk.Space, body: pymunk.Body, config: StepConfig) -> None:
    sub = config.dt / config.substeps
    for _ in range(config.settle_steps):
        for _ in range(config.substeps):
            space.step(sub)
    if config.quasi_static:
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0


def apply_macro_action(
    space: pymunk.Space, body: pymunk.Body, action: MacroAction, config: StepConfig
) -> None:
    """Apply one macro action (world-frame impulse / angular kick at the COM), then settle."""
    j = config.linear_impulse
    com = body.local_to_world(body.center_of_gravity)
    if action == "E":
        body.apply_impulse_at_world_point((j, 0.0), com)
    elif action == "W":
        body.apply_impulse_at_world_point((-j, 0.0), com)
    elif action == "N":
        body.apply_impulse_at_world_point((0.0, j), com)
    elif action == "S":
        body.apply_impulse_at_world_point((0.0, -j), com)
    elif action == "ROT+":
        body.angular_velocity += config.angular_impulse / body.moment
    elif action == "ROT-":
        body.angular_velocity -= config.angular_impulse / body.moment
    # WAIT: no impulse, just settle.
    # Infinite moment for the duration of a non-rotate action: contact cannot spin the load at all,
    # rather than spinning it and having the angle written back afterwards (DSE-059).
    hold = config.hold_orientation and action not in ("ROT+", "ROT-")
    moment = body.moment
    if hold:
        body.moment = float("inf")
    _settle(space, body, config)
    if hold:
        body.moment = moment


def apply_force_handles(
    space: pymunk.Space,
    body: pymunk.Body,
    force_a: tuple[float, float],
    force_b: tuple[float, float],
    config: StepConfig,
) -> None:
    """Two-grip interface: impulses at the two bar ends, then settle (selected behind a flag).

    Equal forces translate; opposed forces apply a couple (rotation). Impulses are body-frame.
    """
    body.apply_impulse_at_local_point(force_a, _GRIP_LEFT)
    body.apply_impulse_at_local_point(force_b, _GRIP_RIGHT)
    _settle(space, body, config)


def measure_rotation_step(config: StepConfig | None = None) -> float:
    """Degrees of rotation one ``ROT+`` realises in free space, for the active load.

    The guard against the DSE-059 failure: a constant that documented a load which no longer
    existed.
    Runs the real actuator in an empty space (no walls, so no contact truncation) and reports what
    it
    actually does, so the authored ``ROTATION_STEP_DEG`` can be asserted against the realised value
    rather than trusted. Free rotation is deterministic, so one sample is the whole distribution.
    """
    config = config or StepConfig()
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = DAMPING
    body = add_load(space, (0.0, 0.0), LOAD_MASS)
    before = body.angle
    apply_macro_action(space, body, "ROT+", config)
    return math.degrees(body.angle - before)


def read_state(space: pymunk.Space, body: pymunk.Body) -> BodyState:
    """Read COM, angle, velocities and a contact flag for the load body."""
    com = body.local_to_world(body.center_of_gravity)
    arbiters: list[pymunk.Arbiter] = []
    body.each_arbiter(arbiters.append)
    return BodyState(
        com_x=float(com.x),
        com_y=float(com.y),
        angle=float(body.angle),
        vx=float(body.velocity.x),
        vy=float(body.velocity.y),
        omega=float(body.angular_velocity),
        in_contact=len(arbiters) > 0,
    )


def detect_collision(state: BodyState) -> bool:
    """Whether the load is in contact with a wall this step."""
    # ponytail: contact flag is enough; add an impulse threshold to exclude soft grazes if needed.
    return state.in_contact


def detect_stuck(states: list[BodyState], *, move_eps: float = 0.02, window: int = 5) -> bool:
    """Whether the COM has ended up where it started over the last ``window`` post-action states.

    Position-based, not velocity-based: in the quasi-static regime velocity is zeroed after each
    settle, so a jam shows up as the COM failing to advance across turns rather than as low speed.

    **Net displacement, not span** (v5). The span form - ``(max-min) < eps`` over the window -
    detects only *immobility*, and immobility was the minority failure on the E3 attempt-1 dataset.
    It scored ``stuck=False`` for all eighteen handoffs of an episode that alternated ``N,S,N,S...``
    against a wall (the COM genuinely moves a full unit each step, and returns), and for an episode
    that pushed ``E`` thirty-three times into a wall it could not pass (contact jitter exceeds
    ``move_eps`` per step). Both are the same thing operationally - a trajectory going nowhere - and
    the field that exists to name it saw neither. Comparing the window's endpoints catches
    immobility, any even-period limit cycle, and a jitter-jammed wall press alike.

    ``window`` is five states (four actions) so a period-2 cycle closes twice inside it; an odd
    window cannot distinguish ``N,S,N`` from one net step north.

    Diagnostic only: no gate reads ``stuck`` and nothing terminates on it (``graph.py`` exits on
    success or budget alone), so this changes what a run *records about itself*, never what it does.
    """
    if len(states) < window:
        return False
    first, last = states[-window], states[-1]
    return abs(last.com_x - first.com_x) + abs(last.com_y - first.com_y) < move_eps
