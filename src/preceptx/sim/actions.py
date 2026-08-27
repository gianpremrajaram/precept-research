"""Macro-action interface, the settle-step, state read-back, and stuck/collision detection.

Agents act through a discrete macro-action (the primary design); a two-grip force-handle interface
sits behind a flag. Each macro action applies one impulse and then settles the space under damping.
In the default quasi-static regime residual velocity is zeroed after settling, so the load is nudged
and comes to rest each turn (matching the damped top-down task) and inverse actions cancel cleanly.
"""

from __future__ import annotations

from typing import Literal

import pymunk
from pydantic import BaseModel, ConfigDict, Field

from preceptx.sim.load import BAR_LEN

MacroAction = Literal["N", "S", "E", "W", "ROT+", "ROT-", "WAIT"]

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
    # angle changes only when the pair deliberately rotates it. Modelled by restoring the angle
    # after every non-rotate action, in the same spirit as ``quasi_static`` zeroing velocity.
    #
    # This is not cosmetic - it closes the third degeneracy (DSE-058). Macro impulses are applied at
    # the COM and carry no torque, but contact at the channel mouth rotated the load anyway, so a
    # translation-only ACTION sequence produced a large STATE rotation and threaded the channel with
    # no rotate command ever issued. Measured at up to 114 deg of contact rotation, it leaked on
    # ~27% of jittered starts (22/30 seeds passed at aperture 0.48) and did so CHAOTICALLY in both
    # aperture and start pose, so no width could tune it away. With orientation held, every aperture
    # from 0.45 to 1.10 certifies 30/30 - the manipulation is restored by construction rather than
    # by a tuned constant, and no friction value was changed to get it.
    hold_orientation: bool = True
    linear_impulse: float = Field(default=3.0, gt=0)
    # Sized for controllable rotation (~34 deg per action) so an agent can aim the T for threading.
    # The old 2.0 spun it ~135 deg/action against the small T moment (~0.29), leaving only 45-deg-
    # multiple orientations reachable - too coarse for the medium/hard threading maneuver (P1-4).
    angular_impulse: float = Field(default=0.5, gt=0)
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
    held = body.angle if config.hold_orientation and action not in ("ROT+", "ROT-") else None
    _settle(space, body, config)
    if held is not None:
        body.angle = held
        space.reindex_shapes_for_body(body)


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
