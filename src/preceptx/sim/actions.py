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

from preceptx.sim.load import HALF_H, T_BAR, T_THICK

MacroAction = Literal["N", "S", "E", "W", "ROT+", "ROT-", "WAIT"]

# Local-frame grip points for the force-handle interface: the two ends of the bar.
_GRIP_LEFT = (-T_BAR / 2.0, HALF_H - T_THICK / 2.0)
_GRIP_RIGHT = (T_BAR / 2.0, HALF_H - T_THICK / 2.0)


class StepConfig(BaseModel):
    """Stepping and impulse parameters. Defaults chosen for stability under thin walls and damping.

    ``dt`` is split into ``substeps`` substeps per settle step to avoid tunnelling; ``settle_steps``
    settle steps run after each impulse. ``quasi_static`` zeroes residual velocity once settled.
    """

    model_config = ConfigDict(extra="forbid")

    dt: float = Field(default=1.0 / 60.0, gt=0)
    substeps: int = Field(default=4, ge=1)
    settle_steps: int = Field(default=30, ge=1)
    linear_impulse: float = Field(default=3.0, gt=0)
    angular_impulse: float = Field(default=2.0, gt=0)
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
    _settle(space, body, config)


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


def detect_stuck(states: list[BodyState], *, move_eps: float = 0.02, window: int = 3) -> bool:
    """Whether the COM has barely moved over the last ``window`` post-action states (jammed).

    Position-based, not velocity-based: in the quasi-static regime velocity is zeroed after each
    settle, so a jam shows up as the COM failing to advance across turns rather than as low speed.
    """
    if len(states) < window:
        return False
    recent = states[-window:]
    xs = [s.com_x for s in recent]
    ys = [s.com_y for s in recent]
    return (max(xs) - min(xs)) + (max(ys) - min(ys)) < move_eps
