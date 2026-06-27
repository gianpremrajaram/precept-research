"""The arena: a top-down, damped three-chamber box joined by two slit gaps, with a goal region.

Chambers run left to right along +x; two internal vertical walls each carry a slit (a gap in y)
centred at ``geometry.slit_y``. The load starts in chamber one and the goal sits in chamber three.
``gravity=(0, 0)`` and ``damping<1`` make the regime quasi-static (the load does not coast).
"""

from __future__ import annotations

from typing import NamedTuple

import pymunk
from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import Difficulty
from preceptx.sim.load import add_t_load

# Quasi-static regime: strong damping so the load settles rather than coasts (roadmap §2.1).
DAMPING = 0.2
LOAD_MASS = 1.0
GOAL_RADIUS = 0.8
WALL_FRICTION = 0.6

# Difficulty maps to slit width. The load's y-extent is T_THICK + T_STEM = 1.3, so "hard" jams a
# head-on push (must rotate) while "easy" clears it; documented here as the difficulty knob.
_DIFFICULTY_SLITS: dict[Difficulty, float] = {"easy": 1.8, "medium": 1.0, "hard": 0.7}


class ArenaGeometry(BaseModel):
    """Static arena dimensions (world units). Three chambers, two internal walls."""

    model_config = ConfigDict(extra="forbid")

    chamber_w: float = Field(default=4.0, gt=0)
    chamber_h: float = Field(default=6.0, gt=0)
    wall_radius: float = Field(default=0.05, gt=0)
    slit_y: float = Field(default=3.0, gt=0)


class Goal(BaseModel):
    """Circular goal region in chamber three."""

    model_config = ConfigDict(extra="forbid")

    center_x: float
    center_y: float
    radius: float = Field(gt=0)


class Scenario(NamedTuple):
    """A built arena bundled with its load body and goal region."""

    space: pymunk.Space
    load: pymunk.Body
    goal: Goal


def _wall(space: pymunk.Space, a: tuple[float, float], b: tuple[float, float], r: float) -> None:
    seg = pymunk.Segment(space.static_body, a, b, r)
    seg.friction = WALL_FRICTION
    space.add(seg)


def build_arena(slit_width: float, geometry: ArenaGeometry) -> pymunk.Space:
    """Build the three-chamber arena with two slits of ``slit_width`` in the internal walls."""
    cw, ch, r, sy = geometry.chamber_w, geometry.chamber_h, geometry.wall_radius, geometry.slit_y
    width = 3.0 * cw
    half = slit_width / 2.0

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = DAMPING

    # Outer boundary.
    _wall(space, (0.0, 0.0), (width, 0.0), r)
    _wall(space, (0.0, ch), (width, ch), r)
    _wall(space, (0.0, 0.0), (0.0, ch), r)
    _wall(space, (width, 0.0), (width, ch), r)

    # Two internal walls, each split into a lower and upper segment around the slit gap.
    for x in (cw, 2.0 * cw):
        _wall(space, (x, 0.0), (x, sy - half), r)
        _wall(space, (x, sy + half), (x, ch), r)
    return space


def make_scenario(difficulty: Difficulty) -> Scenario:
    """Build a ready-to-run scenario (space + load in chamber one + goal in chamber three)."""
    geometry = ArenaGeometry()
    space = build_arena(_DIFFICULTY_SLITS[difficulty], geometry)
    load = add_t_load(space, (geometry.chamber_w / 2.0, geometry.slit_y), LOAD_MASS)
    goal = Goal(center_x=2.5 * geometry.chamber_w, center_y=geometry.slit_y, radius=GOAL_RADIUS)
    return Scenario(space=space, load=load, goal=goal)


def slit_width_for(difficulty: Difficulty) -> float:
    """The slit width for a difficulty (the load's y-extent is 1.3); the graph needs it to build the
    ``SceneState`` for serialisation, which ``make_scenario`` does not return."""
    return _DIFFICULTY_SLITS[difficulty]


def chamber_of(com_x: float, geometry: ArenaGeometry) -> int:
    """Which chamber (1, 2 or 3, left to right) an x-coordinate falls in.

    Boundaries belong to the chamber on their right (``x = chamber_w`` is chamber 2), keeping the
    geodesic continuous across a slit centre. Shared by the serialiser and the outcome labeller.
    """
    if com_x < geometry.chamber_w:
        return 1
    if com_x < 2.0 * geometry.chamber_w:
        return 2
    return 3
