"""The arena: a top-down, damped three-chamber box joined by two slit gaps, with a goal region.

Chambers run left to right along +x; two internal vertical walls each carry a slit (a gap in y)
centred at ``geometry.slit_y``. The load starts in chamber one and the goal sits in chamber three.
``gravity=(0, 0)`` and ``damping<1`` make the regime quasi-static (the load does not coast).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import pymunk
from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ConfigError
from preceptx.data.schema import Difficulty
from preceptx.sim.load import add_t_load

# Quasi-static regime: strong damping so the load settles rather than coasts (roadmap §2.1).
DAMPING = 0.2
LOAD_MASS = 1.0
GOAL_RADIUS = 0.8
WALL_FRICTION = 0.6

# Difficulty maps to slit width. Threading geometry (see sim/feasibility.py + the experiment log):
# the T is a bar (T_BAR=1.4) perpendicular to a stem (T_STEM=1.0); whichever member is aligned with
# travel crosses a thin-wall gap at its 0.3 thickness while the OTHER presents its full length, so
# the TIGHTEST threadable slit is the shorter member = the stem = 1.0 (rotation cannot beat this),
# and a slit >= the full y-extent (T_THICK+T_STEM=1.3) clears head-on with no maneuver. The ladder
# grades across those regimes: easy 1.8 (head-on, trivial), medium 1.2 and hard 1.1 (both need a
# threading maneuver - decoupling the bar and stem wall-crossings - tightening as it hardens).
# The original 1.0/0.7 for medium/hard put medium at zero clearance and hard below the 1.0 threshold
# (geometrically impossible); corrected pre-freeze after the feasibility search caught it (P1-4).
_DIFFICULTY_SLITS: dict[Difficulty, float] = {"easy": 1.8, "medium": 1.2, "hard": 1.1}


def slit_widths() -> dict[Difficulty, float]:
    """A copy of the difficulty -> slit-width map, for ``sim/fingerprint.py``.

    Copied rather than the live dict: the fingerprint must observe the map, never mutate it.
    """
    return dict(_DIFFICULTY_SLITS)


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


class ScenarioJitter(BaseModel):
    """Seeded start-pose jitter region (P0-2): what makes the seed axis a true replication axis.

    Without it, greedy decoding + a fixed scenario + deterministic physics make same-cell episodes
    at different seeds nominally identical trajectories (pseudo-replication). The default region
    keeps the body origin >= 1.2 world units from every chamber-one wall - the T's farthest vertex
    sits ~0.955 from the origin, so any angle is collision-free by construction; ``make_scenario``
    still rejection-checks as a belt-and-braces guard. Geometry (slit widths, arena, goal) stays
    fixed: jittering the pose varies the problem instance without confounding difficulty. A
    zero-width range (e.g. ``x_range=(2.0, 2.0)``) recovers a fixed value for that axis.
    """

    model_config = ConfigDict(extra="forbid")

    x_range: tuple[float, float] = (1.2, 2.8)
    y_range: tuple[float, float] = (1.5, 4.5)
    theta_range: tuple[float, float] = (-math.pi / 2.0, math.pi / 2.0)


_MAX_JITTER_ATTEMPTS = 100  # rejection-sampling cap; unreachable with the default safe region


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


def make_scenario(
    difficulty: Difficulty,
    *,
    rng: Generator | None = None,
    jitter: ScenarioJitter | None = None,
) -> Scenario:
    """Build a ready-to-run scenario (space + load in chamber one + goal in chamber three).

    With ``rng`` the start pose is jittered inside ``jitter`` (defaulted) - the P0-2 replication
    fix - deterministically per rng stream, with a collision-free rejection check that fails loud.
    Without ``rng`` the legacy fixed pose is used (deterministic scripted tests rely on it). The
    realised pose needs no extra record field: it is ``pre_state`` of step 0.
    """
    geometry = ArenaGeometry()
    space = build_arena(_DIFFICULTY_SLITS[difficulty], geometry)
    goal = Goal(center_x=2.5 * geometry.chamber_w, center_y=geometry.slit_y, radius=GOAL_RADIUS)
    if rng is None:
        load = add_t_load(space, (geometry.chamber_w / 2.0, geometry.slit_y), LOAD_MASS)
        return Scenario(space=space, load=load, goal=goal)
    jit = jitter or ScenarioJitter()
    for _ in range(_MAX_JITTER_ATTEMPTS):
        pos = (rng.uniform(*jit.x_range), rng.uniform(*jit.y_range))
        load = add_t_load(space, pos, LOAD_MASS)
        load.angle = rng.uniform(*jit.theta_range)
        space.reindex_shapes_for_body(load)
        if not _overlaps_other(space, load):
            return Scenario(space=space, load=load, goal=goal)
        space.remove(load, *load.shapes)
    raise ConfigError(
        f"could not place a collision-free jittered load in {_MAX_JITTER_ATTEMPTS} attempts; "
        f"the jitter region {jit.model_dump()} leaves too little wall clearance"
    )


def _overlaps_other(space: pymunk.Space, load: pymunk.Body) -> bool:
    """Whether any load shape overlaps a shape not belonging to the load (i.e. a wall)."""
    return any(
        info.shape is not None and info.shape.body is not load
        for shape in load.shapes
        for info in space.shape_query(shape)
    )


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
