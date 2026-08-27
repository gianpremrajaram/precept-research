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
from preceptx.sim.load import add_load

# Quasi-static regime: strong damping so the load settles rather than coasts (roadmap §2.1).
DAMPING = 0.2
LOAD_MASS = 1.0
GOAL_RADIUS = 0.8
WALL_FRICTION = 0.6

# Difficulty maps to channel aperture (DSE-058, the successor task). The load is a convex 1.4 x 0.3
# bar and each internal wall is a CHANNEL of depth `wall_depth`, not a thin segment. Fit through a
# channel is governed by the body's extent - a convex body has no staged-crossing escape - so an
# aperture below the bar length forces an explicit rotation. The rotation-required band is
# [BAR_THICK, BAR_LEN) = [0.3, 1.4); the effective aperture is `nominal - 2 * wall_radius`.
#
# The ladder's lower edge is physical: below 0.40 the effective aperture (nominal - 2*wall_radius
# = nominal - 0.1) falls under BAR_THICK and nothing fits at any angle. Its upper edge is the bar
# length: at 1.40 nominal the bar passes broadside and no rotation is required.
#
# 1.20/0.80/0.50 is certified 30/30 seeds per rung at CERTIFICATION_STEP_CONFIG (effective
# clearances 1.10/0.70/0.40 against a 0.30-thick, 1.40-long bar). Certification uses 30 seeds
# against the pilot's 10 deliberately: an earlier 0.48/0.45/0.42 candidate passed 10/10 on seeds
# 0-9 and leaked on 8 of seeds 10-29. Difficulty grades by ALIGNMENT PRECISION - narrower apertures
# admit a narrower band of passing orientations - and the certificate separates the rungs directly:
# easy needs 1 rotation, medium and hard need 2.
#
# That band exists only because `StepConfig.hold_orientation` holds the load's angle through
# non-rotate actions. Without it the bar aligns ITSELF against the channel mouth under contact
# torque and a translation-only path reappears (0.80 leaked in 7 of 10 seeds), which is what
# collapsed an earlier ladder to 0.48/0.45/0.42. Friction was tested and rejected as the lever -
# the leak survived wall friction 0.2, 0.6 and 1.5 - so no friction constant was tuned.
# `hold_orientation` is a load-bearing modelling assumption here, not an incidental default.
#
# If G1 fails on precision rather than on reasoning, the lever is BAR_THICK: the usable window
# scales with it, so a thicker bar buys absolute clearance without giving up rotation-necessity.
#
# The predecessor T ladder (1.8/1.2/1.1 against thin segment walls) was falsified: rotation was
# unnecessary at every rung. It is preserved in the design log, not here.
_DIFFICULTY_SLITS: dict[Difficulty, float] = {"easy": 1.20, "medium": 0.80, "hard": 0.50}


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
    # Depth of each internal wall along x. > 0 builds a CHANNEL (two faces plus a floor and ceiling)
    # and is what makes orientation binding; 0.0 builds the legacy thin segment, retained so the
    # falsification of the T task stays reproducible from source (DSE-058).
    wall_depth: float = Field(default=1.5, ge=0)


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
    keeps the body origin >= 1.2 world units from every chamber-one wall - the bar's farthest vertex
    sits ~0.716 from the origin, so any angle is collision-free by construction; ``make_scenario``
    still rejection-checks as a belt-and-braces guard. Geometry (apertures, arena, goal) stays
    fixed: jittering the pose varies the problem instance without confounding difficulty. A
    zero-width range (e.g. ``x_range=(2.0, 2.0)``) recovers a fixed value for that axis.

    **theta is concentrated near perpendicular (80-100 deg), not spread over +/-90 (DSE-058).** The
    bar's extent is symmetric about 90 deg, so this band starts every episode broadside to the
    channel - maximally misaligned, and outside the passing band at every rung, so an explicit
    rotation is required from every seed. It is also the region where passive self-alignment fails:
    a broadside bar meets the channel mouth flat-on and gets no aligning torque, whereas oblique
    starts (~45-75 deg) are funnelled into line by contact alone and need no rotate action. The
    narrow theta band is therefore load-bearing for task validity, not a convenience.

    **x_range stops at 2.4** so the bar's farthest vertex (2.4 + 0.7 = 3.1) clears the channel mouth
    at x = chamber_w - wall_depth/2 = 3.25; the old 2.8 would have started some poses inside it.
    """

    model_config = ConfigDict(extra="forbid")

    x_range: tuple[float, float] = (1.2, 2.4)
    y_range: tuple[float, float] = (1.5, 4.5)
    theta_range: tuple[float, float] = (math.radians(80.0), math.radians(100.0))


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

    # Two internal walls. With `wall_depth` > 0 each is a channel: two faces at x +/- depth/2 plus
    # the floor and ceiling joining them, so the load is constrained over an x-interval rather than
    # at a single threshold. That sustained constraint is what a thin segment cannot impose, and it
    # is why the aperture governs orientation here and did not before (DSE-058).
    d = geometry.wall_depth
    for x in (cw, 2.0 * cw):
        x0, x1 = x - d / 2.0, x + d / 2.0
        for face in (x0, x1) if d > 0.0 else (x,):
            _wall(space, (face, 0.0), (face, sy - half), r)
            _wall(space, (face, sy + half), (face, ch), r)
        if d > 0.0:
            _wall(space, (x0, sy - half), (x1, sy - half), r)
            _wall(space, (x0, sy + half), (x1, sy + half), r)
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
        load = add_load(space, (geometry.chamber_w / 2.0, geometry.slit_y), LOAD_MASS)
        # Canonical pose is BROADSIDE (DSE-058). At angle 0 the bar is already aligned with the
        # channel and the instance is trivial - E,E,E... with no rotation - which would derive a
        # step budget from a pose no jittered episode ever sees. pi/2 is the centre of the shipped
        # theta band, so the certificate and the budget describe the modal instance.
        load.angle = math.pi / 2.0
        space.reindex_shapes_for_body(load)
        return Scenario(space=space, load=load, goal=goal)
    jit = jitter or ScenarioJitter()
    for _ in range(_MAX_JITTER_ATTEMPTS):
        pos = (rng.uniform(*jit.x_range), rng.uniform(*jit.y_range))
        load = add_load(space, pos, LOAD_MASS)
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
    """The channel aperture for a difficulty; the graph needs it to build the ``SceneState`` for
    serialisation, which ``make_scenario`` does not return."""
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
