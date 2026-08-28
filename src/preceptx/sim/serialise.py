"""State serialisers for the prompt: numeric tuples, an ASCII occupancy grid, and templated NL.

How the physics state is written into the prompt is an experimental factor (the RoCo lesson that
prompt formatting can masquerade as spatial reasoning), so three forms are selectable by config.
The three are *isomorphic in information* - each exposes the same load pose, goal AND wall/slit
geometry, differing only in surface form - which keeps the serialisation factor a clean A/B over
representation, not over information content. That isomorphism is a claim about the code, not a
hope: the numeric form carried no wall or slit geometry until v3, while the grid drew it and the NL
form named it, so the serialisation axis was silently confounding representation with information
content (found in E1 - see docs/experiment_design_log.md, 2026-08-24).

Every serialiser is pure, deterministic and total; ``deserialise_check`` guards the numeric and
grid forms against dropping the load COM (the grid certifies angle via the occupancy-correctness
tests, not by recovering it from a coarse ASCII raster).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import Serialisation
from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal, chamber_of
from preceptx.sim.load import (
    LOAD_COG_Y,
    LOAD_EXTENT_X,
    LOAD_EXTENT_Y,
    extent_y,
    point_in_load_local,
)


class SceneState(BaseModel):
    """A frozen, serialisable snapshot: load pose, arena geometry, goal and the active slit width.

    Distinct from the live ``Scenario`` (pymunk handles): this carries plain floats, so it feeds the
    prompt, the featuriser (DSE-013) and the handoff record, and is reconstructable from them. The
    grid serialiser needs ``slit_width`` to draw the correct gap for the active difficulty.
    """

    model_config = ConfigDict(extra="forbid")

    load: BodyState
    geometry: ArenaGeometry
    goal: Goal
    slit_width: float = Field(gt=0)


class GridConfig(BaseModel):
    """Resolution of the ASCII occupancy grid. ``cell=0.25`` keeps the bar's 0.3 thickness
    about one cell wide so the rotate-to-clear-the-slit affordance is visible; ``0.5`` aliases it
    away. The pilot may retune this before the Phase-2 serialisation freeze."""

    model_config = ConfigDict(extra="forbid")

    cell: float = Field(default=0.25, gt=0)


_GRID = GridConfig()

# Constant symbol key prepended to every grid (P1-5): without it B acts on an unexplained ASCII
# matrix and the serialisation A/B measures legend absence, not representation. Constant text across
# cells preserves the information-isomorphism argument. NOTE: the line contains a literal "T", so
# grid consumers that locate load rows (channel C3 windowing, the centroid check here) must skip it.
# ponytail: the glyph stays "T" although the load is a convex bar (DSE-058). The legend DEFINES it
# as "T=load", so it is a symbol, not a shape claim - unlike the prose forms, which did assert a T
# and were corrected in v6. Renaming it would touch the C3 row-finder, the centroid check and their
# tests to buy nothing a reader of the legend does not already have.
GRID_LEGEND = "legend: T=load G=goal #=wall .=free | top row = north (+y)"


def serialise(scene: SceneState, mode: Serialisation) -> str:
    """Render ``scene`` to its prompt form for ``mode`` (numeric tuples / ASCII grid / NL)."""
    if mode == "numeric":
        return _numeric(scene)
    if mode == "grid":
        return _grid(scene, _GRID)
    return _nl(scene)


# How many past actions the history line shows (v5). Four is the smallest window that displays a
# period-2 limit cycle twice - the dominant E3 attempt-1 failure (N,S,N,S... and ROT+,ROT-,...),
# where a memoryless greedy policy alternates forever because each state maps back to the other.
# A longer window costs prompt tokens and buys nothing: the pathology is visible in four.
HISTORY_WINDOW = 4

# The line's key. Exported because G3's truth set must exclude it (pilot.py): the history is what
# the sender was shown, but it is not *geometry*, and G3 scores messages against geometry. A literal
# "recent=" in both modules would be a coupling nothing announces when the key changes.
HISTORY_PREFIX = "recent="


def history_line(recent: Sequence[tuple[str, float]]) -> str:
    """Render the last few (action, geodesic gain) pairs as one prompt line (v5).

    Serialisation-independent by construction: the same line is appended to all three forms, so the
    serialisation axis stays a contrast over *representation* and this adds the same information to
    every arm of it. It is deliberately a statement of fact, not advice - it reports what was done
    and what it gained, and leaves "so try something else" as the agent's inference. A directive
    here would make the retune a behavioural instruction rather than an observability fix, and the
    two are not separable after the fact.
    """
    if not recent:
        return f"{HISTORY_PREFIX}()  # no actions taken yet"
    # `or 0.0` normalises -0.0, which formats as a confusing "-0.00" in a prompt a model reads.
    pairs = ", ".join(f"({action}, {gain or 0.0:+.2f})" for action, gain in recent)
    net = sum(gain for _, gain in recent)
    return (
        f"{HISTORY_PREFIX}({pairs})"
        "  # (action, distance gained toward the goal), oldest first;"
        f" net {net:+.2f} over the last {len(recent)}"
    )


def deserialise_check(scene: SceneState, mode: Serialisation) -> bool:
    """Recover the load COM (and, for numeric, angle) from the serialised string and confirm it
    matches the source within tolerance - a guard against the representation dropping state.

    ``numeric`` round-trips COM and angle to its print precision; ``grid`` recovers the COM as the
    centroid of its load cells (within ~one cell); angle is certified separately by the occupancy
    tests. ``nl`` is one-way prose and is not checkable, so it fails loud.
    """
    if mode == "numeric":
        com_x, com_y, angle = _parse_numeric_load(_numeric(scene))
        return (
            math.isclose(com_x, scene.load.com_x, abs_tol=1e-2)
            and math.isclose(com_y, scene.load.com_y, abs_tol=1e-2)
            and math.isclose(angle, scene.load.angle, abs_tol=1e-2)
        )
    if mode == "grid":
        com = _grid_load_centroid(scene, _GRID)
        if com is None:  # load fully off-grid: nothing to certify
            return False
        tol = 1.5 * _GRID.cell
        return abs(com[0] - scene.load.com_x) <= tol and abs(com[1] - scene.load.com_y) <= tol
    raise ValueError(f"deserialise_check supports numeric/grid only, not {mode!r}")


# The pose-dependent clearance line (v8), appended to ALL THREE forms. `load_size` (v4) named the
# object's constants because naming the gap without the object was underdetermined; DSE-058 made
# each wall a channel, so the pass-relevant quantity became the load's PROJECTION, which is state,
# not a constant. Run 232980 is the evidence: 99.9% of messages quoted the angle, 6.6% attempted
# the projection, and 97.6% of E actions were pushes at poses that could not fit. It states the
# span, not a verdict - the slit comparison, the rotate-then-translate ordering, the y-alignment
# and the two-wall repetition all remain the agent's inference. Isomorphism is why it goes in all
# three: withholding it from one form would make that form measure trigonometry, not representation.
_RASTER_GLYPHS = frozenset("TG#.")


def split_grid(grid: str) -> tuple[list[str], list[str]]:
    """Split a serialised grid into its header lines and its raster rows.

    The header was one line (the legend) until v8 added the clearance line, and consumers that
    counted lines - the centroid check here, the C3 row window in agents/channel.py - silently
    mis-indexed every row when it grew. Recognising a raster row by its alphabet costs the same and
    does not need revisiting the next time the header does.
    """
    lines = grid.splitlines()
    cut = next(
        (i for i, line in enumerate(lines) if line and set(line) <= _RASTER_GLYPHS), len(lines)
    )
    return lines[:cut], lines[cut:]


def clearance_line(scene: SceneState) -> str:
    """The load's vertical span at its current angle, for the prompt (v8)."""
    return (
        f"load_extent_y={extent_y(scene.load.angle):.4f}"
        "  # the load's vertical span AT THIS ANGLE; a slit narrower than this cannot admit it"
    )


def _numeric(scene: SceneState) -> str:
    # No velocity line (RD-7): quasi-static settling zeroes velocity before every read, so it was
    # constant dead weight in the prompt. Landed with the grid legend as one serialisation bump
    # (PROMPT_VERSION v2).
    # The wall and slit lines are v3: without them this form named no obstacle at all, so A could
    # only emit generic advice and the numeric-vs-grid contrast was not information-isomorphic.
    # The load's y-extent is 1.3, so the slit interval is what decides whether it threads or jams.
    # `load_size` is v4: naming the gap without naming the object left "aligned with the slit"
    # underdetermined, and the pilot watched a model call com_y=2.0074 aligned with a (2.1, 3.9)
    # gap and push into the wall. The dimensions are constants of the load, not a derived pass
    # band - deriving the threading band from them is still the agent's inference to make.
    # `wall_depth` is v6: DSE-058 made each wall a CHANNEL with x-extent, and this form named one x
    # per wall, so the state understated the obstacle that makes orientation binding. C3 strips it
    # for free - the restrictor whitelists `load=`/`contact=` rather than blacklisting layout keys.
    s, g, geo = scene.load, scene.goal, scene.geometry
    half = scene.slit_width / 2.0
    return (
        f"load=({s.com_x:.4f}, {s.com_y:.4f}, {s.angle:.4f})  # (com_x, com_y, angle)\n"
        f"contact={s.in_contact}\n"
        f"goal=({g.center_x:.4f}, {g.center_y:.4f}, {g.radius:.4f})"
        "  # (center_x, center_y, radius)\n"
        f"walls_x=({geo.chamber_w:.4f}, {2.0 * geo.chamber_w:.4f})  # centre x of each wall\n"
        f"{_wall_depth_line(geo)}"
        f"slit_y=({geo.slit_y - half:.4f}, {geo.slit_y + half:.4f})  "
        f"# the only gap in each wall (width {scene.slit_width:.4f})\n"
        f"load_size=({LOAD_EXTENT_X:.4f}, {LOAD_EXTENT_Y:.4f})  "
        "# (length, thickness) - the WHOLE load must clear the gap, not its centre\n"
        f"{clearance_line(scene)}"
    )


def _grid(scene: SceneState, cfg: GridConfig) -> str:
    cell, geo, goal, s = cfg.cell, scene.geometry, scene.goal, scene.load
    width = 3.0 * geo.chamber_w
    n_cols, n_rows = round(width / cell), round(geo.chamber_h / cell)
    half = scene.slit_width / 2.0
    ca, sa = math.cos(s.angle), math.sin(s.angle)
    # Body origin from the COM: com = origin + R(angle)·(0, COG_Y), so origin = com - R(angle)·cog.
    ox, oy = s.com_x + sa * LOAD_COG_Y, s.com_y - ca * LOAD_COG_Y

    rows: list[str] = []
    for r in range(n_rows - 1, -1, -1):  # +y up: print the top row first
        cy = (r + 0.5) * cell
        line: list[str] = []
        for c in range(n_cols):
            cx = (c + 0.5) * cell
            dx, dy = cx - ox, cy - oy
            if point_in_load_local(ca * dx + sa * dy, -sa * dx + ca * dy):  # world -> load-local
                line.append("T")
            elif math.hypot(cx - goal.center_x, cy - goal.center_y) <= goal.radius:
                line.append("G")
            elif _is_wall(cx, cy, geo, half, cell):
                line.append("#")
            else:
                line.append(".")
        rows.append("".join(line))
    return "\n".join([GRID_LEGEND, clearance_line(scene), *rows])


def _is_wall(cx: float, cy: float, geo: ArenaGeometry, half: float, cell: float) -> bool:
    """Occupancy of the static geometry at a cell centre.

    Fills the whole channel footprint, not just its two faces (v6). ``build_arena`` seals the strip
    between the faces above and below the aperture with the cap segments, so that region is an
    enclosed void the load can never reach: for a planner reading an occupancy raster it is
    impassable, and drawing free space there contradicted the physics. At ``wall_depth = 0`` this
    reduces to the legacy one-cell stripe, so the falsified T arena still renders as it did.
    """
    h = cell / 2.0
    width = 3.0 * geo.chamber_w
    if cx <= h or cx >= width - h or cy <= h or cy >= geo.chamber_h - h:
        return True  # outer boundary ring
    if geo.slit_y - half < cy < geo.slit_y + half:
        return False  # the aperture band is the passage through every internal wall
    return any(
        abs(cx - x) <= geo.wall_depth / 2.0 + h for x in (geo.chamber_w, 2.0 * geo.chamber_w)
    )


def _wall_depth_line(geo: ArenaGeometry) -> str:
    """The numeric form's channel-depth line; empty for a legacy thin-segment arena."""
    if geo.wall_depth <= 0.0:
        return ""
    return (
        f"wall_depth={geo.wall_depth:.4f}  # each wall is a solid channel spanning its centre "
        f"+/-{geo.wall_depth / 2.0:.4f} in x - the load must stay aligned across the whole depth\n"
    )


def _nl(scene: SceneState) -> str:
    s, geo, goal = scene.load, scene.geometry, scene.goal
    chamber, goal_chamber = chamber_of(s.com_x, geo), chamber_of(goal.center_x, geo)
    n_slits = abs(goal_chamber - chamber)
    direction = "east" if goal.center_x >= s.com_x else "west"
    if chamber < goal_chamber:  # there is a slit ahead to thread
        sx, sy = geo.chamber_w * chamber, geo.slit_y
        depth_clause = (
            f" each wall is a {geo.wall_depth:.2f}-deep channel, not a thin line, so the load must "
            "be aligned before it enters and stay aligned all the way through;"
            if geo.wall_depth > 0.0
            else ""
        )
        slit_clause = (
            f" the nearest slit centre is at ({sx:.2f}, {sy:.2f}), "
            f"{math.hypot(sx - s.com_x, sy - s.com_y):.2f} away;{depth_clause}"
        )
    else:
        slit_clause = ""
    return (
        f"The bar-shaped load is in chamber {chamber} at ({s.com_x:.2f}, {s.com_y:.2f}), "
        f"angle {s.angle:.2f} rad ({_orientation(s.angle)}). The goal is at "
        f"({goal.center_x:.2f}, {goal.center_y:.2f}), radius {goal.radius:.2f}, lying {direction} "
        f"beyond {n_slits} slit(s);{slit_clause} the load is "
        f"{'touching' if s.in_contact else 'clear of'} a wall. "
        f"At this angle the load spans {extent_y(s.angle):.4f} vertically, so a slit narrower "
        "than that cannot admit it."
    )


def _orientation(angle: float) -> str:
    a = abs(angle) % math.pi  # fold to [0, pi)
    if a < math.radians(15) or a > math.pi - math.radians(15):
        return "bar horizontal"
    if abs(a - math.pi / 2.0) < math.radians(15):
        return "bar vertical"
    return "tilted"


def _parse_numeric_load(text: str) -> tuple[float, float, float]:
    line = text.splitlines()[0]  # 'load=(x, y, a)  # ...'
    inner = line[line.index("(") + 1 : line.index(")")]
    x, y, a = (float(v) for v in inner.split(","))
    return x, y, a


def _grid_load_centroid(scene: SceneState, cfg: GridConfig) -> tuple[float, float] | None:
    _, rows = split_grid(_grid(scene, cfg))
    n_rows = len(rows)
    xs: list[float] = []
    ys: list[float] = []
    for i, row in enumerate(rows):
        r = n_rows - 1 - i  # the top row was printed first
        for c, char in enumerate(row):
            if char == "T":
                xs.append((c + 0.5) * cfg.cell)
                ys.append((r + 0.5) * cfg.cell)
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)
