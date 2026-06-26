"""State serialisers for the prompt: numeric tuples, an ASCII occupancy grid, and templated NL.

How the physics state is written into the prompt is an experimental factor (the RoCo lesson that
prompt formatting can masquerade as spatial reasoning), so three forms are selectable by config.
The three are *isomorphic in information* - each exposes the same load pose and goal, differing
only in surface form - which keeps the serialisation factor a clean A/B over representation, not
over information content. Every serialiser is pure, deterministic and total; ``deserialise_check``
guards the numeric and grid forms against dropping the load COM (the grid certifies angle via the
occupancy-correctness tests, not by recovering it from a coarse ASCII raster).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import Serialisation
from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal, chamber_of
from preceptx.sim.load import COG_Y, point_in_t_local


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
    """Resolution of the ASCII occupancy grid. ``cell=0.25`` keeps the T's thin members (~0.3 wide)
    about one cell wide so the rotate-to-clear-the-slit affordance is visible; ``0.5`` aliases them
    away. The pilot may retune this before the Phase-2 serialisation freeze."""

    model_config = ConfigDict(extra="forbid")

    cell: float = Field(default=0.25, gt=0)


_GRID = GridConfig()


def serialise(scene: SceneState, mode: Serialisation) -> str:
    """Render ``scene`` to its prompt form for ``mode`` (numeric tuples / ASCII grid / NL)."""
    if mode == "numeric":
        return _numeric(scene)
    if mode == "grid":
        return _grid(scene, _GRID)
    return _nl(scene)


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


def _numeric(scene: SceneState) -> str:
    s, g = scene.load, scene.goal
    return (
        f"load=({s.com_x:.4f}, {s.com_y:.4f}, {s.angle:.4f})  # (com_x, com_y, angle)\n"
        f"vel=({s.vx:.4f}, {s.vy:.4f}, {s.omega:.4f})  # (vx, vy, omega)\n"
        f"contact={s.in_contact}\n"
        f"goal=({g.center_x:.4f}, {g.center_y:.4f}, {g.radius:.4f})  # (center_x, center_y, radius)"
    )


def _grid(scene: SceneState, cfg: GridConfig) -> str:
    cell, geo, goal, s = cfg.cell, scene.geometry, scene.goal, scene.load
    width = 3.0 * geo.chamber_w
    n_cols, n_rows = round(width / cell), round(geo.chamber_h / cell)
    half = scene.slit_width / 2.0
    ca, sa = math.cos(s.angle), math.sin(s.angle)
    # Body origin from the COM: com = origin + R(angle)·(0, COG_Y), so origin = com - R(angle)·cog.
    ox, oy = s.com_x + sa * COG_Y, s.com_y - ca * COG_Y

    rows: list[str] = []
    for r in range(n_rows - 1, -1, -1):  # +y up: print the top row first
        cy = (r + 0.5) * cell
        line: list[str] = []
        for c in range(n_cols):
            cx = (c + 0.5) * cell
            dx, dy = cx - ox, cy - oy
            if point_in_t_local(ca * dx + sa * dy, -sa * dx + ca * dy):  # world -> load-local
                line.append("T")
            elif math.hypot(cx - goal.center_x, cy - goal.center_y) <= goal.radius:
                line.append("G")
            elif _is_wall(cx, cy, geo, half, cell):
                line.append("#")
            else:
                line.append(".")
        rows.append("".join(line))
    return "\n".join(rows)


def _is_wall(cx: float, cy: float, geo: ArenaGeometry, half: float, cell: float) -> bool:
    h = cell / 2.0
    width = 3.0 * geo.chamber_w
    if cx <= h or cx >= width - h or cy <= h or cy >= geo.chamber_h - h:
        return True  # outer boundary ring
    # ponytail: internal walls can render ~2 cells thick at a cell boundary; faithful enough.
    return any(
        abs(cx - x) <= h and not (geo.slit_y - half <= cy <= geo.slit_y + half)
        for x in (geo.chamber_w, 2.0 * geo.chamber_w)
    )


def _nl(scene: SceneState) -> str:
    s, geo, goal = scene.load, scene.geometry, scene.goal
    chamber, goal_chamber = chamber_of(s.com_x, geo), chamber_of(goal.center_x, geo)
    n_slits = abs(goal_chamber - chamber)
    direction = "east" if goal.center_x >= s.com_x else "west"
    if chamber < goal_chamber:  # there is a slit ahead to thread
        sx, sy = geo.chamber_w * chamber, geo.slit_y
        slit_clause = (
            f" the nearest slit centre is at ({sx:.2f}, {sy:.2f}), "
            f"{math.hypot(sx - s.com_x, sy - s.com_y):.2f} away;"
        )
    else:
        slit_clause = ""
    return (
        f"The T-load is in chamber {chamber} at ({s.com_x:.2f}, {s.com_y:.2f}), "
        f"angle {s.angle:.2f} rad ({_orientation(s.angle)}). The goal is at "
        f"({goal.center_x:.2f}, {goal.center_y:.2f}), radius {goal.radius:.2f}, lying {direction} "
        f"beyond {n_slits} slit(s);{slit_clause} the load is "
        f"{'touching' if s.in_contact else 'clear of'} a wall."
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
    rows = _grid(scene, cfg).splitlines()
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
