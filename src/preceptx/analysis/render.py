"""Episode renderer and transcript dumper (review section 7-2).

Two views of a recorded episode, both reconstructed from the persisted ``HandoffRecord``s alone (no
live sim): ``render_episode`` draws the arena, goal and T-load pose per step as a PNG grid, and
``render_transcript`` emits a markdown transcript pairing each state with its message and action.
These are what prompt iteration and G1 debugging need - eyes on messages next to states - and they
satisfy DSE-029's "a committed demonstration trace renders", which no other ticket builds. The
figure is guarded exactly like the other RQ figures: absent the ``viz`` extra it no-ops with a log
line; the markdown transcript (pure text) is always available.

Pose -> world geometry: a record's ``com_x/com_y`` is the body origin (``center_of_gravity`` is the
origin), so the T polygons are ``com + R(angle) . local_vert`` with no COG offset - the same
``load.t_shape_verts`` the physics body is built from, so the drawing cannot drift from the sim.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from preceptx.data.schema import HandoffRecord
from preceptx.sim.arena import GOAL_RADIUS, ArenaGeometry, Goal, slit_width_for
from preceptx.sim.load import t_shape_verts

logger = logging.getLogger(__name__)

_RC = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

Vert = tuple[float, float]


def _pyplot() -> Any | None:
    """Return ``matplotlib.pyplot`` (Agg backend) or ``None`` when the viz extra is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib absent (install the 'viz' extra) - skipping episode render")
        return None
    return plt


def _goal(geo: ArenaGeometry) -> Goal:
    """The fixed goal region make_scenario builds (chamber-three centre)."""
    return Goal(center_x=2.5 * geo.chamber_w, center_y=geo.slit_y, radius=GOAL_RADIUS)


def _t_world(com_x: float, com_y: float, angle: float) -> list[list[Vert]]:
    """The two T polygons (bar, stem) in world coordinates at the given pose."""
    ca, sa = math.cos(angle), math.sin(angle)
    return [
        [(com_x + ca * x - sa * y, com_y + sa * x + ca * y) for x, y in poly]
        for poly in t_shape_verts()
    ]


def _clip(text: str, n: int) -> str:
    """Collapse whitespace and truncate for a compact annotation/cell."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= n else collapsed[: n - 1] + "…"


def _draw_frame(
    ax: Any, record: HandoffRecord, geo: ArenaGeometry, goal: Goal, slit: float
) -> None:
    from matplotlib.patches import Circle, Polygon, Rectangle

    w, h = geo.chamber_w, geo.chamber_h
    half = slit / 2.0
    ax.add_patch(Rectangle((0.0, 0.0), 3.0 * w, h, fill=False, edgecolor="0.6", lw=1.0))
    for wx in (w, 2.0 * w):  # internal walls, each split around the slit gap at slit_y
        ax.plot([wx, wx], [0.0, geo.slit_y - half], color="0.3", lw=2.0)
        ax.plot([wx, wx], [geo.slit_y + half, h], color="0.3", lw=2.0)
    ax.add_patch(Circle((goal.center_x, goal.center_y), goal.radius, color="tab:green", alpha=0.25))
    st = record.post_state
    for poly in _t_world(float(st["com_x"]), float(st["com_y"]), float(st["angle"])):
        ax.add_patch(Polygon(poly, closed=True, facecolor="tab:blue", edgecolor="black", lw=0.8))
    ax.set_xlim(-0.3, 3.0 * w + 0.3)
    ax.set_ylim(-0.3, h + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    flag = "GOAL" if record.success else ("stuck" if record.stuck else "")
    ax.set_title(f"step {record.step} - {record.action['action']} {flag}".strip(), fontsize=8)
    ax.set_xlabel(_clip(record.message_delivered, 42), fontsize=6)


def render_episode(
    records: list[HandoffRecord], *, path: Path, ncols: int = 4, max_frames: int | None = None
) -> Path | None:
    """Draw the episode as a grid of per-step arena frames to ``path``. No-op without the viz extra.

    Each frame shows the arena, the goal and the T at that handoff's post-state, titled with the
    step and action and captioned with the delivered message (top row = north / +y).
    """
    if not records:
        raise ValueError("render_episode needs at least one record")
    plt = _pyplot()
    if plt is None:
        return None
    geo = ArenaGeometry()
    goal = _goal(geo)
    slit = slit_width_for(records[0].difficulty)
    frames = records if max_frames is None else records[:max_frames]
    n = len(frames)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
        flat = list(axes.flat)
        for idx, ax in enumerate(flat):
            if idx < n:
                _draw_frame(ax, frames[idx], geo, goal, slit)
            else:
                ax.axis("off")
        r0 = frames[0]
        fig.suptitle(
            f"{r0.episode_id}  ({r0.condition} / {r0.serialisation} / {r0.difficulty})", fontsize=10
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def _md_cell(text: str, n: int = 60) -> str:
    """Escape pipes/newlines and truncate a message for a markdown table cell."""
    collapsed = " ".join(text.split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= n else collapsed[: n - 1] + "…"


def render_transcript(records: list[HandoffRecord]) -> str:
    """A markdown transcript of the episode: per-step action, message (raw -> delivered), progress.

    Always available (pure text). Shows the raw and delivered message side by side so channel
    degradation (C1 truncation, C4 dropout, C3 restriction) is visible next to the chosen action.
    """
    if not records:
        raise ValueError("render_transcript needs at least one record")
    r0 = records[0]
    terminal = any(r.success for r in records)
    lines = [
        f"# Episode `{r0.episode_id}`",
        "",
        f"- condition **{r0.condition}** | serialisation **{r0.serialisation}** | "
        f"difficulty **{r0.difficulty}** | model `{r0.model}` | seed {r0.seed}",
        f"- {len(records)} handoffs | terminal success: **{terminal}**",
        "",
        "| step | action | message (raw -> delivered) | progress | flags |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        raw, delivered = _md_cell(r.message_raw), _md_cell(r.message_delivered)
        msg = raw if raw == delivered else f"{raw} -> {delivered}"
        active = (("goal", r.success), ("collision", r.collision), ("stuck", r.stuck))
        flags = " ".join(name for name, on in active if on) or "-"
        lines.append(f"| {r.step} | {r.action['action']} | {msg} | {r.progress:+.3f} | {flags} |")
    return "\n".join(lines) + "\n"
