"""Outcome geometry and the four candidate Y labels for CPVI.

CPVI predicts an outcome Y from each handoff; the roadmap pins four options and computes all four so
the headline choice is a later analysis decision (DSE-022), not baked in here. The spatial primitive
is a geodesic distance to the goal routed through the slit centres (a straight line would cut
through the internal walls). The forward-looking labels (next-k progress, terminal success) need the
trajectory, so ``label_episode`` is a post-episode pass over the ordered handoff records.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import HandoffRecord, StatePayload
from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal, chamber_of


class OutcomeConfig(BaseModel):
    """Labeller knobs. ``k`` is the forward horizon in steps for the progress labels; the discrete
    bucket is the chamber index (from geometry), so ``k`` is the only free parameter. ``k`` is fixed
    from the pilot and documented before the main sweep, with k-sensitivity reported."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=3, ge=1)


def _slit_waypoints(geometry: ArenaGeometry) -> list[tuple[float, float]]:
    """The two slit centres the load must route through, left to right."""
    return [(geometry.chamber_w, geometry.slit_y), (2.0 * geometry.chamber_w, geometry.slit_y)]


def geodesic_distance(com: tuple[float, float], goal: Goal, geometry: ArenaGeometry) -> float:
    """Distance from ``com`` to the goal, routed through the slit centres ahead, not through walls.

    Assumes the goal is in chamber three (true by construction in ``make_scenario``): from the
    current chamber, hop through each remaining slit centre and then straight to the goal.
    """
    chamber = chamber_of(com[0], geometry)
    pts = [com, *_slit_waypoints(geometry)[chamber - 1 :], (goal.center_x, goal.center_y)]
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def reached_goal(state: BodyState, goal: Goal) -> bool:
    """Whether the load COM is within the circular goal region."""
    return math.dist((state.com_x, state.com_y), (goal.center_x, goal.center_y)) <= goal.radius


def step_progress(pre: BodyState, post: BodyState, goal: Goal, geometry: ArenaGeometry) -> float:
    """Signed geodesic reduction from ``pre`` to ``post`` (positive = moved toward the goal)."""
    return geodesic_distance((pre.com_x, pre.com_y), goal, geometry) - geodesic_distance(
        (post.com_x, post.com_y), goal, geometry
    )


def _com(payload: StatePayload) -> tuple[float, float]:
    """Pull the COM out of a serialised physics-state dict; fails loud on a malformed payload."""
    return float(payload["com_x"]), float(payload["com_y"])


def label_episode(
    records: list[HandoffRecord], goal: Goal, geometry: ArenaGeometry, cfg: OutcomeConfig
) -> list[HandoffRecord]:
    """Fill the four Y labels on each handoff from the full trajectory (a post-episode pass).

    - ``y_binary_progress``: net geodesic progress over the next ``k`` steps is positive.
    - ``y_continuous_displacement``: that same signed net progress (the unthresholded twin of the
      binary label, so the analysis can ask whether the continuous form carries more usable info).
    - ``y_discrete_config``: the chamber the load is in at the handoff (1, 2 or 3).
    - ``y_terminal_success``: the episode reaches the goal at this step or any later one.

    The window anchors on the state entering the handoff (``pre_state``) and ends on the post-state
    ``k`` actions on, so ``k=1`` recovers exactly ``step_progress`` for that handoff.
    """
    n = len(records)
    geod_pre = [geodesic_distance(_com(r.pre_state), goal, geometry) for r in records]
    geod_post = [geodesic_distance(_com(r.post_state), goal, geometry) for r in records]
    successes = [r.success for r in records]

    labelled: list[HandoffRecord] = []
    for i, r in enumerate(records):
        end = min(i + cfg.k - 1, n - 1)  # post-state k actions on from this handoff
        net = geod_pre[i] - geod_post[end]
        labelled.append(
            r.model_copy(
                update={
                    "y_binary_progress": net > 0.0,
                    "y_continuous_displacement": net,
                    "y_discrete_config": chamber_of(_com(r.pre_state)[0], geometry),
                    "y_terminal_success": any(successes[i:]),
                }
            )
        )
    return labelled
