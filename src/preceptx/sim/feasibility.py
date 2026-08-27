"""Feasibility certificates and per-difficulty step budgets (P1-4).

Without a solvability proof, a pilot G1 failure is ambiguous - "the models can't do it" versus
"nobody can within the budget", the most expensive misdiagnosis available. A search over the macro
actions on the deterministic simulator finds a short oracle solution per difficulty; the step budget
is set to a multiple of that length so a capable pair has real slack (the single default of 12 was
almost certainly too small for hard, whose slit is far below the load's head-on y-extent).

Rotation is *not* required at any shipped difficulty: the walls are thin segments, so the bar and
the stem cross the gap at different instants and a translation-only path threads every slit down to
the shorter member (1.0). DSE-057's restricted-action search proves this by exhaustion. The claim
that hard "must rotate" was an inference from the head-on extent, never a search result.

The search is Markovian on the load *pose*: quasi-static settling zeroes velocity after every
action (``StepConfig.quasi_static``), so a node is fully described by ``(origin x, origin y,
angle)`` and is restored by placing a fresh load at that pose - no trajectory replay. It is A* over
the six pose-changing macro actions (``WAIT`` only settles, a no-op from a rested state), guided by
the same geodesic-to-goal distance the labeller uses, with states deduplicated on a coarse pose
grid. A* over BFS keeps the node count tractable given each expansion runs a full physics settle;
the returned path is a genuine action sequence that reaches the goal (sound by construction),
near-optimal in length, which is all the budget needs. The oracle path length doubles as a thesis
statistic (optimal ``k`` vs the LLM's path length).
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import Difficulty
from preceptx.sim.actions import MacroAction, StepConfig, apply_macro_action, read_state
from preceptx.sim.arena import (
    LOAD_MASS,
    ArenaGeometry,
    Goal,
    Scenario,
    build_arena,
    make_scenario,
    slit_width_for,
)
from preceptx.sim.load import add_load
from preceptx.sim.outcomes import geodesic_distance, reached_goal

logger = logging.getLogger(__name__)

# The six pose-changing macro actions; WAIT only settles (a no-op from a rested state under
# quasi-static settling), so it never helps reach the goal and is dropped from the search.
_SEARCH_ACTIONS: tuple[MacroAction, ...] = ("N", "S", "E", "W", "ROT+", "ROT-")

# Pose-deduplication resolution: poses within one grid cell in position AND one bucket in angle are
# the same search state. Fine enough to find the narrow threading passage for medium/hard (the
# tight slits), coarse enough to bound the frontier. The angle bucket (18 deg) is below the ~34 deg
# per-action rotation step, so consecutive rotations are never collapsed into one state.
_POS_RES = 0.15
_ANG_RES = math.pi / 10.0

# A* heuristic optimism: no single action reduces the geodesic distance to the goal by more than
# this (an E-push advances ~1 world unit under the damped regime), so geodesic / this is a lower
# bound on the remaining action count and keeps the heuristic admissible-ish (near-optimal paths).
_MAX_PROGRESS_PER_STEP = 1.5

_MAX_EXPANSIONS = 20000  # hard termination bound; exceeding it means the search space blew up
DEFAULT_MAX_DEPTH = 40  # cap on solution length explored (hard needs rotate-thread-push-thread)

# Step budget = ceil(this x the shortest oracle solution) - generous slack over the optimum for the
# start-pose jitter (P0-2) and LLM suboptimality (review P1-4: "~2-3x the found optimum").
BUDGET_MULTIPLIER = 2.5

# Two-tier collision fidelity (DSE-057). `StepConfig.substeps` is a pure resolution knob: the total
# simulated time of a macro action is `settle_steps * dt` regardless of it, so raising it changes
# only how finely contacts are resolved. The shipped default of 4 is an anti-tunnelling setting for
# the *experiment*, and every frozen E3 certificate (easy 7, medium 13, hard 13) is stable from 4 up
# to 64 - so nothing already run is invalidated and the default is deliberately left alone.
#
# It is NOT sufficient to certify NEW geometry. A candidate tunnel path was found solvable at 4 and
# died at 8: at coarse resolution a macro impulse can drive the load through an aperture narrower
# than its own outline before contact resolves. A feasibility verdict that moves with the integrator
# cannot gate a GPU run, so every new-geometry acceptance check runs at this profile instead.
# E3's results are results for the recorded, versioned default simulator - not claims about a
# continuum-physics benchmark.
CERTIFICATION_STEP_CONFIG = StepConfig(substeps=64)

# Certified per-difficulty step budgets, FROZEN from ``certify()`` on the SUCCESSOR geometry
# (DSE-058: convex 1.4x0.3 bar, channel depth 1.5, apertures 1.20/0.80/0.50, broadside canonical
# pose, orientation held through non-rotate actions). Easy solves in 8 steps with ONE rotation;
# medium and hard in 10 with two. Budget is ceil(2.5 x optimum), so 20/25/25.
#
# Budget width is part of the certification, not a free parameter bolted on afterwards: a longer
# budget admits longer degenerate paths, and an earlier ladder that certified 10/10 at budget 25
# leaked as soon as the budget rose to 28. Re-run the acceptance check after ANY budget change.
#
# Difficulty separates in the certificate itself (easy needs one rotation, medium and hard two) and
# again in tolerance: the band of orientations that clears the channel narrows as the aperture does,
# which an exact oracle barely feels and an imprecise agent feels a great deal.
#
# tests/unit/sim/test_feasibility.py re-derives and
# certifies these (budget >= the found optimum) so a physics change that breaks feasibility fails
# loudly rather than silently starving the pilot. Regenerate with
# ``python -m preceptx.sim.feasibility``.
STEP_BUDGETS: dict[Difficulty, int] = {
    "easy": 20,
    "medium": 25,
    "hard": 25,
}


class _Pose(NamedTuple):
    x: float  # body origin, not COM (the origin fully determines placement; COM is derived)
    y: float
    theta: float


class FeasibilityResult(BaseModel):
    """One difficulty's certificate: is it solvable, in how few steps, and the derived budget."""

    model_config = ConfigDict(extra="forbid")

    difficulty: Difficulty
    solvable: bool
    optimal_steps: int | None  # length of the shortest solution found (None if unsolved in bound)
    budget: int | None  # ceil(BUDGET_MULTIPLIER * optimal_steps); None when unsolved
    path: list[MacroAction] = Field(default_factory=list)  # the oracle solution (demos/figures)
    expansions: int  # nodes expanded - the search cost, for auditing


def _key(pose: _Pose) -> tuple[int, int, int]:
    """Discretise a pose to its dedup cell (position grid, angle bucket wrapped to 0..2pi)."""
    return (
        round(pose.x / _POS_RES),
        round(pose.y / _POS_RES),
        round((pose.theta % math.tau) / _ANG_RES),
    )


def _apply(
    pose: _Pose,
    action: MacroAction,
    slit: float,
    geometry: ArenaGeometry,
    goal: Goal,
    step_cfg: StepConfig,
) -> tuple[_Pose, tuple[float, float], bool]:
    """Restore ``pose`` in a fresh arena, apply one action, return (new pose, new COM, reached)."""
    space = build_arena(slit, geometry)
    body = add_load(space, (pose.x, pose.y), LOAD_MASS)
    body.angle = pose.theta
    space.reindex_shapes_for_body(body)
    apply_macro_action(space, body, action, step_cfg)
    st = read_state(space, body)
    return (
        _Pose(float(body.position.x), float(body.position.y), float(body.angle)),
        (st.com_x, st.com_y),
        reached_goal(st, goal),
    )


def solve(
    difficulty: Difficulty,
    *,
    step_cfg: StepConfig | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    actions: tuple[MacroAction, ...] | None = None,
    scenario: Scenario | None = None,
) -> FeasibilityResult:
    """A* over macro actions from a start pose to the goal for one difficulty.

    Defaults to the fixed (un-jittered) start pose - the nominal instance the budget multiplier then
    pads for jitter and model suboptimality. Sound by construction (the path is real physics); the
    heuristic makes it near-optimal, which is all the budget needs.

    ``actions`` restricts the action set and ``scenario`` supplies a jittered start (DSE-057). Both
    default to the frozen behaviour, so ``certify()`` and the budget test are unaffected.
    Restricting the set turns the search into a *necessity* proof: exhausting it without reaching
    the goal shows no path of that kind exists, which a failing hand-written policy never can.
    """
    step_cfg = step_cfg or StepConfig()
    actions = actions or _SEARCH_ACTIONS
    geometry = ArenaGeometry()
    slit = slit_width_for(difficulty)
    # canonical fixed start + goal (reuse real construction) unless the caller supplied one
    scenario = scenario or make_scenario(difficulty)
    goal = scenario.goal
    start = _Pose(
        float(scenario.load.position.x), float(scenario.load.position.y), float(scenario.load.angle)
    )
    start_com = read_state(scenario.space, scenario.load)

    def h(com: tuple[float, float]) -> float:
        return geodesic_distance(com, goal, geometry) / _MAX_PROGRESS_PER_STEP

    counter = itertools.count()  # stable heap tie-break so poses are never compared
    frontier: list[tuple[float, int, int, _Pose, list[MacroAction]]] = [
        (h((start_com.com_x, start_com.com_y)), next(counter), 0, start, [])
    ]
    visited: set[tuple[int, int, int]] = {_key(start)}
    expansions = 0

    while frontier:
        _f, _c, g, pose, path = heapq.heappop(frontier)
        if g >= max_depth:
            continue
        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            raise RuntimeError(
                f"feasibility search for {difficulty!r} exceeded {_MAX_EXPANSIONS} expansions "
                "without reaching the goal; the search space blew up (check the physics/geometry)"
            )
        for action in actions:
            new_pose, com, reached = _apply(pose, action, slit, geometry, goal, step_cfg)
            new_path = [*path, action]
            if reached:
                steps = len(new_path)
                return FeasibilityResult(
                    difficulty=difficulty,
                    solvable=True,
                    optimal_steps=steps,
                    budget=math.ceil(BUDGET_MULTIPLIER * steps),
                    path=new_path,
                    expansions=expansions,
                )
            k = _key(new_pose)
            if k in visited:
                continue
            visited.add(k)
            heapq.heappush(frontier, (g + 1 + h(com), next(counter), g + 1, new_pose, new_path))

    return FeasibilityResult(
        difficulty=difficulty,
        solvable=False,
        optimal_steps=None,
        budget=None,
        path=[],
        expansions=expansions,
    )


_DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")


def certify(step_cfg: StepConfig | None = None) -> dict[Difficulty, FeasibilityResult]:
    """Run the search for every difficulty - the certificate the test and the CLI both consume."""
    return {d: solve(d, step_cfg=step_cfg) for d in _DIFFICULTIES}


def _main() -> None:
    """Print the per-difficulty certificate and derived budgets (dev tool, not an experiment)."""
    logging.basicConfig(level=logging.INFO)
    for d, res in certify().items():
        logger.info(
            "%s: solvable=%s optimal_steps=%s budget=%s expansions=%d path=%s",
            d,
            res.solvable,
            res.optimal_steps,
            res.budget,
            res.expansions,
            "".join(a.ljust(4) for a in res.path),
        )


if __name__ == "__main__":
    _main()
