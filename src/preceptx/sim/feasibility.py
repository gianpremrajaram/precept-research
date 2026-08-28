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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ConfigError
from preceptx.data.schema import Difficulty
from preceptx.sim.actions import (
    ROTATION_STEP_DEG,
    MacroAction,
    StepConfig,
    apply_macro_action,
    read_state,
)
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
# tight slits), coarse enough to bound the frontier.
#
# The angle bucket is DERIVED from the rotation step and must stay below it (DSE-059). It was a bare
# 18 deg, correct against the 57.8 deg step of the day and silently wrong the moment the step became
# 12 deg: a bucket wider than the step collapses consecutive rotations into one search state, so the
# planner prunes the very poses the threading manoeuvre needs and can report a solvable rung
# unsolvable. Half the step keeps every distinct rotation distinguishable with margin.
_POS_RES = 0.15
_ANG_RES = math.radians(ROTATION_STEP_DEG / 2.0)
if math.radians(ROTATION_STEP_DEG) <= _ANG_RES:  # pragma: no cover - guards a future retune
    raise ConfigError(
        f"pose dedup bucket ({math.degrees(_ANG_RES):.2f} deg) is not below the rotation step "
        f"({ROTATION_STEP_DEG:.2f} deg); consecutive rotations would collapse into one search state"
    )

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

# Certified per-difficulty step budgets, FROZEN from ``certify()`` on the CORRECTED geometry
# (DSE-059: convex 1.4x0.3 bar, channel depth 1.5, apertures 1.20/0.80/0.64, broadside canonical
# pose, orientation held by infinite moment, 12 deg rotation step). Easy solves in 12 steps with
# FIVE rotations; medium and hard in 14 with seven. Budget is ceil(2.5 x optimum), so 30/35/35.
#
# The certified paths are now PLANNABLE: every rotation is a free-space rotation, so each is exactly
# ROTATION_STEP_DEG and the whole solution is "rotate k times, then push east". The DSE-058 paths
# were not - medium's and hard's each depended on a CONTACT-TRUNCATED rotation (34.68 and 42.49 deg
# against a 57.79 deg free quantum) landing by luck inside the window. Those certificates were sound
# as physics and useless as a target: they certified a task solvable by exploiting contact, and the
# agents were then asked to solve it by reasoning. ``certify_plannable`` now enforces the
# difference.
#
# Difficulty grades by ROTATION-COUNT SLACK, not by rotation count: medium and hard both need seven
# rotations, but easy tolerates a miscount of +/-3, medium +/-1 and hard +/-0. Separating them by
# count instead would require hard's window to exclude the lattice point medium's admits, which is
# exactly the knife-edge geometry DSE-059 removed - so the trade is deliberate and is logged in D26.
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
    "easy": 30,
    "medium": 35,
    "hard": 35,
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

# A realised rotation may fall short of the free-space quantum only by this much before the path
# counts as contact-exploiting. Free rotation is exactly deterministic, so any shortfall is contact.
_PLANNABLE_ROT_TOL_DEG = 0.5


class PlannabilityError(ConfigError):
    """A certified path reaches the goal only by exploiting contact (DSE-060)."""


def replay(
    difficulty: Difficulty, path: list[MacroAction], *, step_cfg: StepConfig | None = None
) -> tuple[bool, list[float]]:
    """Replay ``path`` from the canonical start; return (reached goal, realised rotation
    degrees)."""
    step_cfg = step_cfg or StepConfig()
    scenario = make_scenario(difficulty)
    body = scenario.load
    rotations: list[float] = []
    for action in path:
        before = body.angle
        apply_macro_action(scenario.space, body, action, step_cfg)
        if action in ("ROT+", "ROT-"):
            rotations.append(abs(math.degrees(body.angle - before)))
    return reached_goal(read_state(scenario.space, body), scenario.goal), rotations


def assert_plannable(
    difficulty: Difficulty, path: list[MacroAction], *, step_cfg: StepConfig | None = None
) -> None:
    """Fail loud if a certified path only works by exploiting contact-truncated rotation.

    The guard for the DSE-060 failure. A* searches real physics, so anything it returns is *sound* -
    but soundness is not the property a task certificate needs. The DSE-058 certificates for medium
    and hard each threaded the channel on a rotation that contact had cut short (34.68 and 42.49 deg
    against a 57.79 deg free quantum), landing inside the window by arithmetic luck. That certifies
    a
    task solvable by exploiting contact dynamics, and the agents are then asked to solve it by
    reasoning about a quantum the record says is constant. A plannable path is one whose every
    rotation is the quantum the agent is told about, so the plan an agent can state is a plan that
    works.
    """
    reached, rotations = replay(difficulty, path, step_cfg=step_cfg)
    if not reached:
        raise PlannabilityError(
            f"certified path for {difficulty!r} does not reach the goal on replay"
        )
    truncated = [r for r in rotations if abs(r - ROTATION_STEP_DEG) > _PLANNABLE_ROT_TOL_DEG]
    if truncated:
        raise PlannabilityError(
            f"certified path for {difficulty!r} exploits contact: rotations "
            f"{[round(r, 2) for r in truncated]} deg differ from the free quantum "
            f"{ROTATION_STEP_DEG} deg by more than {_PLANNABLE_ROT_TOL_DEG} deg. The path is sound "
            "physics but not a plan an agent could state and execute."
        )


def scripted_policy_solves(
    difficulty: Difficulty, *, seeds: int = 10, step_cfg: StepConfig | None = None
) -> tuple[int, int]:
    """Run the obvious plan - rotate onto the nearest passing angle, then push east - per seed.

    The DSE-063 smoke, and the thing no A* certificate can tell you: the oracle proves a path
    exists,
    this proves the path a competent planner would actually choose is one of them. It issues no
    model
    calls, so it runs on a laptop in seconds and gates a GPU submission rather than trailing it.
    """
    step_cfg = step_cfg or StepConfig()
    budget = STEP_BUDGETS[difficulty]
    solved = 0
    for seed in range(seeds):
        scenario = make_scenario(difficulty, rng=np.random.default_rng(seed))
        body = scenario.load
        # rotations onto the lattice point nearest flat (mod 180), then push east for the remainder
        theta = math.degrees(body.angle) % 180.0
        k = round((theta if theta <= 90.0 else theta - 180.0) / ROTATION_STEP_DEG)
        plan: list[MacroAction] = ["ROT-" if k > 0 else "ROT+"] * abs(k)
        plan += ["E"] * (budget - len(plan))
        for action in plan[:budget]:
            apply_macro_action(scenario.space, body, action, step_cfg)
            if reached_goal(read_state(scenario.space, body), scenario.goal):
                solved += 1
                break
    return solved, seeds


# The scripted policy must solve at least this share of jittered seeds for a task to certify. Set to
# 1.0 deliberately: the policy is the obvious plan against a deterministic actuator, so anything
# below 1.0 means some start pose cannot be corrected by the action set - the DSE-059 fault - and
# that is a property of the task, not of the agent. There is no reason to tolerate any of it.
_SCRIPTED_POLICY_MIN = 1.0
_SCRIPTED_POLICY_SEEDS = 16


def certify(step_cfg: StepConfig | None = None) -> dict[Difficulty, FeasibilityResult]:
    """Certify every difficulty: solvable, by a PLANNABLE path, and reachable by the obvious policy.

    Three limbs, because run 227886 passed the first and failed the other two invisibly
    (DSE-059/060,
    D26). A* proves a path exists; ``assert_plannable`` proves it is a path an agent could state;
    ``scripted_policy_solves`` proves the plan a competent agent would actually choose works from
    every jittered start, which is the only limb that sees the start-pose distribution at all.
    """
    results = {d: solve(d, step_cfg=step_cfg) for d in _DIFFICULTIES}
    for difficulty, result in results.items():
        if not result.solvable:
            continue
        assert_plannable(difficulty, result.path, step_cfg=step_cfg)
        solved, seeds = scripted_policy_solves(
            difficulty, seeds=_SCRIPTED_POLICY_SEEDS, step_cfg=step_cfg
        )
        if solved < _SCRIPTED_POLICY_MIN * seeds:
            raise PlannabilityError(
                f"{difficulty!r} certifies as solvable but the scripted rotate-then-push policy "
                f"solves only {solved}/{seeds} jittered starts. A* searches from the "
                "canonical pose, so a start-pose offset the action set cannot correct is "
                "invisible to it."
            )
    return results


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
