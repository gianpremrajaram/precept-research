"""Feasibility certificate tests (P1-4).

Fast unit coverage runs the budget assertions only on ``easy`` (a handful of expansions). Since
DSE-059 all three rungs certify with plannable paths, so the plannability and scripted-policy guards
below DO cover medium and hard - they are the checks whose absence let run 227886 ship.
"""

from __future__ import annotations

import math

import pytest

from preceptx.sim.actions import (
    ROTATION_STEP_DEG,
    MacroAction,
    StepConfig,
    apply_macro_action,
    read_state,
)
from preceptx.sim.arena import make_scenario
from preceptx.sim.feasibility import (
    _ANG_RES,
    BUDGET_MULTIPLIER,
    FeasibilityResult,
    PlannabilityError,
    assert_plannable,
    oracle_action,
    replay,
    scripted_policy_solves,
    solve,
)
from preceptx.sim.outcomes import reached_goal


def test_oracle_action_aligns_onto_the_lattice_then_pushes() -> None:
    """The per-pose form of the scripted policy G3's correctness limb scores against."""
    assert oracle_action(0.0) == "E"  # already flat: push
    assert oracle_action(math.radians(30.0)) == "ROT-"  # rotated +: unwind
    assert oracle_action(math.radians(-30.0)) == "ROT+"
    assert oracle_action(math.pi) == "E"  # the bar is symmetric under a half-turn
    # Inside half a rotation step of flat there is no step to take, so the policy pushes.
    assert oracle_action(math.radians(ROTATION_STEP_DEG / 2.0 - 0.1)) == "E"


def test_easy_solvable_and_budget_pads_the_optimum() -> None:
    res = solve("easy")
    assert isinstance(res, FeasibilityResult)
    assert res.solvable
    assert res.optimal_steps is not None and 0 < res.optimal_steps <= 12
    assert res.budget == math.ceil(BUDGET_MULTIPLIER * res.optimal_steps)
    assert len(res.path) == res.optimal_steps  # the returned path is the whole solution


def test_returned_path_is_sound() -> None:
    # The certificate is only trustworthy if the oracle path is real: replaying it on the actual
    # simulator (same fixed start, same StepConfig) must reach the goal.
    res = solve("easy")
    scene = make_scenario("easy")
    cfg = StepConfig()
    for action in res.path:
        apply_macro_action(scene.space, scene.load, action, cfg)
    assert reached_goal(read_state(scene.space, scene.load), scene.goal)


# --- DSE-060/063: a certificate must describe a path an agent could actually follow ---------------


def test_the_pose_dedup_bucket_stays_below_the_rotation_step() -> None:
    """A bucket wider than the step collapses consecutive rotations into one search state.

    It was a bare 18 deg - correct against the 57.8 deg step of the day, and silently wrong the
    moment DSE-059 made the step 12 deg. The planner would then prune the very poses the threading
    manoeuvre needs and could report a solvable rung unsolvable.
    """
    assert math.degrees(_ANG_RES) < ROTATION_STEP_DEG


def test_certified_paths_use_only_free_space_rotations() -> None:
    """The DSE-060 guard. A* returns sound physics, which is not the same as a followable plan.

    The DSE-058 certificates for medium and hard each threaded the channel on a rotation contact
    had cut short (34.68 and 42.49 deg against a 57.79 deg free quantum), landing in the window by
    arithmetic luck. That certifies a task solvable by exploiting contact while the agent is told
    the quantum is constant.
    """
    for difficulty in ("easy", "medium", "hard"):
        result = solve(difficulty)
        assert result.solvable
        assert_plannable(difficulty, result.path)  # raises PlannabilityError if it does not hold
        _, rotations = replay(difficulty, result.path)
        assert rotations, f"{difficulty} certifies with no rotation at all"
        for realised in rotations:
            assert realised == pytest.approx(ROTATION_STEP_DEG, abs=0.5)


def test_a_contact_exploiting_path_is_rejected() -> None:
    """The guard must actually fire, not merely be satisfiable by the paths we happen to ship."""
    # Drive the load into the channel mouth first, so the next rotation is truncated by contact.
    exploit: list[MacroAction] = ["E", "E", "E", "ROT+", "ROT+", "ROT+"]
    with pytest.raises(PlannabilityError):
        assert_plannable("hard", exploit)


def test_the_scripted_policy_solves_every_jittered_start() -> None:
    """The DSE-063 smoke, and the limb A* structurally cannot provide.

    The oracle searches from the CANONICAL pose, where the load sits exactly on the slit centre, so
    a start-pose offset the action set cannot correct is invisible to it. That is how run 227886
    shipped with a y-jitter of (1.5, 4.5) against a 1.034-unit positional quantum, capping success
    at 77/39/23% by geometry alone before either agent reasoned about anything.
    """
    for difficulty in ("easy", "medium", "hard"):
        solved, seeds = scripted_policy_solves(difficulty, seeds=16)
        assert solved == seeds, f"{difficulty}: scripted policy solved only {solved}/{seeds}"
