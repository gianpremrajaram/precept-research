"""Feasibility certificate tests (P1-4).

Fast unit coverage runs the search only on ``easy`` (a handful of expansions); ``medium``/``hard``
are slow and, on the shipped geometry, marginal/infeasible - a task-design finding covered in the
CHANGELOG and the review notes, not asserted here as a green expectation.
"""

from __future__ import annotations

import math

from preceptx.sim.actions import StepConfig, apply_macro_action, read_state
from preceptx.sim.arena import make_scenario
from preceptx.sim.feasibility import BUDGET_MULTIPLIER, FeasibilityResult, solve
from preceptx.sim.outcomes import reached_goal


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
