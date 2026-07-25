"""Task-feasibility certificate (P1-4): the tight-slit difficulties stay solvable within budget.

Slower than a unit test - it runs the threading search on medium and hard (~7s) - so it lives in
the integration tier. Feasibility is load-bearing: if a physics change silently makes medium/hard
unsolvable, the pilot's G1/G2 gates would misread it as "the models can't coordinate" (the P1-4
misdiagnosis), far more expensive to catch after a run than here.
"""

from __future__ import annotations

import pytest

from preceptx.data.schema import Difficulty
from preceptx.sim.feasibility import STEP_BUDGETS, solve


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_tight_difficulty_is_feasible_within_budget(difficulty: Difficulty) -> None:
    res = solve(difficulty)
    assert res.solvable, f"{difficulty} is no longer solvable - the geometry/physics regressed"
    assert res.optimal_steps is not None
    assert res.optimal_steps <= STEP_BUDGETS[difficulty]  # the frozen budget must cover the optimum
