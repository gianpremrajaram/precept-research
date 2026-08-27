"""Is rotation actually necessary? The rung-2 acceptance check, on CPU, with no model in the loop.

E3 attempt 2 failed because the easy cell does not test coordination: the T's y-extent never exceeds
1.553 at any orientation, so a 1.8-wide slit clears head-on from every angle in the circle, and a
policy of pushing east alone solves most seeds. PREREGISTRATION SS6 fixes the rung-2 criterion as
"the A* optimum must contain >= 1 rotation and finish strictly inside budget, for every jittered
seed at every difficulty". This measures both halves of that.

    uv run python scripts/check_rotation_need.py            # the pilot's ten seeds
    uv run python scripts/check_rotation_need.py --seeds 20 # a wider sample
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from preceptx.agents.graph import _JITTER_SALT
from preceptx.sim.actions import StepConfig, apply_macro_action, read_state
from preceptx.sim.arena import ScenarioJitter, make_scenario, slit_width_for
from preceptx.sim.feasibility import STEP_BUDGETS, reached_goal
from preceptx.sim.load import t_shape_verts

DIFFICULTIES = ("easy", "medium", "hard")


def extent_range() -> tuple[float, float, float]:
    """Min and max y-extent of the T outline over a full turn, and the angle of the max.

    A slit wider than the maximum admits the load head-on whatever its orientation, which is what
    makes rotation unnecessary rather than merely unused.
    """
    bar, stem = t_shape_verts()
    verts = np.array([*bar, *stem])
    theta = np.linspace(-math.pi, math.pi, 20001)[:, None]
    y = verts[:, 0][None, :] * np.sin(theta) + verts[:, 1][None, :] * np.cos(theta)
    ext = y.max(1) - y.min(1)
    return float(ext.min()), float(ext.max()), math.degrees(theta[int(ext.argmax()), 0])


def solves_without_rotation(difficulty: str, seed: int) -> tuple[bool, int, float]:
    """Run the rotation-free policy (close the y gap, then push east) on one jittered seed."""
    scenario = make_scenario(
        difficulty, rng=np.random.default_rng([seed, _JITTER_SALT]), jitter=ScenarioJitter()
    )
    start = read_state(scenario.space, scenario.load)
    budget = STEP_BUDGETS[difficulty]
    config = StepConfig()
    for step in range(budget):
        state = read_state(scenario.space, scenario.load)
        gap = state.com_y - scenario.goal.center_y
        action = ("S" if gap > 0 else "N") if abs(gap) > scenario.goal.radius else "E"
        apply_macro_action(scenario.space, scenario.load, action, config)
        if reached_goal(read_state(scenario.space, scenario.load), scenario.goal):
            return True, step + 1, math.degrees(start.angle)
    return False, budget, math.degrees(start.angle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="number of jittered seeds to test")
    args = parser.parse_args()

    lo, hi, at = extent_range()
    print(f"T y-extent over a full turn: min {lo:.3f} (0 deg), max {hi:.3f} (at {at:+.1f} deg)\n")

    failures = []
    for difficulty in DIFFICULTIES:
        slit = slit_width_for(difficulty)
        head_on = "ALWAYS (rotation cannot be required)" if hi < slit else "not at every angle"
        solved = [solves_without_rotation(difficulty, s) for s in range(args.seeds)]
        n_ok = sum(ok for ok, _, _ in solved)
        print(f"{difficulty:<7} slit {slit} | clears head-on: {head_on}")
        print(f"        rotation-free policy solves {n_ok}/{args.seeds} seeds within budget")
        print(f"        solved seeds: {[s for s, (ok, _, _) in enumerate(solved) if ok]}")
        if hi < slit or n_ok:
            failures.append(difficulty)

    print()
    if failures:
        print(f"REJECTED - rotation is not necessary at: {', '.join(failures)}")
        print("The rung-2 criterion requires every difficulty to need a rotation. Not met.")
        return 1
    print("ACCEPTED - every difficulty requires a rotation and is solvable inside budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
