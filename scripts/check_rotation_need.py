"""Is rotation actually necessary? The rung-2 acceptance check, on CPU, with no model in the loop.

E3 attempt 2 failed on a cell that does not test coordination, so PREREGISTRATION SS6 fixes a rung-2
criterion the arena must meet before any re-gate. This decides it per difficulty and per jittered
seed, with the same A* oracle the step budgets came from:

  1. the full-action optimum is solvable and finishes inside the certified budget;
  2. that optimum contains at least one rotation; and
  3. the SAME search restricted to translations alone is exhausted without reaching the goal.

Limb 3 is the load-bearing one and it is why this script calls the solver rather than a policy. A
hand-written rotation-free policy that fails shows only that *that* policy fails; exhausting the
restricted search shows no translation-only path exists at all. The first version of this script
made exactly that error and certified a false premise - it reported medium and hard as needing a
rotation when a translation-only oracle solves both (DSE-057).

No monotone-rotation probe: a rotation-free path is trivially monotone, so limb 3 subsumes it.

    uv run python scripts/check_rotation_need.py            # the pilot's ten seeds
    uv run python scripts/check_rotation_need.py --seeds 20 # a wider sample
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from preceptx.agents.graph import _JITTER_SALT
from preceptx.sim.actions import MacroAction, StepConfig, apply_macro_action, read_state
from preceptx.sim.arena import ScenarioJitter, make_scenario, slit_width_for
from preceptx.sim.feasibility import CERTIFICATION_STEP_CONFIG, STEP_BUDGETS, solve
from preceptx.sim.load import t_shape_verts

DIFFICULTIES = ("easy", "medium", "hard")
_PUSH: MacroAction = "E"  # the probe motion for the reported sensitivity: push into the channel
_PUSH_STEPS = 10
_TRANSLATIONS: tuple[MacroAction, ...] = ("N", "S", "E", "W")
_ROTATIONS = frozenset({"ROT+", "ROT-"})

# Verdicts, worst first. Only OK passes; each other value names which limb failed and why.
_UNSOLVABLE = "unsolvable"  # no full-action path at all - the arena is broken, not merely easy
_OVER_BUDGET = "over_budget"  # solvable, but the optimum exceeds the budget the agents are given
_NO_ROTATION_IN_OPTIMUM = "zero_rotation_optimum"  # the cheapest path never turns the load
_ROTATION_NOT_NECESSARY = "translation_only_feasible"  # a translation-only path reaches the goal
_PASSIVE_ALIGNMENT = "passive_self_alignment"  # no rotate action, but contact turned the load
_OK = "ok"

# Limb 7 (DSE-058). "Translation-only" is a claim about the ACTION set, not about the body angle:
# macro impulses are applied at the COM and carry no torque, but contact at the aperture mouth
# rotates the load anyway. A candidate that passes limbs 1-3 can still be degenerate if the load
# aligns itself, so the realised ANGLE trajectory is measured separately. More than this much drift
# under translation-only actions means reorientation is available for free.
#
# The limb runs on seeds that PASS limbs 1-3 - that is the case limbs 1-3 cannot see. Its first
# version ran it only where limb 3 had already REJECTED the seed, so it could relabel a failure but
# never cause one; and it measured under the shipped StepConfig, where `hold_orientation` restores
# the pre-action angle after every non-rotate action and the drift is identically zero. It could not
# fire on any input. Both are fixed here, and `unheld_drift_deg` reports the counterfactual.
MAX_PASSIVE_DRIFT_DEG = 15.0


def extent_range() -> tuple[float, float, float]:
    """Min and max y-extent of the T outline over a full turn, and the angle of the max.

    Context for the report, deliberately *not* part of the criterion: a slit wider than the maximum
    proves rotation is unnecessary, but a slit narrower than it proves nothing, because the load can
    still cross a thin wall one member at a time without ever turning.
    """
    bar, stem = t_shape_verts()
    verts = np.array([*bar, *stem])
    theta = np.linspace(-math.pi, math.pi, 20001)[:, None]
    y = verts[:, 0][None, :] * np.sin(theta) + verts[:, 1][None, :] * np.cos(theta)
    ext = y.max(1) - y.min(1)
    return float(ext.min()), float(ext.max()), math.degrees(theta[int(ext.argmax()), 0])


def _scenario(difficulty: str, seed: int) -> object:
    """The declared start state for one seed: the exact pose the pilot will run."""
    return make_scenario(
        difficulty, rng=np.random.default_rng([seed, _JITTER_SALT]), jitter=ScenarioJitter()
    )


def passive_drift_deg(
    difficulty: str, seed: int, path: list[MacroAction], step_cfg: StepConfig | None = None
) -> float:
    """Largest |angle - start angle| reached while executing a translation-only action sequence.

    Separates the two constructs the certification must not conflate. ``explicit_rotation_actions``
    is what limb 3 tests; this is its state-level counterpart. Macro impulses are applied at the COM
    and carry no torque, but contact at the aperture mouth rotates the load anyway, so a candidate
    can pass limbs 1-3 and still be degenerate. Replayed rather than read from the search, which
    records poses at action boundaries only.

    Measured under ``step_cfg`` - the config the candidate is being certified at, so the limb tests
    the world the episodes will run in. On the shipped config that is zero by construction; see
    ``unheld_drift_deg`` for the sensitivity that number does not carry.
    """
    cfg = step_cfg if step_cfg is not None else StepConfig()
    sc = _scenario(difficulty, seed)
    start = read_state(sc.space, sc.load).angle  # type: ignore[attr-defined]
    worst = 0.0
    for action in path:
        apply_macro_action(sc.space, sc.load, action, cfg)  # type: ignore[attr-defined]
        worst = max(worst, abs(read_state(sc.space, sc.load).angle - start))  # type: ignore[attr-defined]
    return math.degrees(worst)


def unheld_drift_deg(difficulty: str, seed: int, path: list[MacroAction]) -> float:
    """The same drift with ``hold_orientation`` disabled: how much of limb 7 the guard is carrying.

    Reported, not gated. The shipped physics holds the load's angle through non-rotate actions, so
    passive self-alignment is impossible there and limb 7 passes by construction - which is a
    stronger guarantee than any per-seed check, but only as good as the assumption behind it. This
    is the number that answers "is rotation-necessity here an artefact of `hold_orientation`?", and
    it belongs in the certification report rather than in a reviewer's imagination.
    """
    return passive_drift_deg(difficulty, seed, path, StepConfig(hold_orientation=False))


def check_seed(difficulty: str, seed: int, *, certify: bool = False) -> tuple[str, str]:
    """Run every limb on one jittered start pose; return (verdict, one-line detail).

    ``certify=True`` runs the searches at ``CERTIFICATION_STEP_CONFIG`` (substeps=64). That is
    mandatory for accepting NEW geometry: a candidate was once solvable at the shipped substeps=4
    and unsolvable at 8, and a verdict that moves with the integrator cannot gate a GPU run.
    """
    scenario = _scenario(difficulty, seed)
    budget = STEP_BUDGETS[difficulty]
    cfg = CERTIFICATION_STEP_CONFIG if certify else None

    full = solve(difficulty, scenario=scenario, step_cfg=cfg)  # type: ignore[arg-type]
    if not full.solvable or full.optimal_steps is None:
        return _UNSOLVABLE, f"no path in {full.expansions} expansions"
    if full.optimal_steps > budget:
        return _OVER_BUDGET, f"optimum {full.optimal_steps} > budget {budget}"
    n_rot = sum(a in _ROTATIONS for a in full.path)
    if n_rot == 0:
        return _NO_ROTATION_IN_OPTIMUM, f"optimum {full.optimal_steps} steps, 0 rotations"

    # Limb 3, the necessity proof: exhaust the translation-only search inside the agents' budget.
    flat = solve(
        difficulty,
        scenario=_scenario(difficulty, seed),
        actions=_TRANSLATIONS,
        max_depth=budget,
        step_cfg=cfg,  # type: ignore[arg-type]
    )
    if flat.solvable:
        return _ROTATION_NOT_NECESSARY, f"translation-only path in {flat.optimal_steps} steps"

    # Limb 7, on a seed that has now passed limbs 1-3: replay the optimum's translations alone and
    # watch the angle. If contact alone reorients the load this far, the commanded rotations the
    # optimum contains are not doing the work that limb 3 credits them with.
    translations = [a for a in full.path if a not in _ROTATIONS]
    drift = passive_drift_deg(difficulty, seed, translations, cfg)
    if drift > MAX_PASSIVE_DRIFT_DEG:
        return _PASSIVE_ALIGNMENT, f"{drift:.0f} deg of contact rotation with no rotate action"
    return _OK, f"optimum {full.optimal_steps} steps, {n_rot} rotations; no translation-only path"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="number of jittered seeds to test")
    parser.add_argument(
        "--certify",
        action="store_true",
        help="run searches at substeps=64; MANDATORY for accepting new geometry (slow)",
    )
    args = parser.parse_args()

    lo, hi, at = extent_range()
    print(f"predecessor T y-extent: min {lo:.3f} (0 deg), max {hi:.3f} (at {at:+.1f} deg)")
    print("(the falsified T, kept for reference - the shipped load is a convex bar; and the")
    print(" criterion below is decided by the solver, not by any extent)\n")

    failed: list[str] = []
    for difficulty in DIFFICULTIES:
        results = [check_seed(difficulty, s, certify=args.certify) for s in range(args.seeds)]
        # The sensitivity behind limb 7, printed with the verdict it qualifies (certification only).
        # A straight eastward push is the canonical passive-alignment scenario - it drives the load
        # into the channel mouth with no rotate action - so this needs no solver and no optimum.
        unheld = (
            max(unheld_drift_deg(difficulty, s, [_PUSH] * _PUSH_STEPS) for s in range(args.seeds))
            if args.certify
            else None
        )
        tally = Counter(v for v, _ in results)
        n_ok = tally[_OK]
        slit = slit_width_for(difficulty)
        print(f"{difficulty:<7} slit {slit} | {n_ok}/{args.seeds} seeds meet all three limbs")
        for verdict, count in tally.most_common():
            example = next(d for v, d in results if v == verdict)
            print(f"        {count:2d}x {verdict:<26} e.g. {example}")
        if unheld is not None:
            print(
                f"        worst drift without hold_orientation: {unheld:.0f} deg "
                f"over {_PUSH_STEPS} straight pushes (reported, not gated)"
            )
        if n_ok < args.seeds:
            failed.append(difficulty)

    print()
    if failed:
        print(f"REJECTED - rotation is not necessary, or not solvable, at: {', '.join(failed)}")
        print(
            "The rung-2 criterion requires every seed at every difficulty to meet all three limbs."
        )
        return 1
    print("ACCEPTED - every difficulty needs a rotation and is solvable inside budget.")
    if args.certify:
        print(
            "Limb 7 passes by construction: `hold_orientation` holds the angle through\n"
            "non-rotate actions, so no seed CAN self-align. The drift above is what that\n"
            "assumption carries - a declared modelling choice, not a tuned constant."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
