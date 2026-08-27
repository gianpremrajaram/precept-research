"""Regression tests for the rung-2 CPU instruments (DSE-057, DSE-058).

These scripts gate a GPU submission and a research-design decision, so a silent wrong answer from
either is expensive. Both have now given one. `check_rotation_need.py` first decided "is rotation
necessary?" with one hand-written policy, which cannot tell "no translation-only path exists" from
"my policy did not find one" - and that difference was the whole finding. Then the successor
geometry was chosen on fixed illustrative poses and leaked on 7 of 10 real jittered seeds, once
passive self-alignment was measured. The tests below pin what each episode taught.

**On the predecessor T task.** Its falsification - a translation-only oracle path exists at every
shipped slit width - is a recorded *finding*, not a live test. Reproducing it from source would need
the solver parameterised over load shape and wall type, which is a larger refactor than it is worth;
the numbers, the method and the commit are in `docs/experiment_design_log.md` (2026-08-27) and
`docs/EXPERIMENTS.md`. What survives here is the geometric fact the finding rests on, which
`t_shape_verts` still supports, plus the T outline itself so it cannot be quietly deleted.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from preceptx.sim.arena import slit_width_for
from preceptx.sim.feasibility import STEP_BUDGETS, FeasibilityResult, solve
from preceptx.sim.load import BAR_LEN, BAR_THICK, t_shape_verts

REPO_ROOT = Path(__file__).resolve().parents[3]
_TRANSLATIONS = ("N", "S", "E", "W")
_ROTATIONS = frozenset({"ROT+", "ROT-"})
DIFFICULTIES = ("easy", "medium", "hard")


def _load(name: str) -> Any:
    """Import one of the ``scripts/`` modules by path; they are tools, not an installed package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_translation_only_path_exists_at_any_difficulty(difficulty: str) -> None:
    """The successor task's defining property, and the one the predecessor failed.

    Exhausting the restricted search - not a failing hand-written policy - is what establishes that
    rotation is necessary. If this starts passing a translation-only path, the manipulation has been
    lost again and no GPU time should be spent until it is understood.
    """
    result = solve(difficulty, actions=_TRANSLATIONS, max_depth=STEP_BUDGETS[difficulty])
    assert not result.solvable, (
        f"{difficulty} admits a translation-only path - rotation is optional"
    )


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_the_optimum_is_in_budget_and_turns_the_load(difficulty: str) -> None:
    result = solve(difficulty)
    assert result.solvable and result.optimal_steps is not None
    assert result.optimal_steps <= STEP_BUDGETS[difficulty]
    assert any(a in _ROTATIONS for a in result.path), "optimum never rotates"


def test_the_ladder_sits_inside_the_certified_band() -> None:
    """Both edges are geometric once orientation is grip-held.

    The effective aperture is the nominal width less the two wall radii. It must exceed the bar
    thickness for any orientation to pass, and stay under the bar length or the load clears the
    channel broadside with no turn at all. Before `hold_orientation` the usable band was far
    narrower and its upper edge was an empirical self-alignment threshold near 0.5; holding the
    orientation restored the full geometric band, which is why the ladder can now spread.
    """
    for difficulty in DIFFICULTIES:
        effective = slit_width_for(difficulty) - 2 * 0.05
        assert effective > BAR_THICK, f"{difficulty} admits no orientation at all"
        assert effective < BAR_LEN, f"{difficulty} lets the load through broadside"


def test_check_seed_accepts_every_declared_seed() -> None:
    """The instrument must accept the shipped ladder on the exact poses the pilot will run."""
    check = _load("check_rotation_need")
    for difficulty in DIFFICULTIES:
        verdicts = [check.check_seed(difficulty, s)[0] for s in range(10)]
        assert all(v == check._OK for v in verdicts), f"{difficulty}: {set(verdicts)}"


def test_an_unsolvable_arena_is_rejected_rather_than_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The false-accept the first version shipped: it never checked solvability at all.

    Monkeypatched rather than run for real because proving unsolvability by exhaustion is far too
    slow for the unit tier.
    """
    check = _load("check_rotation_need")
    unsolvable = FeasibilityResult(
        difficulty="hard", solvable=False, optimal_steps=None, budget=None, path=[], expansions=1
    )
    monkeypatch.setattr(check, "solve", lambda *a, **k: unsolvable)
    assert check.check_seed("hard", 0)[0] == check._UNSOLVABLE


def test_passive_self_alignment_is_a_distinct_verdict() -> None:
    """Limb 7 keeps two constructs apart that the first certification pass conflated.

    "Translation-only" is a claim about the ACTION set; the body angle is a separate matter, because
    contact at the aperture mouth rotates the load with no rotate action issued. A candidate that
    passes limbs 1-3 on that basis must be rejected, not accepted.
    """
    check = _load("check_rotation_need")
    assert check.MAX_PASSIVE_DRIFT_DEG > 0
    assert check._PASSIVE_ALIGNMENT != check._ROTATION_NOT_NECESSARY


def test_limb_7_can_reject_a_seed_that_passed_limbs_1_to_3(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limb runs on the PASS path, which is the only place it can tell anyone anything.

    Its first version ran the drift check inside the branch where limb 3 had already rejected the
    seed, so it could relabel a failure but never cause one - a pre-registered acceptance criterion
    that no input could fail. This drives an otherwise-clean seed (solvable, rotation in the
    optimum, no translation-only path) through with a drift over the limit and demands a reject.
    """
    check = _load("check_rotation_need")
    ok = FeasibilityResult(
        difficulty="hard",
        solvable=True,
        optimal_steps=6,
        budget=STEP_BUDGETS["hard"],
        path=["ROT+", "E", "E"],
        expansions=9,
    )
    none_found = FeasibilityResult(
        difficulty="hard", solvable=False, optimal_steps=None, budget=None, path=[], expansions=9
    )
    calls = iter([ok, none_found])
    monkeypatch.setattr(check, "solve", lambda *a, **k: next(calls))
    monkeypatch.setattr(check, "passive_drift_deg", lambda *a, **k: 90.0)
    assert check.check_seed("hard", 0)[0] == check._PASSIVE_ALIGNMENT


def test_the_drift_instrument_can_actually_see_drift() -> None:
    """The measurement must be non-vacuous, which the shipped `StepConfig` alone does not give.

    `hold_orientation` restores the pre-action angle after every non-rotate action, so drift under
    the shipped config is identically zero on every input - the first limb 7 measured exactly that
    and reported 0.0 for every seed it ever saw. Both halves are pinned: zero under the guard (the
    property the task relies on) and non-zero without it (proof the instrument is not stuck at 0).
    """
    check = _load("check_rotation_need")
    push = ["E"] * 10
    assert check.passive_drift_deg("hard", 0, push) == pytest.approx(0.0, abs=1e-9)
    assert check.unheld_drift_deg("hard", 0, push) > 1.0


def test_the_predecessor_t_outline_is_preserved_with_its_geometry() -> None:
    """The T is the subject of a published falsification; deleting it would orphan the finding.

    The load-bearing number is its extent envelope: a minimum of 1.300 at 0 deg against a shipped
    easy slit of 1.8, which is why rotation could not be required there at any angle.
    """
    bar, stem = t_shape_verts()
    verts = np.array([*bar, *stem])
    theta = np.linspace(-math.pi, math.pi, 4001)[:, None]
    y = verts[:, 0][None, :] * np.sin(theta) + verts[:, 1][None, :] * np.cos(theta)
    extent = y.max(1) - y.min(1)
    assert float(extent.min()) == pytest.approx(1.300, abs=1e-3)
    assert float(extent.max()) == pytest.approx(1.553, abs=1e-3)


def test_terminal_cycle_counts_actions_not_repetitions() -> None:
    """The constant thresholds trailing actions; the CHANGELOG used to call them repetitions."""
    diagnose = _load("diagnose_cycles")
    assert diagnose.MIN_CYCLE_ACTIONS == 4
    assert diagnose.terminal_cycle(["ROT+", "ROT-", "ROT+"]) == 0
    assert diagnose.terminal_cycle(["ROT+", "ROT-", "ROT+", "ROT-"]) == 4
    assert diagnose.terminal_cycle(["E", "E", "E", "E"]) == 4
    assert diagnose.terminal_cycle(["N", "E", "E", "E", "E"]) == 4
