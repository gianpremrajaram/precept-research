from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preceptx.sim.actions import (
    BodyState,
    MacroAction,
    StepConfig,
    apply_force_handles,
    apply_macro_action,
    detect_collision,
    detect_stuck,
    read_state,
)
from preceptx.sim.arena import make_scenario

_INVERSE_PAIRS: list[tuple[MacroAction, MacroAction]] = [
    ("N", "S"),
    ("S", "N"),
    ("E", "W"),
    ("W", "E"),
    ("ROT+", "ROT-"),
    ("ROT-", "ROT+"),
]


@pytest.mark.parametrize(
    ("action", "axis", "sign"),
    [
        ("E", "com_x", 1.0),
        ("W", "com_x", -1.0),
        ("N", "com_y", 1.0),
        ("S", "com_y", -1.0),
        ("ROT+", "angle", 1.0),
        ("ROT-", "angle", -1.0),
    ],
)
def test_macro_action_moves_in_expected_direction(action: str, axis: str, sign: float) -> None:
    scenario = make_scenario("easy")
    cfg = StepConfig()
    s0 = read_state(scenario.space, scenario.load)
    apply_macro_action(scenario.space, scenario.load, action, cfg)  # type: ignore[arg-type]
    s1 = read_state(scenario.space, scenario.load)
    assert sign * (getattr(s1, axis) - getattr(s0, axis)) > 1e-3


def test_wait_is_a_noop_modulo_settling() -> None:
    scenario = make_scenario("easy")
    cfg = StepConfig()
    s0 = read_state(scenario.space, scenario.load)
    apply_macro_action(scenario.space, scenario.load, "WAIT", cfg)
    s1 = read_state(scenario.space, scenario.load)
    assert abs(s1.com_x - s0.com_x) < 1e-3
    assert abs(s1.com_y - s0.com_y) < 1e-3
    assert abs(s1.angle - s0.angle) < 1e-3


@settings(max_examples=12, deadline=None)
@given(pair=st.sampled_from(_INVERSE_PAIRS))
def test_inverse_actions_return_near_origin(pair: tuple[MacroAction, MacroAction]) -> None:
    a, b = pair
    scenario = make_scenario("easy")
    cfg = StepConfig()
    s0 = read_state(scenario.space, scenario.load)
    apply_macro_action(scenario.space, scenario.load, a, cfg)
    apply_macro_action(scenario.space, scenario.load, b, cfg)
    s1 = read_state(scenario.space, scenario.load)
    assert abs(s1.com_x - s0.com_x) < 0.05
    assert abs(s1.com_y - s0.com_y) < 0.05
    assert abs(s1.angle - s0.angle) < 0.05


def test_detect_collision_true_against_wall() -> None:
    scenario = make_scenario("hard")
    cfg = StepConfig()
    for _ in range(6):
        apply_macro_action(scenario.space, scenario.load, "E", cfg)
    assert detect_collision(read_state(scenario.space, scenario.load))


def test_detect_collision_false_in_open_chamber() -> None:
    scenario = make_scenario("easy")
    cfg = StepConfig()
    apply_macro_action(scenario.space, scenario.load, "WAIT", cfg)
    assert not detect_collision(read_state(scenario.space, scenario.load))


def _body(com_x: float, com_y: float) -> BodyState:
    return BodyState(
        com_x=com_x, com_y=com_y, angle=0.41, vx=0.0, vy=0.0, omega=0.0, in_contact=True
    )


def test_detect_stuck_true_when_com_static() -> None:
    # Scripted jam: the COM holds across the window even though the pose is in contact.
    jammed = _body(3.56, 2.50)
    assert detect_stuck([jammed] * 5)


def test_detect_stuck_true_on_period_two_limit_cycle() -> None:
    """N,S,N,S returns the COM to where it started - the dominant E3 attempt-1 failure.

    The pre-v5 span form scored this False (the COM moves a full unit every step), so the field
    that exists to name a trajectory going nowhere missed the failure mode that consumed the run.
    """
    cycle = [
        _body(2.36, 3.0),
        _body(2.36, 4.0),
        _body(2.36, 3.0),
        _body(2.36, 4.0),
        _body(2.36, 3.0),
    ]
    assert detect_stuck(cycle)


def test_detect_stuck_true_on_jittering_wall_press() -> None:
    """E pressed into a wall it cannot pass: per-step contact jitter, zero net displacement."""
    press = [_body(3.78 + j, 3.0) for j in (0.0, 0.03, -0.02, 0.04, 0.005)]
    assert detect_stuck(press)


def test_detect_stuck_false_when_moving() -> None:
    scenario = make_scenario("easy")
    cfg = StepConfig()
    states: list[BodyState] = []
    for _ in range(6):
        apply_macro_action(scenario.space, scenario.load, "E", cfg)
        states.append(read_state(scenario.space, scenario.load))
    assert not detect_stuck(states)


def _trajectory(actions: list[MacroAction]) -> list[tuple[float, ...]]:
    scenario = make_scenario("medium")
    cfg = StepConfig()
    traj: list[tuple[float, ...]] = []
    for action in actions:
        apply_macro_action(scenario.space, scenario.load, action, cfg)
        s = read_state(scenario.space, scenario.load)
        traj.append((s.com_x, s.com_y, s.angle, s.vx, s.vy, s.omega))
    return traj


def test_fixed_action_sequence_is_deterministic() -> None:
    actions: list[MacroAction] = ["E", "ROT+", "N", "E", "ROT-", "W"]
    assert _trajectory(actions) == _trajectory(actions)


def test_force_handles_equal_translate_opposed_rotate() -> None:
    cfg = StepConfig()
    # Equal forces on both grips: net translation, negligible rotation.
    s = make_scenario("easy")
    t0 = read_state(s.space, s.load)
    apply_force_handles(s.space, s.load, (0.0, 3.0), (0.0, 3.0), cfg)
    t1 = read_state(s.space, s.load)
    # Impulses are BODY-frame, and the canonical pose is broadside (DSE-058), so body +y is world
    # -x. Assert on displacement magnitude rather than a world axis: the property under test is
    # "equal forces translate and do not rotate", which holds at any start orientation.
    assert math.hypot(t1.com_x - t0.com_x, t1.com_y - t0.com_y) > 1e-3
    assert abs(t1.angle - t0.angle) < 1e-2

    # Opposed forces on the two grips: a couple, so rotation.
    s = make_scenario("easy")
    r0 = read_state(s.space, s.load)
    apply_force_handles(s.space, s.load, (0.0, 3.0), (0.0, -3.0), cfg)
    r1 = read_state(s.space, s.load)
    assert abs(r1.angle - r0.angle) > 1e-2
