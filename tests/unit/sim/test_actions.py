from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preceptx.sim.actions import (
    ANGULAR_IMPULSE,
    DEG_PER_UNIT_IMPULSE,
    ROTATION_STEP_DEG,
    BodyState,
    MacroAction,
    StepConfig,
    apply_force_handles,
    apply_macro_action,
    detect_collision,
    detect_stuck,
    measure_rotation_step,
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


# --- DSE-059: the rotation quantum, and the orientation hold that must actually hold -------------


def test_the_shipped_impulse_realises_the_authored_rotation_step() -> None:
    """The guard that would have caught DSE-059 the day DSE-057 changed the load.

    ``angular_impulse`` is derived from ``ROTATION_STEP_DEG`` through a measured constant, and the
    constant is only valid for the load's moment of inertia. When the T became a bar of 1.71x
    smaller moment, the same impulse silently went from ~34 to 57.8 deg per action - and nothing
    failed,
    because the only thing asserting the intent was a code comment. This asserts it instead.
    """
    assert measure_rotation_step() == pytest.approx(ROTATION_STEP_DEG, abs=0.05)
    assert pytest.approx(ROTATION_STEP_DEG / DEG_PER_UNIT_IMPULSE) == ANGULAR_IMPULSE


def test_free_rotation_is_deterministic_and_linear_in_the_impulse() -> None:
    """Free rotation has no noise at all, so the quantum is exact rather than an average.

    Load-bearing for the whole design: reachability is a lattice argument, and a lattice only exists
    if the step is a constant. Any spread observed in a real run is therefore contact truncation,
    which can only ever *reduce* the realised angle - never scatter it either way.
    """
    for impulse in (0.05, ANGULAR_IMPULSE, 0.3):
        steps = {
            round(measure_rotation_step(StepConfig(angular_impulse=impulse)), 9) for _ in range(3)
        }
        assert len(steps) == 1, f"rotation at impulse {impulse} is not deterministic: {steps}"
    half = measure_rotation_step(StepConfig(angular_impulse=ANGULAR_IMPULSE / 2.0))
    assert half == pytest.approx(ROTATION_STEP_DEG / 2.0, abs=0.02)


def test_hold_orientation_prevents_contact_rotation_rather_than_reverting_it() -> None:
    """The DSE-059 regression: the load must not rotate *during* a non-rotate action either.

    The original hold restored the angle after the settle, so contact could spin the load flat, slip
    it through a channel it does not fit, and hand back the original angle. From 30 deg on medium
    (geometric window +/-17.2 deg) the load reached 0.48 deg mid-action and passed. Sampling the
    angle inside the settle is the only way to see it: every after-the-fact read said 30.00.
    """
    scenario = make_scenario("medium")
    body = scenario.load
    body.angle = math.radians(30.0)
    scenario.space.reindex_shapes_for_body(body)
    config = StepConfig()

    sampled: list[float] = []
    for _ in range(10):
        before = body.angle
        body.moment = float("inf") if config.hold_orientation else body.moment
        apply_macro_action(scenario.space, body, "E", config)
        sampled.append(abs(math.degrees(body.angle - before)))
    assert max(sampled) == pytest.approx(0.0, abs=1e-9)
    # and it is genuinely blocked, not squeezed through, at an angle that does not fit
    assert read_state(scenario.space, body).com_x < 4.0
