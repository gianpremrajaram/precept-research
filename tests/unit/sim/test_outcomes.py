from __future__ import annotations

import math

from preceptx.data.schema import HandoffRecord, StatePayload
from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal
from preceptx.sim.outcomes import (
    OutcomeConfig,
    geodesic_distance,
    label_episode,
    reached_goal,
    step_progress,
)

_GEO = ArenaGeometry()
_GOAL = Goal(center_x=10.0, center_y=3.0, radius=0.8)


def _state(x: float, y: float) -> BodyState:
    return BodyState(com_x=x, com_y=y, angle=0.0, vx=0.0, vy=0.0, omega=0.0, in_contact=False)


def _payload(x: float, y: float) -> StatePayload:
    return _state(x, y).model_dump()


def _record(
    step: int, pre: tuple[float, float], post: tuple[float, float], success: bool
) -> HandoffRecord:
    return HandoffRecord(
        episode_id="e0",
        step=step,
        condition="C0",
        serialisation="numeric",
        difficulty="hard",
        model="stub",
        seed=0,
        state=_payload(*pre),
        state_str="",
        message_raw="",
        message_delivered="",
        action={},
        pre_state=_payload(*pre),
        post_state=_payload(*post),
        progress=0.0,
        success=success,
        collision=False,
        stuck=False,
    )


def test_geodesic_decreases_along_solving_trajectory() -> None:
    coms = [(2.0, 3.0), (4.0, 3.0), (8.0, 3.0), (10.0, 3.0)]
    gds = [geodesic_distance(c, _GOAL, _GEO) for c in coms]
    assert all(gds[i] > gds[i + 1] for i in range(len(gds) - 1))


def test_geodesic_increases_when_pushed_away() -> None:
    assert geodesic_distance((2.0, 3.0), _GOAL, _GEO) > geodesic_distance((4.0, 3.0), _GOAL, _GEO)


def test_geodesic_routes_through_slits_not_straight_through_walls() -> None:
    # An off-centre start must detour up to the slit centre, so routing beats the straight line.
    routed = geodesic_distance((2.0, 1.0), _GOAL, _GEO)
    straight = math.dist((2.0, 1.0), (_GOAL.center_x, _GOAL.center_y))
    assert routed > straight


def test_reached_goal_only_in_region() -> None:
    assert reached_goal(_state(10.0, 3.0), _GOAL)
    assert reached_goal(_state(10.5, 3.0), _GOAL)  # within radius 0.8
    assert not reached_goal(_state(10.9, 3.0), _GOAL)  # 0.9 > 0.8
    assert not reached_goal(_state(2.0, 3.0), _GOAL)


def test_step_progress_sign() -> None:
    assert step_progress(_state(2.0, 3.0), _state(3.0, 3.0), _GOAL, _GEO) > 0
    assert step_progress(_state(3.0, 3.0), _state(2.0, 3.0), _GOAL, _GEO) < 0


def test_label_episode_four_labels_on_solving_episode() -> None:
    records = [
        _record(0, (2.0, 3.0), (4.0, 3.0), success=False),
        _record(1, (4.0, 3.0), (8.0, 3.0), success=False),
        _record(2, (8.0, 3.0), (10.0, 3.0), success=True),
    ]
    out = label_episode(records, _GOAL, _GEO, OutcomeConfig(k=1))
    assert [r.y_binary_progress for r in out] == [True, True, True]
    assert all((r.y_continuous_displacement or 0.0) > 0 for r in out)
    assert [r.y_discrete_config for r in out] == [1, 2, 3]
    assert [r.y_terminal_success for r in out] == [True, True, True]  # success at the last step


def test_label_episode_terminal_false_and_backward_progress() -> None:
    records = [_record(0, (4.0, 3.0), (2.0, 3.0), success=False)]  # pushed away, never succeeds
    out = label_episode(records, _GOAL, _GEO, OutcomeConfig(k=1))
    assert out[0].y_binary_progress is False
    assert (out[0].y_continuous_displacement or 0.0) < 0
    assert out[0].y_terminal_success is False


def test_label_episode_populates_every_label() -> None:
    records = [_record(0, (2.0, 3.0), (4.0, 3.0), success=False)]
    out = label_episode(records, _GOAL, _GEO, OutcomeConfig(k=3))
    r = out[0]
    assert None not in (
        r.y_binary_progress,
        r.y_continuous_displacement,
        r.y_discrete_config,
        r.y_terminal_success,
    )


def test_label_episode_is_deterministic() -> None:
    records = [
        _record(0, (2.0, 3.0), (4.0, 3.0), success=False),
        _record(1, (4.0, 3.0), (8.0, 3.0), success=True),
    ]
    a = label_episode(records, _GOAL, _GEO, OutcomeConfig(k=2))
    b = label_episode(records, _GOAL, _GEO, OutcomeConfig(k=2))
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
