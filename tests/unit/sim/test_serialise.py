from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal
from preceptx.sim.load import COG_Y
from preceptx.sim.serialise import (
    GridConfig,
    SceneState,
    deserialise_check,
    serialise,
)

_CELL = GridConfig().cell


def _scene(
    com_x: float, com_y: float, *, angle: float = 0.0, in_contact: bool = False, slit: float = 1.0
) -> SceneState:
    load = BodyState(
        com_x=com_x, com_y=com_y, angle=angle, vx=0.0, vy=0.0, omega=0.0, in_contact=in_contact
    )
    return SceneState(
        load=load,
        geometry=ArenaGeometry(),
        goal=Goal(center_x=10.0, center_y=3.0, radius=0.8),
        slit_width=slit,
    )


def _char_at(grid: str, x: float, y: float) -> str:
    """Char of the grid cell containing world point (x, y)."""
    rows = grid.splitlines()
    return rows[len(rows) - 1 - int(y / _CELL)][int(x / _CELL)]


def test_each_mode_is_deterministic() -> None:
    scene = _scene(2.0, 3.0 + COG_Y, angle=0.3)
    for mode in ("numeric", "grid", "nl"):
        assert serialise(scene, mode) == serialise(scene, mode)  # type: ignore[arg-type]


def test_numeric_round_trips_com_and_angle() -> None:
    scene = _scene(2.5, 3.5, angle=0.42)
    assert deserialise_check(scene, "numeric")


def test_grid_occupancy_on_known_pose() -> None:
    # Body origin at (2, 3): the COM read-back is offset by the COG in +y.
    scene = _scene(2.0, 3.0 + COG_Y, angle=0.0)
    grid = serialise(scene, "grid")
    assert _char_at(grid, 2.0, 3.2) == "T"  # stem, just above the origin
    assert _char_at(grid, 10.0, 3.0) == "G"  # goal centre
    assert _char_at(grid, 4.0, 1.0) == "#"  # internal wall below the slit
    assert _char_at(grid, 1.0, 5.0) == "."  # open chamber-one cell


def test_grid_draws_the_active_slit_width() -> None:
    # The hard slit (0.7) leaves a narrower gap than the easy slit (1.8) in the internal wall.
    def gap_cells(slit: float) -> int:
        grid = serialise(_scene(2.0, 3.0 + COG_Y, slit=slit), "grid").splitlines()
        col = int(ArenaGeometry().chamber_w / _CELL)
        return sum(1 for row in grid if row[col] != "#")

    assert gap_cells(0.7) < gap_cells(1.8)


def test_grid_deserialise_recovers_com() -> None:
    assert deserialise_check(_scene(6.0, 3.0), "grid")
    assert deserialise_check(_scene(6.0, 3.0, angle=math.pi / 2.0), "grid")  # rotated pose


def test_nl_is_templated_and_mentions_chamber_and_goal() -> None:
    text = serialise(_scene(2.0, 3.0), "nl")
    assert "chamber 1" in text
    assert "(10.00, 3.00)" in text  # goal coordinates are present (hybrid qual+quant)
    assert "rad" in text


def test_nl_deserialise_check_raises() -> None:
    with pytest.raises(ValueError, match="numeric/grid only"):
        deserialise_check(_scene(2.0, 3.0), "nl")


@settings(max_examples=60, deadline=None)
@given(
    com_x=st.floats(-1.0, 13.0, allow_nan=False, allow_infinity=False),
    com_y=st.floats(-1.0, 7.0, allow_nan=False, allow_infinity=False),
    angle=st.floats(-math.pi, math.pi, allow_nan=False, allow_infinity=False),
    slit=st.sampled_from([0.7, 1.0, 1.8]),
)
def test_serialisers_never_raise_on_valid_or_extreme_poses(
    com_x: float, com_y: float, angle: float, slit: float
) -> None:
    scene = _scene(com_x, com_y, angle=angle, slit=slit)
    for mode in ("numeric", "grid", "nl"):
        assert isinstance(serialise(scene, mode), str)  # type: ignore[arg-type]
    assert isinstance(deserialise_check(scene, "numeric"), bool)
    assert isinstance(deserialise_check(scene, "grid"), bool)
