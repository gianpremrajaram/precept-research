from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preceptx.agents.prompts import _SYSTEM_A, _SYSTEM_B
from preceptx.sim.actions import BodyState
from preceptx.sim.arena import ArenaGeometry, Goal, usable_gap
from preceptx.sim.load import BAR_LEN, BAR_THICK, COG_Y, extent_y
from preceptx.sim.serialise import (
    GRID_LEGEND,
    GridConfig,
    SceneState,
    deserialise_check,
    history_line,
    serialise,
    split_grid,
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
    rows = grid.splitlines()[1:]  # skip the constant legend header
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


def test_grid_draws_the_channel_body_not_just_its_faces() -> None:
    """The raster must not show free space where 1.5 world-units of wall stand (v6).

    DSE-058 made each internal wall a channel spanning `wall_depth` in x, and `_is_wall` still drew
    a one-cell stripe at the centre - so the grid arm was handed a thin threshold while the physics
    imposed a deep passage. `build_arena` seals the strip between the faces with the cap segments,
    making it an enclosed void the load can never enter, so `#` is the correct occupancy for it.
    """
    scene = _scene(2.0, 3.0 + COG_Y, angle=0.0)
    grid = serialise(scene, "grid")
    for x in (3.4, 4.0, 4.6):  # near face, centre, far face - all inside the 1.5-deep channel
        assert _char_at(grid, x, 1.0) == "#", x
    assert _char_at(grid, 2.8, 1.0) == "."  # clear of the channel on the chamber-one side
    assert _char_at(grid, 4.6, 3.0) == "."  # the aperture band stays open across the whole depth

    thin = SceneState(
        load=scene.load,
        geometry=ArenaGeometry(wall_depth=0.0),
        goal=scene.goal,
        slit_width=scene.slit_width,
    )
    # The legacy thin-segment arena must still render as it did, or the falsified T task stops
    # being reproducible from source.
    assert _char_at(serialise(thin, "grid"), 4.6, 1.0) == "."


def test_grid_draws_the_active_slit_width() -> None:
    # The hard slit (0.5) leaves a narrower gap than the easy slit (1.2) in the internal wall.
    def gap_cells(slit: float) -> int:
        grid = serialise(_scene(2.0, 3.0 + COG_Y, slit=slit), "grid").splitlines()[1:]
        col = int(ArenaGeometry().chamber_w / _CELL)
        return sum(1 for row in grid if row[col] != "#")

    assert gap_cells(0.5) < gap_cells(1.2)


def test_grid_carries_a_constant_legend_header() -> None:
    # P1-5: B must not act on an unexplained ASCII matrix; constant text across cells keeps the
    # serialisation A/B about representation, not legend absence.
    a = serialise(_scene(2.0, 3.0 + COG_Y), "grid")
    b = serialise(_scene(6.0, 4.0, angle=1.0), "grid")
    assert a.splitlines()[0] == GRID_LEGEND
    assert b.splitlines()[0] == GRID_LEGEND
    assert "T=load" in GRID_LEGEND and "G=goal" in GRID_LEGEND


def test_numeric_names_the_load_size_alongside_the_slit() -> None:
    # v4: naming the gap without naming the object leaves "aligned with the slit" underdetermined -
    # the threading band is +/-0.25 about the centre for a 1.8 gap and a 1.3-tall load, not +/-0.9.
    text = serialise(_scene(2.0, 3.0, slit=1.8), "numeric")
    assert "load_size=(1.4000, 0.3000)" in text
    assert "slit_y=(2.1000, 3.9000)" in text  # the gap is still stated as its own interval


def test_numeric_and_nl_name_the_channel_depth() -> None:
    """Both prose forms named one x per wall while the wall spanned 1.5 units of it (v6).

    PREREGISTRATION SS2 claims the three serialisations are information-isomorphic. Depth is what
    makes orientation binding on the successor task, so a form that omits it is not carrying the
    same information as the grid, which draws it.
    """
    scene = _scene(2.0, 3.0 + COG_Y)
    assert "wall_depth=1.5000" in serialise(scene, "numeric")
    assert "1.50-deep channel" in serialise(scene, "nl")

    thin = SceneState(
        load=scene.load,
        geometry=ArenaGeometry(wall_depth=0.0),
        goal=scene.goal,
        slit_width=scene.slit_width,
    )
    assert "wall_depth" not in serialise(thin, "numeric")  # no channel to name
    assert "channel" not in serialise(thin, "nl")


def test_no_prompt_surface_calls_the_load_a_t() -> None:
    """The load is a convex bar; every sentence that still called it a T was factually wrong (v6).

    This is the defect DSE-057 was spent falsifying, in its purest form: a prompt that is grounded
    in the numbers and wrong about the object. The grid legend keeps a literal "T" deliberately -
    it DEFINES the glyph as "T=load", so it is a symbol rather than a shape claim - and is excluded.
    """
    scene = _scene(2.0, 3.0 + COG_Y)
    surfaces = [serialise(scene, "numeric"), serialise(scene, "nl"), _SYSTEM_A, _SYSTEM_B]
    surfaces += [line for line in serialise(scene, "grid").splitlines() if "legend:" not in line]
    for text in surfaces:
        assert "T-load" not in text
        assert "T-shaped" not in text


def test_numeric_has_no_dead_vel_line() -> None:
    # RD-7: quasi-static settling zeroes velocity before every read; the line carried no signal.
    text = serialise(_scene(2.0, 3.0), "numeric")
    assert "vel=" not in text
    assert "load=" in text and "goal=" in text and "contact=" in text


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


def test_history_line_is_empty_before_any_action() -> None:
    line = history_line([])
    assert line.startswith("recent=()")
    assert "no actions taken yet" in line


def test_history_line_names_each_action_and_its_gain() -> None:
    line = history_line([("N", 0.34), ("E", 1.01)])
    assert "(N, +0.34)" in line and "(E, +1.01)" in line
    assert "net +1.35 over the last 2" in line


def test_history_line_surfaces_a_period_two_limit_cycle_as_zero_net() -> None:
    """The v5 line exists to make N,S,N,S visible as motion that gains nothing."""
    line = history_line([("N", 0.34), ("S", -0.34), ("N", 0.34), ("S", -0.34)])
    assert "net +0.00 over the last 4" in line


def test_history_line_carries_no_advice() -> None:
    """It reports fact, never instruction: a directive would make the v5 retune a behavioural
    change rather than an observability one, and the two are inseparable after the fact."""
    line = history_line([("ROT+", 0.0), ("ROT-", 0.0)])
    for word in ("try", "should", "instead", "different", "not working"):
        assert word not in line.lower()


# --- v8: the pose-dependent clearance line (D26) ----------------------------


def test_extent_y_is_the_projection_not_either_constant() -> None:
    # The failure v8 exists to remove: agents substituted BAR_THICK for the projection at poses
    # where the bar was near-vertical and 4.7x wider than that.
    assert extent_y(0.0) == pytest.approx(BAR_THICK)
    assert extent_y(math.pi / 2.0) == pytest.approx(BAR_LEN)
    assert extent_y(math.pi) == pytest.approx(BAR_THICK)  # a half-turn is the same span
    vertical = extent_y(math.radians(98.8))  # the modal pose of run 232980
    assert vertical == pytest.approx(1.4293, abs=1e-3)
    assert vertical > 4.0 * BAR_THICK


@pytest.mark.parametrize("mode", ["numeric", "grid", "nl"])
def test_every_form_carries_both_the_span_and_the_usable_clearance(mode: str) -> None:
    # Isomorphism: withholding either half from one form would make that form measure trigonometry
    # rather than representation, which is the axis the serialisation A/B is for. Both halves,
    # because the span alone is only actionable against a clearance - and the nl form named no
    # aperture at all before v9, so v8's sentence there invited a comparison it could not perform.
    scene = _scene(2.0, 3.0, angle=math.radians(98.8))
    text = serialise(scene, mode)  # type: ignore[arg-type]
    assert "1.429" in text
    assert f"{usable_gap(scene.slit_width, scene.geometry):.4f}" in text


def test_the_stated_clearance_is_the_one_the_walls_impose_not_the_declared_width() -> None:
    # The v9 correction. The wall faces carry a `wall_radius` lip, so the free gap is 2 x that
    # narrower than `slit_width`; v8 stated the span beside the declared width and so certified as
    # passable poses that jam - measured, the true limit is 38.0/17.0/10.0 deg at 1.20/0.80/0.64.
    geo = ArenaGeometry()
    for width in (1.20, 0.80, 0.64):
        scene = _scene(2.0, 3.0, slit=width)
        assert usable_gap(width, geo) == pytest.approx(width - 2.0 * geo.wall_radius)
        assert f"slit_clearance={width - 2.0 * geo.wall_radius:.4f}" in serialise(scene, "numeric")
        # The looser number must not be the one the line offers as the bar to clear.
        assert f"slit_clearance={width:.4f}" not in serialise(scene, "numeric")


def test_grid_splits_header_from_raster_by_alphabet() -> None:
    header, rows = split_grid(serialise(_scene(6.0, 3.0), "grid"))
    assert header[0] == GRID_LEGEND
    assert any("load_extent_y=" in line for line in header)
    assert any("slit_clearance=" in line for line in header)
    assert rows and all(set(row) <= set("TG#.") for row in rows)
    assert any("T" in row for row in rows)  # the load is inside the raster, not the header
