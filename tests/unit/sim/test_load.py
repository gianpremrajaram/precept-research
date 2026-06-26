from __future__ import annotations

import pymunk
import pytest

from preceptx.sim.load import HALF_H, T_FRICTION, add_t_load


def _space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    return space


def test_add_t_load_mass_moment_and_shapes() -> None:
    space = _space()
    body = add_t_load(space, (2.0, 3.0), 1.0)
    shapes = [s for s in space.shapes if s.body is body]
    assert body.mass == 1.0
    assert body.moment > 0.0
    assert len(shapes) == 2  # bar + stem
    assert all(s.friction == T_FRICTION for s in shapes)


def test_t_load_y_extent_is_symmetric_about_position() -> None:
    # The slit logic relies on the load's vertical extent being centred on the body position.
    space = _space()
    body = add_t_load(space, (0.0, 0.0), 1.0)
    ys = [v.y for s in space.shapes if s.body is body for v in s.get_vertices()]
    assert min(ys) == pytest.approx(-HALF_H)
    assert max(ys) == pytest.approx(HALF_H)
