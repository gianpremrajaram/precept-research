"""The T-shaped dynamic load: one rigid body carrying two box shapes forming a T.

The T's long axis must rotate to clear a narrow slit (DSE-006); the geometry below is constructed
so the body's vertical extent is symmetric about its position, i.e. placing the body at a slit's
y-centre centres the load on the gap. Dimensions are module constants (the task uses one load).
"""

from __future__ import annotations

import pymunk

# T geometry (world units). Bar across the top, stem hanging below; see module docstring.
T_THICK = 0.3  # bar/stem thickness
T_BAR = 1.4  # bar length (the load's x-extent at angle 0)
T_STEM = 1.0  # stem length
T_FRICTION = 0.6

# Half the total vertical extent; bar and stem are placed so the body is symmetric in y about 0.
HALF_H = (T_THICK + T_STEM) / 2.0

# Local-frame y of the area centroid (= centre of gravity for uniform density). The bar sits above
# the stem, so the COG is offset from the body origin in +y; serialisers need it to place the
# footprint from a COM-only read-back (BodyState reports the COM, not the body origin).
_BAR_CY = HALF_H - T_THICK / 2.0
_STEM_CY = HALF_H - T_THICK - T_STEM / 2.0
_AREA_BAR = T_BAR * T_THICK
_AREA_STEM = T_STEM * T_THICK
COG_Y = (_AREA_BAR * _BAR_CY + _AREA_STEM * _STEM_CY) / (_AREA_BAR + _AREA_STEM)

Vert = tuple[float, float]


def _box_verts(cx: float, cy: float, w: float, h: float) -> list[Vert]:
    """Four corners of an axis-aligned box centred at ``(cx, cy)``."""
    hw, hh = w / 2.0, h / 2.0
    return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]


def _t_shape_verts() -> tuple[list[Vert], list[Vert]]:
    """Local-frame vertices for the bar (top) and stem (below), symmetric in y about 0."""
    bar = _box_verts(0.0, HALF_H - T_THICK / 2.0, T_BAR, T_THICK)
    stem = _box_verts(0.0, HALF_H - T_THICK - T_STEM / 2.0, T_THICK, T_STEM)
    return bar, stem


def add_t_load(space: pymunk.Space, pos: tuple[float, float], mass: float) -> pymunk.Body:
    """Add a dynamic T-load at ``pos``; mass is split by area and moment summed over both boxes."""
    bar, stem = _t_shape_verts()
    area_bar, area_stem = T_BAR * T_THICK, T_STEM * T_THICK
    area = area_bar + area_stem
    m_bar, m_stem = mass * area_bar / area, mass * area_stem / area
    moment = pymunk.moment_for_poly(m_bar, bar) + pymunk.moment_for_poly(m_stem, stem)

    body = pymunk.Body(mass, moment)
    body.position = pos
    bar_shape, stem_shape = pymunk.Poly(body, bar), pymunk.Poly(body, stem)
    bar_shape.friction = stem_shape.friction = T_FRICTION
    space.add(body, bar_shape, stem_shape)
    return body


def point_in_t_local(lx: float, ly: float) -> bool:
    """Whether a point in the load's local frame lies inside the T footprint (the bar or stem box).

    The canonical T geometry lives here, so the grid serialiser rasterises against this rather than
    re-deriving the box bounds; it matches the boxes ``_t_shape_verts`` builds.
    """
    in_bar = abs(lx) <= T_BAR / 2.0 and HALF_H - T_THICK <= ly <= HALF_H
    in_stem = abs(lx) <= T_THICK / 2.0 and -HALF_H <= ly <= HALF_H - T_THICK
    return in_bar or in_stem
