"""The dynamic load. Two shapes live here: the active convex bar, and the legacy T.

**The bar is the active load (DSE-058).** It is the *successor* task's body, adopted after the
T-load benchmark was falsified as a rotation-necessity manipulation: because the T is non-convex,
its collision-free configuration space through a gap is not characterised by a bounding y-extent,
so the bar and the stem cross at different instants and a translation-only path exists at every
slit width. A convex body has no such escape - fit is governed by extent - which is what makes an
explicit rotation necessary. See `docs/experiment_design_log.md` (2026-08-27) for the falsification.

**The T is retained deliberately**, not as dead code: it is the subject of a published finding, and
`scripts/check_rotation_need.py` plus the unit tests reproduce that falsification from source. Do
not delete it without also retiring the finding.

Both bodies are symmetric in y about the body origin, so placing the body at a slit's y-centre
centres the load on the gap. Dimensions are module constants (the task uses one load).
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

# Convex bar geometry (world units) - the ACTIVE load. Same footprint as the T's crossbar, so the
# arena scale and the grip span are unchanged; what changes is that there is no second member to
# cross the gap independently. Rotation-required aperture band is [BAR_THICK, BAR_LEN) = [0.3, 1.4),
# against the T's empty band. The shipped ladder sits at 1.20/0.80/0.50 (see sim/arena.py).
BAR_LEN = 1.4
BAR_THICK = 0.3

# The bar is symmetric about its origin, so its centre of gravity IS the origin. Serialisers read
# back a COM and need the offset to recover the body origin; for the bar that offset is zero.
LOAD_COG_Y = 0.0
LOAD_EXTENT_X = BAR_LEN  # x-extent at angle 0
LOAD_EXTENT_Y = BAR_THICK  # y-extent at angle 0

Vert = tuple[float, float]


def _box_verts(cx: float, cy: float, w: float, h: float) -> list[Vert]:
    """Four corners of an axis-aligned box centred at ``(cx, cy)``."""
    hw, hh = w / 2.0, h / 2.0
    return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]


def t_shape_verts() -> tuple[list[Vert], list[Vert]]:
    """Local-frame vertices for the bar (top) and stem (below), symmetric in y about 0.

    Public because it is the canonical T outline: ``add_t_load`` builds the physics shapes from it
    and the episode renderer (analysis/render.py) draws the same two polygons, so the figure and the
    simulated body cannot drift. World placement is ``com + R(angle)·vert`` (the body origin equals
    the read-back COM, since ``center_of_gravity`` is the origin).
    """
    bar = _box_verts(0.0, HALF_H - T_THICK / 2.0, T_BAR, T_THICK)
    stem = _box_verts(0.0, HALF_H - T_THICK - T_STEM / 2.0, T_THICK, T_STEM)
    return bar, stem


def bar_shape_verts() -> list[Vert]:
    """Local-frame vertices of the convex bar, centred on the body origin."""
    return _box_verts(0.0, 0.0, BAR_LEN, BAR_THICK)


def load_polys() -> list[list[Vert]]:
    """Every polygon of the ACTIVE load, for renderers and any outline consumer.

    A list because the legacy T needed two; the bar returns one. Consumers iterate rather than
    unpack, so swapping the active shape does not ripple outward.
    """
    return [bar_shape_verts()]


def add_load(space: pymunk.Space, pos: tuple[float, float], mass: float) -> pymunk.Body:
    """Add the ACTIVE load (the convex bar) at ``pos`` as a single dynamic polygon."""
    verts = bar_shape_verts()
    body = pymunk.Body(mass, pymunk.moment_for_poly(mass, verts))
    body.position = pos
    shape = pymunk.Poly(body, verts)
    shape.friction = T_FRICTION
    space.add(body, shape)
    return body


def point_in_load_local(lx: float, ly: float) -> bool:
    """Whether a local-frame point lies inside the ACTIVE load's footprint (the grid serialiser)."""
    return abs(lx) <= BAR_LEN / 2.0 and abs(ly) <= BAR_THICK / 2.0


def add_t_load(space: pymunk.Space, pos: tuple[float, float], mass: float) -> pymunk.Body:
    """Add the dynamic load at ``pos``; mass is split by area and moment summed over its boxes."""
    bar, stem = t_shape_verts()
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
