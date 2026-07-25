from __future__ import annotations

import math

import numpy as np
import pymunk

from preceptx.sim.arena import (
    ArenaGeometry,
    Scenario,
    ScenarioJitter,
    _overlaps_other,
    build_arena,
    make_scenario,
)


def _internal_segments(space: pymunk.Space, x: float) -> list[pymunk.Segment]:
    segs = [s for s in space.static_body.shapes if isinstance(s, pymunk.Segment)]
    return [s for s in segs if s.a.x == x and s.b.x == x]


# DSE-006 tests drive pymunk.step directly (not the DSE-007 API) to keep the tickets separate.
def _push_east(scenario: Scenario, nudges: int = 10) -> float:
    """Apply repeated head-on +x nudges with raw stepping; return the furthest COM x reached."""
    body = scenario.load
    max_x = float(body.position.x)
    for _ in range(nudges):
        com = body.local_to_world(body.center_of_gravity)
        body.apply_impulse_at_world_point((3.0, 0.0), com)
        for _ in range(30):
            for _ in range(4):
                scenario.space.step(1.0 / 60.0 / 4.0)
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
        max_x = max(max_x, float(body.position.x))
    return max_x


def test_build_arena_has_outer_and_two_split_internal_walls() -> None:
    geo = ArenaGeometry()
    space = build_arena(0.7, geo)
    segs = [s for s in space.static_body.shapes if isinstance(s, pymunk.Segment)]
    assert len(segs) == 8  # 4 outer + 2 internal walls each split into 2 around the slit

    half = 0.7 / 2.0
    for x in (geo.chamber_w, 2.0 * geo.chamber_w):
        pair = _internal_segments(space, x)
        assert len(pair) == 2
        tops = sorted(max(s.a.y, s.b.y) for s in pair)
        bots = sorted(min(s.a.y, s.b.y) for s in pair)
        assert bots[0] == 0.0 and tops[0] == geo.slit_y - half  # lower segment up to the gap
        assert bots[1] == geo.slit_y + half and tops[1] == geo.chamber_h  # upper segment past gap


def test_goal_is_in_chamber_three() -> None:
    geo = ArenaGeometry()
    goal = make_scenario("easy").goal
    assert 2.0 * geo.chamber_w < goal.center_x < 3.0 * geo.chamber_w


def test_make_scenario_reconstructs_identically() -> None:
    a, b = make_scenario("hard"), make_scenario("hard")
    assert tuple(a.load.position) == tuple(b.load.position)
    assert a.load.angle == b.load.angle
    assert a.goal == b.goal
    va = [tuple(v) for s in a.space.shapes if s.body is a.load for v in s.get_vertices()]
    vb = [tuple(v) for s in b.space.shapes if s.body is b.load for v in s.get_vertices()]
    assert va == vb


def test_easy_slit_load_passes_under_nudge() -> None:
    # Wide slit: a straight head-on nudge clears the first internal wall (x = chamber_w).
    scenario = make_scenario("easy")
    assert _push_east(scenario) > ArenaGeometry().chamber_w


def test_hard_slit_load_jams_without_rotation() -> None:
    # Narrow slit: the same head-on nudge jams; the load never crosses the wall without rotating.
    scenario = make_scenario("hard")
    assert _push_east(scenario) < ArenaGeometry().chamber_w


# P0-2 jitter: the seed axis must be true replication - same seed reproduces the same problem
# instance, different seeds produce different ones, and every jittered pose is legal.


def _jitter_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 2**16])  # mirrors the EpisodeRunner spawn key


def test_jitter_same_seed_reproduces_identical_pose() -> None:
    a = make_scenario("hard", rng=_jitter_rng(7))
    b = make_scenario("hard", rng=_jitter_rng(7))
    assert tuple(a.load.position) == tuple(b.load.position)
    assert a.load.angle == b.load.angle


def test_jitter_different_seeds_give_different_poses() -> None:
    poses = {
        (round(s.load.position.x, 6), round(s.load.position.y, 6), round(s.load.angle, 6))
        for s in (make_scenario("hard", rng=_jitter_rng(seed)) for seed in range(5))
    }
    assert len(poses) == 5  # five seeds, five distinct problem instances


def test_jitter_pose_within_bounds_and_collision_free() -> None:
    jit = ScenarioJitter()
    for seed in range(10):
        s = make_scenario("hard", rng=_jitter_rng(seed), jitter=jit)
        assert jit.x_range[0] <= s.load.position.x <= jit.x_range[1]
        assert jit.y_range[0] <= s.load.position.y <= jit.y_range[1]
        assert jit.theta_range[0] <= s.load.angle <= jit.theta_range[1]
        assert not _overlaps_other(s.space, s.load)  # never spawned into a wall


def test_jitter_zero_width_ranges_recover_a_fixed_pose() -> None:
    jit = ScenarioJitter(x_range=(2.0, 2.0), y_range=(3.0, 3.0), theta_range=(0.0, 0.0))
    s = make_scenario("hard", rng=_jitter_rng(3), jitter=jit)
    assert tuple(s.load.position) == (2.0, 3.0)
    assert s.load.angle == 0.0


def test_jitter_goal_stays_fixed() -> None:
    # Only the pose varies; geometry and goal are the difficulty knobs and stay put.
    assert make_scenario("hard", rng=_jitter_rng(1)).goal == make_scenario("hard").goal
    assert math.isclose(make_scenario("hard", rng=_jitter_rng(2)).goal.center_x, 10.0)
