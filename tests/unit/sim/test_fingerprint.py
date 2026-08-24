from __future__ import annotations

import json
import subprocess
import sys

import pytest

from preceptx.sim import arena, load, serialise
from preceptx.sim.fingerprint import ENVIRONMENT_SCHEMA_VERSION, simulation_fingerprint

_DIGEST_SNIPPET = (
    "from preceptx.sim.fingerprint import simulation_fingerprint;"
    "print(simulation_fingerprint().digest())"
)


def test_the_same_world_fingerprints_identically() -> None:
    assert simulation_fingerprint().digest() == simulation_fingerprint().digest()


def test_the_digest_is_stable_across_processes_and_hash_seeds() -> None:
    """The guard is worthless if the digest depends on the process.

    ``PYTHONHASHSEED`` randomises str/set hashing per interpreter, so a fingerprint that leaked
    set iteration order or an object ``repr`` would differ run to run - and every resume would
    then look like a geometry change. Two subprocesses under different seeds pin that shut.
    """
    digests = {
        subprocess.run(
            [sys.executable, "-c", _DIGEST_SNIPPET],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "12345")
    }
    assert len(digests) == 1
    assert digests == {simulation_fingerprint().digest()}


def test_the_payload_is_json_primitives_only() -> None:
    """What is hashed must also be readable: the manifest carries this payload to explain a re-key."""
    payload = json.loads(simulation_fingerprint().model_dump_json())
    assert payload["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    for group in ("slit_widths", "arena", "load", "grid"):
        assert payload[group], f"{group} is empty - a whole category would go unguarded"
        assert all(isinstance(v, (int, float)) for v in payload[group].values())


@pytest.mark.parametrize(
    ("module", "attr", "value"),
    [
        (arena, "DAMPING", 0.3),
        (arena, "LOAD_MASS", 2.0),
        (arena, "GOAL_RADIUS", 1.1),
        (arena, "WALL_FRICTION", 0.9),
        (load, "T_THICK", 0.4),
        (load, "T_BAR", 1.6),
        (load, "T_STEM", 1.2),
        (load, "T_FRICTION", 0.9),
    ],
)
def test_every_world_constant_changes_the_digest(
    monkeypatch: pytest.MonkeyPatch, module: object, attr: str, value: float
) -> None:
    before = simulation_fingerprint().digest()
    monkeypatch.setattr(module, attr, value)
    assert simulation_fingerprint().digest() != before


def test_a_slit_width_retune_changes_the_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lever the pre-registration actually pulls (PREREGISTRATION section 6, roadmap 3.1)."""
    before = simulation_fingerprint().digest()
    monkeypatch.setitem(arena._DIFFICULTY_SLITS, "hard", 1.4)
    assert simulation_fingerprint().digest() != before


class _WiderArena(arena.ArenaGeometry):
    chamber_w: float = 5.0


class _CoarserGrid(serialise.GridConfig):
    cell: float = 0.5


def test_arena_dimensions_change_the_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    before = simulation_fingerprint().digest()
    monkeypatch.setattr(arena, "ArenaGeometry", _WiderArena)
    assert simulation_fingerprint().digest() != before


def test_grid_resolution_changes_the_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grid arm's cell size is a prompt-shaping value, and its own docstring flags it as a
    pre-freeze retune candidate - so it belongs in dataset identity alongside PROMPT_VERSION."""
    before = simulation_fingerprint().digest()
    monkeypatch.setattr(serialise, "GridConfig", _CoarserGrid)
    assert simulation_fingerprint().digest() != before


def test_the_schema_version_is_an_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A behavioural change with no constant to point at still has a way to force a re-key."""
    before = simulation_fingerprint().digest()
    monkeypatch.setattr("preceptx.sim.fingerprint.ENVIRONMENT_SCHEMA_VERSION", 2)
    assert simulation_fingerprint().digest() != before


def test_slit_widths_returns_a_copy() -> None:
    widths = arena.slit_widths()
    widths["hard"] = 99.0
    assert arena.slit_widths()["hard"] != 99.0
