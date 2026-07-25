from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from preceptx.config import ModelConfig
from preceptx.experiments.sweep import SweepConfig, episode_id, expand, sweep_hash
from preceptx.sim.actions import StepConfig
from preceptx.sim.arena import ScenarioJitter
from preceptx.sim.outcomes import OutcomeConfig

MODEL = ModelConfig(name="m", revision="rev", tier="8b")


def _sweep(**overrides: Any) -> SweepConfig:
    base: dict[str, Any] = {
        "conditions": ["C0", "C4"],
        "serialisations": ["numeric"],
        "difficulties": ["easy", "hard"],
        "seeds": [1, 2, 3],
        "model": MODEL,
    }
    base.update(overrides)
    return SweepConfig(**base)


def test_expand_is_full_cartesian_product() -> None:
    cells = expand(_sweep())
    assert len(cells) == 2 * 1 * 2 * 3  # conditions x serialisations x difficulties x seeds
    assert len({episode_id(c) for c in cells}) == len(cells)  # ids unique per cell


def test_expand_cells_carry_the_axis_values() -> None:
    cells = expand(_sweep(conditions=["C2"], difficulties=["medium"], seeds=[7]))
    assert len(cells) == 1
    assert (cells[0].condition, cells[0].difficulty, cells[0].seed) == ("C2", "medium", 7)


def test_sweep_hash_is_stable_and_config_sensitive() -> None:
    assert sweep_hash(_sweep()) == sweep_hash(_sweep())  # deterministic
    assert sweep_hash(_sweep()) != sweep_hash(_sweep(seeds=[9]))  # changes with the grid


def test_sweep_hash_covers_jitter_step_and_outcome_knobs() -> None:
    # P0-2/P1-6: a silent change to the jitter region, impulse, or the label horizon k must roll
    # the hash - otherwise a re-run dataset would be silently relabelled under the same identity.
    base = sweep_hash(_sweep())
    assert sweep_hash(_sweep(jitter=ScenarioJitter(x_range=(1.5, 2.5)))) != base
    assert sweep_hash(_sweep(step=StepConfig(linear_impulse=4.0))) != base
    assert sweep_hash(_sweep(outcome=OutcomeConfig(k=5))) != base


def test_empty_axis_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _sweep(seeds=[])
