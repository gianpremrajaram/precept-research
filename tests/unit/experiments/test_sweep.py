from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from preceptx.agents.prompts import GATE_FEEDBACK_VERSION
from preceptx.config import ModelConfig
from preceptx.experiments.sweep import (
    SweepConfig,
    build_sweep_manifest,
    dataset_hash_for,
    episode_id,
    expand,
    sweep_hash,
)
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


def test_dataset_identity_moves_with_the_prompt_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # The sweep config carries no prompt version, so without this a prompt bump would resume into
    # the previous prompt's dataset and silently pool episodes generated under two prompt surfaces.
    sweep = _sweep()
    before = dataset_hash_for(sweep)
    monkeypatch.setattr("preceptx.experiments.sweep.PROMPT_VERSION", "v99")
    assert dataset_hash_for(sweep) != before
    assert sweep_hash(sweep) == sweep_hash(sweep)  # the config hash itself is unmoved


def test_sweep_hash_ignores_concurrency() -> None:
    # Concurrency is an execution knob, not a result-shaping one: a resumed run that changes worker
    # count must keep writing into the same dataset, not orphan every completed episode.
    assert sweep_hash(_sweep(concurrency=1)) == sweep_hash(_sweep(concurrency=8))
    assert dataset_hash_for(_sweep(concurrency=1)) == dataset_hash_for(_sweep(concurrency=8))


def test_manifest_records_the_gate_feedback_version() -> None:
    """DSE-045: the retry template is part of the RQ3b treatment, so a run that used one must say
    which one. It is recorded but deliberately absent from the dataset hash until DSE-018 lands."""
    manifest = build_sweep_manifest(_sweep(), dataset_hash="d", prompt_version="v4")
    assert manifest.gate_feedback_version == GATE_FEEDBACK_VERSION
    assert "gate_feedback_version" in manifest.model_dump(mode="json")


def test_gate_feedback_version_does_not_re_key_the_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    before = dataset_hash_for(_sweep())
    monkeypatch.setattr("preceptx.experiments.sweep.GATE_FEEDBACK_VERSION", "v99")
    assert dataset_hash_for(_sweep()) == before
