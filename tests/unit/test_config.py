from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hypothesis import given
from hypothesis import strategies as st
from omegaconf import OmegaConf

from preceptx.config import ConfigError, ExperimentConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _valid() -> dict[str, Any]:
    return {
        "condition": "C0",
        "serialisation": "numeric",
        "difficulty": "easy",
        "model": {"name": "Qwen/Qwen3-14B-Instruct", "revision": "abc123", "tier": "14b"},
        "seed": 0,
    }


def test_load_config_produces_typed_object() -> None:
    config = load_config(_valid())
    assert isinstance(config, ExperimentConfig)
    assert config.condition == "C0"
    assert config.model.tier == "14b"


def test_hydra_tree_composes_and_validates() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="experiment",
            overrides=["condition=C3", "serialisation=grid", "model=qwen8b", "seed=5"],
        )
    config = load_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    assert config.condition == "C3"
    assert config.serialisation == "grid"
    assert config.model.name == "Qwen/Qwen3-8B-Instruct"
    assert config.seed == 5


def test_invalid_condition_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(_valid() | {"condition": "C9"})


def test_unpinned_revision_rejected() -> None:
    bad = _valid()
    bad["model"] = {"name": "m", "revision": "", "tier": "14b"}
    with pytest.raises(ConfigError):
        load_config(bad)


def test_extra_field_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(_valid() | {"surprise": 1})


def test_negative_seed_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(_valid() | {"seed": -1})


@given(
    condition=st.sampled_from(["C0", "C1", "C2", "C3", "C4"]),
    serialisation=st.sampled_from(["numeric", "grid", "nl"]),
    difficulty=st.sampled_from(["easy", "medium", "hard"]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_valid_combinations_always_load(
    condition: str, serialisation: str, difficulty: str, seed: int
) -> None:
    config = load_config(
        _valid()
        | {
            "condition": condition,
            "serialisation": serialisation,
            "difficulty": difficulty,
            "seed": seed,
        }
    )
    assert config.condition == condition
    assert config.seed == seed
