"""Typed experiment config: Hydra composes the YAML tree, Pydantic validates it here.

A single ``experiment`` config selects one cell of condition x serialisation x difficulty x model
x seed. The resolved Hydra ``DictConfig`` is validated into an ``ExperimentConfig`` at every entry
point via ``load_config``; experiment code consumes the validated model, never a raw ``DictConfig``.
Invalid combinations - an unknown condition, an unpinned model revision - fail loud as a
``ConfigError`` with a clear message, because a run built from a bad config is worse than no run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from preceptx.data.schema import Condition, Difficulty, Serialisation


class ConfigError(ValueError):
    """A config failed validation or composed into an invalid combination."""


class ModelConfig(BaseModel):
    """Identity of the served model. Revision is mandatory: an unpinned run is not a result."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    tier: str = Field(min_length=1)


class ExperimentConfig(BaseModel):
    """One factorial cell of the RQ1 sweep, validated and ready to run."""

    model_config = ConfigDict(extra="forbid")

    condition: Condition
    serialisation: Serialisation
    difficulty: Difficulty
    model: ModelConfig
    seed: int = Field(ge=0)


def load_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a resolved config mapping (from ``OmegaConf.to_container``) into the typed config.

    Wraps Pydantic's ``ValidationError`` in ``ConfigError`` so callers catch one named exception and
    get a clear, composed message naming every offending field.
    """
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid experiment config: {exc}") from exc
