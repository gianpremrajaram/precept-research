"""Sweep configuration, grid expansion, run summary, and the sweep manifest (DSE-012).

``SweepConfig`` lists the RQ1 grid axes (condition x serialisation x difficulty x seed) plus the
fixed model, channel, and step budget; ``expand`` takes their Cartesian product into validated
``ExperimentConfig`` cells - one episode per cell, with replication carried by the seed axis (greedy
decoding plus deterministic physics make repeated identical cells pointless). ``SweepManifest`` is
the run-level reproducibility record for a grid (the per-cell ``RunManifest`` in ``manifest.py``
models a single cell); it reuses the git/dep capture there and carries the resolved sweep, its hash,
the prompt version, and the run summary.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import sys

from pydantic import BaseModel, ConfigDict, Field

from preceptx.agents.channel import ChannelConfig
from preceptx.config import ExperimentConfig, ModelConfig
from preceptx.data.schema import Condition, Difficulty, Serialisation
from preceptx.manifest import dep_versions, git_sha

SWEEP_MANIFEST_VERSION = 1


class SweepConfig(BaseModel):
    """The RQ1-style grid: axes as lists, plus the fixed model / channel / step budget."""

    model_config = ConfigDict(extra="forbid")

    conditions: list[Condition] = Field(min_length=1)
    serialisations: list[Serialisation] = Field(min_length=1)
    difficulties: list[Difficulty] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    model: ModelConfig
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    max_steps: int = Field(default=12, gt=0)
    concurrency: int = Field(default=4, gt=0)


class RunSummary(BaseModel):
    """Per-run rollup: cells, episodes, handoffs, success rate, wall time."""

    model_config = ConfigDict(extra="forbid")

    n_cells: int
    n_episodes: int
    n_handoffs: int
    success_rate: float
    wall_time_s: float


class SweepManifest(BaseModel):
    """Run-level reproducibility record for a grid sweep."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = SWEEP_MANIFEST_VERSION
    git_sha: str
    sweep: SweepConfig
    sweep_hash: str
    dataset_hash: str
    model_name: str
    model_revision: str
    prompt_version: str
    command: list[str]
    dep_versions: dict[str, str]
    timestamp: str
    summary: RunSummary | None = None


def expand(sweep: SweepConfig) -> list[ExperimentConfig]:
    """Cartesian product of the grid axes into single-cell configs (seed = replication)."""
    return [
        ExperimentConfig(
            condition=cond, serialisation=ser, difficulty=diff, model=sweep.model, seed=seed
        )
        for cond, ser, diff, seed in itertools.product(
            sweep.conditions, sweep.serialisations, sweep.difficulties, sweep.seeds
        )
    ]


def episode_id(cell: ExperimentConfig) -> str:
    """Stable, unique id per cell - the resumability key (idempotent on re-run)."""
    return f"{cell.condition}-{cell.serialisation}-{cell.difficulty}-{cell.model.name}-s{cell.seed}"


def sweep_hash(sweep: SweepConfig) -> str:
    """Content hash of the resolved sweep config (sorted-key JSON, sha256, 16 hex)."""
    canonical = json.dumps(sweep.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_sweep_manifest(
    sweep: SweepConfig, *, dataset_hash: str, prompt_version: str
) -> SweepManifest:
    """Assemble the run-level manifest from the sweep plus the live environment."""
    return SweepManifest(
        git_sha=git_sha(),
        sweep=sweep,
        sweep_hash=sweep_hash(sweep),
        dataset_hash=dataset_hash,
        model_name=sweep.model.name,
        model_revision=sweep.model.revision,
        prompt_version=prompt_version,
        command=list(sys.argv),
        dep_versions=dep_versions(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
    )
