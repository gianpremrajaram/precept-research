"""Sweep configuration, grid expansion, run summary, and the sweep manifest (DSE-012).

``SweepConfig`` lists the RQ1 grid axes (condition x serialisation x difficulty x seed) plus the
fixed model, channel, scenario jitter, step and outcome configs, and step budget; ``expand`` takes
their Cartesian product into validated ``ExperimentConfig`` cells - one episode per cell, with
replication carried by the seed axis: the seed drives the start-pose jitter (P0-2), so different
seeds are genuinely different problem instances, not identical greedy replays. ``SweepManifest`` is
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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from preceptx.agents.channel import ChannelConfig
from preceptx.agents.prompts import GATE_FEEDBACK_VERSION, PROMPT_VERSION
from preceptx.config import ExperimentConfig, ModelConfig
from preceptx.data.schema import Condition, Difficulty, Serialisation
from preceptx.data.writer import dataset_hash
from preceptx.manifest import ServeEnv, dep_versions, git_sha, serve_env
from preceptx.serving.client import ServingConfig
from preceptx.sim.actions import StepConfig
from preceptx.sim.arena import ScenarioJitter
from preceptx.sim.feasibility import STEP_BUDGETS
from preceptx.sim.fingerprint import SimulationFingerprint, simulation_fingerprint
from preceptx.sim.outcomes import OutcomeConfig

SWEEP_MANIFEST_VERSION = 2


class SweepConfig(BaseModel):
    """The RQ1-style grid: axes as lists, plus the fixed model / channel / step budget."""

    model_config = ConfigDict(extra="forbid")

    conditions: list[Condition] = Field(min_length=1)
    serialisations: list[Serialisation] = Field(min_length=1)
    difficulties: list[Difficulty] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    model: ModelConfig
    # The optional second role (DSE-049): when set, agent B is served by this model instead of
    # `model`, and the caller must supply a matching client_b. None = self-play, the primary cell.
    model_b: ModelConfig | None = None
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    # Result-shaping knobs carried here so they reach sweep_hash and the manifest (P0-2, P1-6):
    # a silent change to the jitter region, impulse parameters, or the label horizon k would
    # otherwise relabel a re-run dataset without changing its hash.
    jitter: ScenarioJitter = Field(default_factory=ScenarioJitter)
    step: StepConfig = Field(default_factory=StepConfig)
    outcome: OutcomeConfig = Field(default_factory=OutcomeConfig)
    # Per-difficulty step budget (P1-4): each difficulty's certified feasibility budget (~2.5x the
    # oracle optimum from sim/feasibility.py), so hard is not starved relative to easy. A bare int
    # is accepted and broadcast to every difficulty (so a caller can still pass one budget).
    max_steps: dict[Difficulty, int] = Field(default_factory=lambda: dict(STEP_BUDGETS))
    concurrency: int = Field(default=4, gt=0)

    @field_validator("max_steps", mode="before")
    @classmethod
    def _broadcast_scalar_budget(cls, v: object) -> object:
        if isinstance(v, int) and not isinstance(v, bool):
            return {"easy": v, "medium": v, "hard": v}
        return v


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
    # The world the episodes were simulated in (sim/fingerprint.py). ``simulation_digest`` is
    # folded into dataset_hash_for, so a geometry retune starts a new dataset instead of
    # resuming into the one it was meant to replace; the payload is recorded whole beside it
    # because the digest says a dataset's identity changed and only the payload says why.
    simulation: SimulationFingerprint
    simulation_digest: str
    # The retry-feedback template version (DSE-045). Recorded, deliberately NOT folded into
    # dataset_hash_for: the gate is unbuilt (DSE-018), so today the template reaches no model and
    # hashing it would re-key every existing dataset over a string nothing reads. It must join the
    # dataset hash when DSE-018 makes retries live, at which point it does shape the data.
    gate_feedback_version: str = GATE_FEEDBACK_VERSION
    command: list[str]
    dep_versions: dict[str, str]
    timestamp: str
    # Where the episodes were actually served (§7-7): interim-GPU pilot data must stay permanently
    # distinguishable from Myriad data. Deliberately NOT part of sweep_hash - the substrate is an
    # environment property, and separating roots (not hashes) keeps interim/Myriad datasets apart.
    serving_substrate: str = "unspecified"
    endpoint_base_url: str = ""
    # Which model and endpoint served each role (DSE-049). None/"" means B was served by A's model
    # at A's endpoint, i.e. self-play; a heterogeneous pair records both identities.
    model_b_name: str | None = None
    model_b_revision: str | None = None
    endpoint_base_url_b: str = ""
    # The wire format the schema constraint was sent in (DSE-032): a local-pilot dataset served
    # under `response_format` must stay distinguishable from a vLLM `guided_json` one, because the
    # constraining engine (llama.cpp/Outlines vs xgrammar) differs even though the schema does not.
    structured_mode: str = "guided_json"
    # The resolved decoding config per role, api key redacted. Temperature, decoding seed, the token
    # budget and the thinking switch all shape what the model emits, and none of them live in
    # SweepConfig - so without this the manifest recorded WHERE a run was served but not HOW it was
    # decoded, and two datasets differing only in max_tokens were indistinguishable after the fact.
    serving_a: ServingConfig | None = None
    serving_b: ServingConfig | None = None
    # The server-side stack (vLLM/torch versions, the physical GPU), captured by serve.sh and
    # read from the sidecar it writes. None off the cluster, where there is no separate server
    # process to describe - dep_versions already covers everything the client can see.
    serve_env: ServeEnv | None = None
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
    """Content hash of the resolved sweep config (sorted-key JSON, sha256, 16 hex).

    ``concurrency`` is excluded: it is an execution knob, not a result-shaping one, and hashing it
    would re-key the dataset when a resumed run changes worker count - orphaning every completed
    episode under the old hash.
    """
    canonical = json.dumps(sweep.model_dump(mode="json", exclude={"concurrency"}), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def dataset_hash_for(sweep: SweepConfig) -> str:
    """The dataset directory a sweep writes to - the one derivation every caller must use.

    Folds the live ``PROMPT_VERSION`` in, so a prompt bump starts a new dataset instead of resuming
    into the old one. Every reader (the runner, the drivers, the CLI) goes through here, because a
    caller that derived the hash without the prompt version would look in the wrong directory.

    Folds the simulation fingerprint in for the same reason (sim/fingerprint.py): the world
    constants are not ``SweepConfig`` fields, so without it the pre-registered difficulty retune
    resumes into the pre-retune dataset and re-reports its verdict.
    """
    return dataset_hash(
        sweep_hash(sweep),
        prompt_version=PROMPT_VERSION,
        simulation_digest=simulation_fingerprint().digest(),
    )


def build_sweep_manifest(
    sweep: SweepConfig,
    *,
    dataset_hash: str,
    prompt_version: str,
    serving_substrate: str = "unspecified",
    endpoint_base_url: str = "",
    endpoint_base_url_b: str = "",
    structured_mode: str = "guided_json",
    serving_a: ServingConfig | None = None,
    serving_b: ServingConfig | None = None,
) -> SweepManifest:
    """Assemble the run-level manifest from the sweep plus the live environment."""
    fingerprint = simulation_fingerprint()
    return SweepManifest(
        git_sha=git_sha(),
        simulation=fingerprint,
        simulation_digest=fingerprint.digest(),
        sweep=sweep,
        sweep_hash=sweep_hash(sweep),
        dataset_hash=dataset_hash,
        model_name=sweep.model.name,
        model_revision=sweep.model.revision,
        prompt_version=prompt_version,
        command=list(sys.argv),
        dep_versions=dep_versions(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        serving_substrate=serving_substrate,
        endpoint_base_url=endpoint_base_url,
        model_b_name=None if sweep.model_b is None else sweep.model_b.name,
        model_b_revision=None if sweep.model_b is None else sweep.model_b.revision,
        endpoint_base_url_b=endpoint_base_url_b,
        structured_mode=structured_mode,
        serving_a=_redact(serving_a),
        serving_b=_redact(serving_b),
        serve_env=serve_env(),
    )


def _redact(cfg: ServingConfig | None) -> ServingConfig | None:
    """Never write a key into an artefact, even the placeholder one (CLAUDE.md)."""
    return None if cfg is None else cfg.model_copy(update={"api_key": "REDACTED"})
