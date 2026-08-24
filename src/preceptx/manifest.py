"""The run manifest - the reproducibility backbone every run must write.

A ``RunManifest`` records git SHA, config hash, model + encoder revisions, dependency versions, the
exact command, seed, timestamp, and (once known) key metrics and artefact paths. A run without a
complete manifest is not audit-usable and does not count as done (CLAUDE.md). The schema is a stable
contract consumed by the examiner appendix (DSE-030); changing it is a result-affecting change.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ExperimentConfig

MANIFEST_VERSION = 1

# Dependencies whose installed versions are pinned into every manifest for reproducibility.
# pymunk literally shapes trajectories; scipy/statsmodels shape the reported statistics;
# joblib/sentence-transformers shape persisted probes and embeddings (P1-9). Server-side
# vllm/torch versions cannot be seen from the client environment at all; scripts/myriad/serve.sh
# writes them to a sidecar that ``serve_env`` below reads into the manifest.
_TRACKED_DEPS = (
    "pydantic",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "langgraph",
    "openai",
    "hydra-core",
    "omegaconf",
    "pymunk",
    "scipy",
    "statsmodels",
    "joblib",
    "sentence-transformers",
)


class ManifestError(RuntimeError):
    """A manifest could not be built (e.g. git SHA unavailable) or read back."""


class RunManifest(BaseModel):
    """The mandatory per-run reproducibility record."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = MANIFEST_VERSION
    git_sha: str
    config: ExperimentConfig
    config_hash: str
    model_name: str
    model_revision: str
    encoder_revision: str | None = None
    seed: int = Field(ge=0)
    command: list[str]
    dep_versions: dict[str, str]
    timestamp: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artefact_paths: dict[str, str] = Field(default_factory=dict)


def config_hash(config: ExperimentConfig) -> str:
    """A stable content hash of the resolved config (sorted-key JSON, sha256, 16 hex chars)."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ManifestError(
            "could not resolve git SHA; runs must be made from a git checkout"
        ) from exc
    return out.stdout.strip()


class ServeEnv(BaseModel):
    """The server-side serving environment, as captured by ``scripts/myriad/serve.sh``.

    vLLM's and torch's versions and the physical GPU live on the compute node, in the process
    that serves the model - the client that writes the manifest cannot import them or see the
    card. They were previously echoed to the job log only, which made the run of record's
    server-side stack recoverable solely by a human copying four lines out of
    ``precept-pilot.o<jobid>``. ``digest`` is over the sidecar bytes, so the invocation record
    of a manual benchmark can name exactly which capture it ran against.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    digest: str
    values: dict[str, str]


def serve_env() -> ServeEnv | None:
    """Read the serving-environment sidecar named by ``PRECEPTX_SERVE_ENV``, if there is one.

    ``None`` off the cluster: a local run has no separate server process to describe, and the
    absence is itself accurate rather than a degraded reading. A sidecar that is named but
    unreadable is an error, not a silent ``None`` - it means the job wrote one and we lost it.
    """
    raw = os.environ.get("PRECEPTX_SERVE_ENV", "")
    if not raw:
        return None
    path = Path(raw)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManifestError(
            f"PRECEPTX_SERVE_ENV points at {path}, which cannot be read: {exc}"
        ) from exc
    try:
        values = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"serving-environment sidecar {path} is not valid JSON: {exc}") from exc
    return ServeEnv(
        path=str(path),
        digest=hashlib.sha256(payload).hexdigest()[:16],
        values={str(k): str(v) for k, v in values.items()},
    )


def git_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Recorded beside the SHA because a dirty tree means the SHA does not describe the code that
    ran, which is the difference between a reproducible artefact and a plausible-looking one.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain"], check=False, capture_output=True, text=True
    )
    return bool(out.stdout.strip())


def dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dep in _TRACKED_DEPS:
        try:
            versions[dep] = metadata.version(dep)
        except metadata.PackageNotFoundError:
            versions[dep] = "not-installed"
    return versions


def build_manifest(
    config: ExperimentConfig,
    *,
    encoder_revision: str | None = None,
    metrics: dict[str, Any] | None = None,
    artefact_paths: dict[str, str] | None = None,
) -> RunManifest:
    """Assemble a complete manifest from the config plus the live environment."""
    return RunManifest(
        git_sha=git_sha(),
        config=config,
        config_hash=config_hash(config),
        model_name=config.model.name,
        model_revision=config.model.revision,
        encoder_revision=encoder_revision,
        seed=config.seed,
        command=list(sys.argv),
        dep_versions=dep_versions(),
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        metrics=metrics or {},
        artefact_paths=artefact_paths or {},
    )


def write_manifest(manifest: RunManifest, path: Path | str) -> None:
    """Persist a manifest as pretty JSON."""
    Path(path).write_text(manifest.model_dump_json(indent=2))


def read_manifest(path: Path | str) -> RunManifest:
    """Load a manifest back into a validated ``RunManifest``."""
    try:
        return RunManifest.model_validate_json(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
