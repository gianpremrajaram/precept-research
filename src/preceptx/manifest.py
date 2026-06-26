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
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ExperimentConfig

MANIFEST_VERSION = 1

# Dependencies whose installed versions are pinned into every manifest for reproducibility.
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


def _git_sha() -> str:
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


def _dep_versions() -> dict[str, str]:
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
        git_sha=_git_sha(),
        config=config,
        config_hash=config_hash(config),
        model_name=config.model.name,
        model_revision=config.model.revision,
        encoder_revision=encoder_revision,
        seed=config.seed,
        command=list(sys.argv),
        dep_versions=_dep_versions(),
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
