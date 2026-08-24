from __future__ import annotations

import json
from pathlib import Path

import pytest

from preceptx.config import ExperimentConfig
from preceptx.manifest import (
    ManifestError,
    build_manifest,
    config_hash,
    read_manifest,
    serve_env,
    write_manifest,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "condition": "C0",
            "serialisation": "numeric",
            "difficulty": "easy",
            "model": {"name": "Qwen/Qwen3-14B-Instruct", "revision": "abc123", "tier": "14b"},
            "seed": 7,
        }
    )


def test_config_hash_is_stable_and_content_sensitive() -> None:
    config = _config()
    assert config_hash(config) == config_hash(config)
    other = config.model_copy(update={"seed": 8})
    assert config_hash(config) != config_hash(other)


def test_build_manifest_captures_environment() -> None:
    manifest = build_manifest(_config(), metrics={"success_rate": 0.6})
    assert len(manifest.git_sha) == 40  # full SHA from a real checkout
    assert manifest.model_name == "Qwen/Qwen3-14B-Instruct"
    assert manifest.model_revision == "abc123"
    assert manifest.seed == 7
    assert manifest.metrics["success_rate"] == 0.6
    assert manifest.dep_versions["pydantic"] != "not-installed"
    assert manifest.command  # argv captured


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = build_manifest(_config())
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


# --- the server-side environment sidecar (scripts/myriad/serve.sh) ------------------------------


def test_serve_env_is_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sidecar off the cluster: a local run has no separate server process to describe, and
    saying so is accurate rather than a degraded reading."""
    monkeypatch.delenv("PRECEPTX_SERVE_ENV", raising=False)
    assert serve_env() is None


def test_serve_env_reads_the_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM's and torch's versions live in the server process; the client cannot import them."""
    path = tmp_path / "serve_env.json"
    path.write_text(json.dumps({"vllm": "0.18.1", "torch": "2.6.0", "gpu": "NVIDIA A100-40GB"}))
    monkeypatch.setenv("PRECEPTX_SERVE_ENV", str(path))
    captured = serve_env()
    assert captured is not None
    assert captured.values["vllm"] == "0.18.1"
    assert captured.path == str(path)
    assert len(captured.digest) == 16  # over the bytes, so a benchmark row can name this capture


def test_a_named_but_missing_sidecar_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named-and-unreadable is not the same as absent: it means the job wrote one and we lost it."""
    monkeypatch.setenv("PRECEPTX_SERVE_ENV", str(tmp_path / "nope.json"))
    with pytest.raises(ManifestError, match="cannot be read"):
        serve_env()


def test_a_corrupt_sidecar_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "serve_env.json"
    path.write_text("{not json")
    monkeypatch.setenv("PRECEPTX_SERVE_ENV", str(path))
    with pytest.raises(ManifestError, match="not valid JSON"):
        serve_env()
