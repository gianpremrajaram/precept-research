from __future__ import annotations

from pathlib import Path

from preceptx.config import ExperimentConfig
from preceptx.manifest import build_manifest, config_hash, read_manifest, write_manifest


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
