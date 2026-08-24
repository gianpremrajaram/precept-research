from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from preceptx.config import ConfigError, ModelConfig
from preceptx.data.schema import HandoffRecord
from preceptx.data.writer import load_records
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for
from preceptx.serving.client import LLMClient, ServingConfig
from preceptx.sim import arena
from preceptx.sim.fingerprint import simulation_fingerprint

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


def _client() -> LLMClient:
    return LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _wait_script(request: httpx.Request) -> httpx.Response:
    if b"guided_json" in request.content or b"response_format" in request.content:
        return httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    return httpx.Response(200, json=_completion("hold position"))


def _sweep(concurrency: int = 4) -> SweepConfig:
    return SweepConfig(
        conditions=["C0", "C4"],
        serialisations=["numeric"],
        difficulties=["easy"],
        seeds=[1, 2, 3],
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        max_steps=2,
        concurrency=concurrency,
    )


def _data_dir(sweep: SweepConfig, root: Path) -> Path:
    return root / dataset_hash_for(sweep)


@respx.mock
def test_run_grid_writes_one_record_set_per_cell(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    summary = run_grid(sweep, _client(), root=tmp_path)
    assert summary.n_cells == 6  # 2 conditions x 3 seeds
    assert summary.n_episodes == 6
    assert summary.n_handoffs == 6 * 2  # max_steps=2 per episode
    run_dir = tmp_path / f"{dataset_hash_for(sweep)}-run"
    assert (run_dir / "manifest.json").exists()  # run manifest persisted beside the dataset dir
    records = load_records(dataset_hash_for(sweep), root=tmp_path)
    assert len({r.episode_id for r in records}) == 6  # no dropped/duplicated episodes


@respx.mock
def test_run_grid_is_concurrency_safe(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep(concurrency=4)  # the serialised-write lock must survive 4 concurrent episodes
    run_grid(sweep, _client(), root=tmp_path)
    records = load_records(dataset_hash_for(sweep), root=tmp_path)
    assert len(records) == 6 * 2  # every cell's records land exactly once (no part-index race)
    assert len({r.episode_id for r in records}) == 6


@respx.mock
def test_run_grid_jitters_start_pose_per_seed(tmp_path: Path) -> None:
    # P0-2: the default sweep jitter makes seeds true replicates - three seeds, three distinct
    # problem instances - while the same seed gives the SAME instance across conditions (paired).
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    records = load_records(dataset_hash_for(sweep), root=tmp_path)

    def pose(r: HandoffRecord) -> tuple[float, float, float]:
        return (r.pre_state["com_x"], r.pre_state["com_y"], r.pre_state["angle"])

    step0 = [r for r in records if r.step == 0]
    c0_poses = {r.seed: pose(r) for r in step0 if r.condition == "C0"}
    c4_poses = {r.seed: pose(r) for r in step0 if r.condition == "C4"}
    assert len(set(c0_poses.values())) == 3  # distinct instance per seed
    assert c0_poses == c4_poses  # same seed -> same instance across conditions (matched pairs)


@respx.mock
def test_manifest_records_substrate_endpoint_and_result_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    monkeypatch.setenv("PRECEPTX_SERVING_SUBSTRATE", "interim-test")
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    run_dir = tmp_path / f"{dataset_hash_for(sweep)}-run"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["serving_substrate"] == "interim-test"  # §7-7: interim data stays labelled
    assert manifest["endpoint_base_url"] == BASE_URL
    assert manifest["sweep"]["jitter"]["x_range"] == [1.2, 2.8]  # P0-2 knob is audit-visible
    assert manifest["sweep"]["outcome"]["k"] == 3  # P1-6: the label horizon is manifested
    assert manifest["sweep"]["step"]["linear_impulse"] == 3.0


@respx.mock
def test_run_grid_resume_skips_completed_cells(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    parts_first = len(list(_data_dir(sweep, tmp_path).glob("part-*.parquet")))
    summary = run_grid(sweep, _client(), root=tmp_path)  # rerun: everything already complete
    parts_second = len(list(_data_dir(sweep, tmp_path).glob("part-*.parquet")))
    assert parts_second == parts_first  # no new parts written on resume
    assert summary.n_episodes == 6  # summary still reports the full grid
    records = load_records(dataset_hash_for(sweep), root=tmp_path)
    assert len(records) == 6 * 2  # not duplicated


# --- DSE-049 / DSE-032: what the sweep manifest records about serving --------------------------

BASE_URL_B = "http://localhost:8001/v1"
CHAT_B = f"{BASE_URL_B}/chat/completions"


def _manifest(sweep: SweepConfig, root: Path) -> dict[str, object]:
    run_dir = root / f"{dataset_hash_for(sweep)}-run"
    return json.loads((run_dir / "manifest.json").read_text())


@respx.mock
def test_manifest_records_structured_mode_and_self_play_roles(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    client = LLMClient(
        ServingConfig(
            model="m", base_url=BASE_URL, max_retries=0, structured_mode="response_format"
        )
    )
    run_grid(sweep, client, root=tmp_path)
    manifest = _manifest(sweep, tmp_path)
    assert manifest["structured_mode"] == "response_format"
    assert manifest["model_b_name"] is None  # self-play: B is served by A's model
    assert manifest["endpoint_base_url_b"] == ""


@respx.mock
def test_manifest_records_both_role_identities(tmp_path: Path) -> None:
    respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion("hold position")),
    )
    respx.post(CHAT_B).mock(
        return_value=httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"}))),
    )
    sweep = _sweep().model_copy(
        update={"model_b": ModelConfig(name="mb", revision="rev-b", tier="14b")}
    )
    client_b = LLMClient(ServingConfig(model="mb", base_url=BASE_URL_B, max_retries=0))
    run_grid(sweep, _client(), client_b, root=tmp_path)

    manifest = _manifest(sweep, tmp_path)
    assert manifest["model_name"] == "m" and manifest["model_revision"] == "rev"
    assert manifest["model_b_name"] == "mb" and manifest["model_b_revision"] == "rev-b"
    assert manifest["endpoint_base_url_b"] == BASE_URL_B


def test_second_client_without_a_declared_model_fails_loud(tmp_path: Path) -> None:
    # A second endpoint with no model_b block would leave the manifest lying about role B.
    with pytest.raises(ConfigError):
        run_grid(_sweep(), _client(), _client(), root=tmp_path)


@respx.mock
def test_manifest_records_the_decoding_config_without_the_key(tmp_path: Path) -> None:
    # Temperature, seed, token budget and the thinking switch shape what the model emits and live
    # nowhere in SweepConfig; the api key must never reach an artefact.
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(
        sweep,
        LLMClient(
            ServingConfig(
                model="m", base_url=BASE_URL, max_retries=0, max_tokens=64, thinking_switch="/x"
            )
        ),
        root=tmp_path,
    )
    serving = _manifest(sweep, tmp_path)["serving_a"]
    assert isinstance(serving, dict)
    assert serving["max_tokens"] == 64 and serving["thinking_switch"] == "/x"
    assert serving["temperature"] == 0.0 and serving["seed"] == 0
    assert serving["api_key"] == "REDACTED"


# --- dataset identity carries the world it was simulated in (sim/fingerprint.py) ----------------


@respx.mock
def test_a_geometry_retune_schedules_a_fresh_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the fingerprint exists to prevent, exercised end to end through ``run_grid``.

    Before the world reached dataset identity: widen a slit, re-run, and this read the *pre*-retune
    episode ids, found the grid already complete, scheduled nothing, and let the driver re-report
    the old verdict against the new geometry. On the cluster, on the verdict of record, looking
    exactly like a success. A unit test on the digest alone would not have caught it - the claim
    is specifically that ``run_grid`` resolves and consumes the new identity.
    """
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    before = dataset_hash_for(sweep)
    assert len(load_records(before, root=tmp_path)) == 6 * 2

    monkeypatch.setitem(arena._DIFFICULTY_SLITS, "easy", 2.4)  # the pre-registered retune lever
    after = dataset_hash_for(sweep)
    assert after != before

    summary = run_grid(sweep, _client(), root=tmp_path)
    assert summary.n_episodes == 6  # a full fresh grid, not "0 pending"
    assert len(load_records(after, root=tmp_path)) == 6 * 2
    assert len(load_records(before, root=tmp_path)) == 6 * 2  # the old dataset is left intact


@respx.mock
def test_resuming_into_a_foreign_world_fails_loud(tmp_path: Path) -> None:
    """Defence in depth: a recorded fingerprint that disagrees with this process aborts the run.

    Unreachable through the hash (different world, different directory), so this stands in for what
    identity cannot cover - a hand-copied directory, or a future change to how the hash is composed.
    """
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    path = tmp_path / f"{dataset_hash_for(sweep)}-run" / "manifest.json"
    payload = json.loads(path.read_text())
    payload["simulation_digest"] = "0" * 16
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="simulation fingerprint"):
        run_grid(sweep, _client(), root=tmp_path)


@respx.mock
def test_a_dataset_with_no_manifest_still_resumes(tmp_path: Path) -> None:
    """The guard must not fail closed on a missing manifest.

    The manifest is written when a sweep *finishes*, so its absence is the ordinary
    killed-at-wallclock case - the one resumability exists to serve. Failing closed there would
    have broken the feature the guard sits inside.
    """
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    (tmp_path / f"{dataset_hash_for(sweep)}-run" / "manifest.json").unlink()
    assert run_grid(sweep, _client(), root=tmp_path).n_episodes == 6


@respx.mock
def test_the_manifest_records_the_world_and_its_digest(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=_wait_script)
    sweep = _sweep()
    run_grid(sweep, _client(), root=tmp_path)
    manifest = _manifest(sweep, tmp_path)
    assert manifest["simulation_digest"] == simulation_fingerprint().digest()
    simulation = manifest["simulation"]
    assert isinstance(simulation, dict)
    # The payload, not only the digest: a digest says identity changed, the payload says why.
    assert simulation["slit_widths"] == {"easy": 1.8, "medium": 1.2, "hard": 1.1}
