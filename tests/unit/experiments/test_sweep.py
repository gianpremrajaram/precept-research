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
from preceptx.gate.integration import GateConfig
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


def _job_227886_grid(**overrides: object) -> SweepConfig:
    """The grid job 227886 ran: C0-C4 x easy/medium/hard x seeds 0-31 on Qwen3-14B."""
    return SweepConfig(
        conditions=["C0", "C1", "C2", "C3", "C4"],
        serialisations=["numeric"],
        difficulties=["easy", "medium", "hard"],
        seeds=list(range(32)),
        model=ModelConfig(
            name="Qwen/Qwen3-14B",
            revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
            tier="14b",
        ),
        **overrides,  # type: ignore[arg-type]
    )


def test_the_recorded_grid_still_hashes_to_its_own_dataset() -> None:
    """Job 227886's dataset identity is reproducible from its own configuration, forever.

    Adding a field to ``SweepConfig`` changes ``model_dump``, and ``sweep_hash`` hashes that dump -
    so a naive addition silently re-points every recorded dataset. The budgets are pinned here
    rather than defaulted BECAUSE DSE-059 moved the defaults to 30/35/35: this asserts that the
    227886 configuration still keys to 227886's directory, which is the property that matters and
    the one a default change must never break. What a *new* default hashes to is the next test.

    ``hold_orientation`` is pinned by VALUE and its value did not change - DSE-059 altered what
    True *means*, not what it is - so ``sweep_hash`` cannot see that change.
    ``ENVIRONMENT_SCHEMA_VERSION`` is what re-keys it, which is the case that field exists for, and
    why the next test asserts on ``dataset_hash_for`` and not on ``sweep_hash`` alone.
    """
    recorded = _job_227886_grid(
        jitter=ScenarioJitter(y_range=(1.5, 4.5)),
        step=StepConfig(angular_impulse=0.5),
        max_steps={"easy": 20, "medium": 25, "hard": 25},
    )
    assert (
        recorded.gate is None
    )  # ungated is the default, and the default must hash as it always did
    assert sweep_hash(recorded) == "afcd6a53ee11edd7"


def test_the_corrected_actuator_keys_a_new_dataset() -> None:
    """DSE-059 is a new task generation and must not share 227886's directory.

    The corrected rotation quantum, the true orientation hold and the 0.64 hard aperture change what
    an episode *is*, so resuming into 54ed65e6cc9e7d17 would mix two physics under one hash - the
    exact failure ``sim/fingerprint.py`` exists to prevent. Asserting inequality rather than a
    literal keeps this test from becoming a second place to update on every legitimate retune.
    """
    assert sweep_hash(_job_227886_grid()) != "afcd6a53ee11edd7"
    assert dataset_hash_for(_job_227886_grid()) != "54ed65e6cc9e7d17"


def test_each_gate_arm_keys_its_own_dataset() -> None:
    """The RQ3b comparison is four arms over one grid; four arms sharing a dataset is no comparison.

    Without the gate in the hash they collide in one run directory and ``run_grid``'s resume reads
    arm 1's episode ids as arms 2-4's completed work - the whole causal contrast collapsing into one
    arm run four times, with nothing in the artefacts to say so. ``off`` is a declared arm too: it
    runs the gate machinery with the gate never firing, which is a different run of record from an
    ungated RQ1 sweep even though the episodes would match.
    """
    base = _sweep()
    hashes = {
        mode: dataset_hash_for(base.model_copy(update={"gate": GateConfig(mode=mode)}))
        for mode in ("off", "active", "matched_random", "random_trigger")
    }
    hashes["ungated"] = dataset_hash_for(base)
    assert len(set(hashes.values())) == 5, hashes

    # It is the arm's identity that keys it, not merely the presence of a gate: two arms differing
    # only in retry budget or control rate are different treatments and must not pool either.
    active = base.model_copy(update={"gate": GateConfig(mode="active")})
    assert dataset_hash_for(active) != dataset_hash_for(
        base.model_copy(update={"gate": GateConfig(mode="active", max_retries=3)})
    )
    assert dataset_hash_for(active) != dataset_hash_for(
        base.model_copy(update={"gate": GateConfig(mode="active", statistic_key="info")})
    )
