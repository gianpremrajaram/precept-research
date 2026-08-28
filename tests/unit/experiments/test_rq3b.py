from __future__ import annotations

from typing import Any

import pytest

from preceptx.config import ConfigError
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.rq3b import (
    GATE_MODES,
    RQ3bConfig,
    analyse_rq3b,
    rq3b_sweeps,
    write_rq3b,
)
from preceptx.experiments.sweep import ModelConfig, SweepConfig, dataset_hash_for
from preceptx.gate.integration import GateConfig, GateMode

MODEL = ModelConfig(name="m", revision="rev", tier="8b")


def _sweep(**overrides: Any) -> SweepConfig:
    base: dict[str, Any] = {
        "conditions": ["C0"],
        "serialisations": ["numeric"],
        "difficulties": ["easy"],
        "seeds": [1, 2, 3],
        "model": MODEL,
    }
    base.update(overrides)
    return SweepConfig(**base)


def _episodes(
    mode: str, *, n: int, n_success: int, steps: int = 10, blocks: int = 0
) -> list[HandoffRecord]:
    """``n`` episodes of ``steps`` handoffs each, the first ``n_success`` reaching the goal."""
    cond: Condition = "C0"
    out: list[HandoffRecord] = []
    for ep in range(n):
        won = ep < n_success
        for step in range(steps):
            out.append(
                HandoffRecord(
                    episode_id=f"{mode}-e{ep}",
                    step=step,
                    condition=cond,
                    serialisation="numeric",
                    difficulty="easy",
                    model="m",
                    seed=ep,
                    state={},
                    state_str="s",
                    observation="s",
                    message_raw="m",
                    message_delivered="m",
                    action={},
                    pre_state={},
                    post_state={},
                    progress=0.0,
                    success=won,
                    collision=False,
                    stuck=False,
                    gate_blocked=step < blocks,
                    gate_retries=1 if step < blocks else 0,
                    y_binary_progress=won,
                    y_terminal_success=won,
                )
            )
    return out


# --- mode assembly ----------------------------------------------------------


def test_the_four_arms_differ_only_in_the_gate_and_never_share_a_dataset() -> None:
    arms = rq3b_sweeps(_sweep(), statistic_key="cosine")
    assert tuple(arms) == GATE_MODES
    assert arms["off"].gate is None  # the ungated arm can reuse an existing RQ1 dataset
    assert {m: a.gate.mode for m, a in arms.items() if a.gate is not None} == {
        "active": "active",
        "matched_random": "matched_random",
        "random_trigger": "random_trigger",
    }
    # Everything but `gate` is identical, so a difference between arms cannot come from the grid.
    assert len({a.model_dump_json(exclude={"gate"}) for a in arms.values()}) == 1
    # The load-bearing property: four arms, four directories. Pooling would not error, it would
    # average the treatment into its own control.
    assert len({dataset_hash_for(a) for a in arms.values()}) == 4


def test_a_gated_base_sweep_is_refused() -> None:
    gated = _sweep(gate=GateConfig(mode="active", statistic_key="cosine"))
    with pytest.raises(ConfigError, match="ungated base sweep"):
        rq3b_sweeps(gated, statistic_key="cosine")


def test_control_settings_reach_every_gated_arm() -> None:
    arms = rq3b_sweeps(_sweep(), statistic_key="fail", max_retries=3, random_rate=0.4, gate_seed=7)
    gated = [a.gate for a in arms.values() if a.gate is not None]
    assert {g.statistic_key for g in gated} == {"fail"}
    assert {g.max_retries for g in gated} == {3}
    assert {g.random_rate for g in gated} == {0.4}
    assert {g.seed for g in gated} == {7}


# --- H6 ---------------------------------------------------------------------


def _four_arms(successes: dict[str, int], **kw: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    records = {
        m: _episodes(m, n=24, n_success=successes[m], **({"blocks": 2} if m != "off" else {}), **kw)
        for m in GATE_MODES
    }
    return records, {m: f"hash-{m}" for m in GATE_MODES}


def test_h6_supported_when_the_gate_beats_both_score_blind_controls() -> None:
    records, hashes = _four_arms({"active": 20, "matched_random": 4, "random_trigger": 4, "off": 3})
    result = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=RQ3bConfig(n_boot=2000))
    assert result.verdict.startswith("H6 SUPPORTED")
    success = {c.control: c for c in result.contrasts if c.outcome == "success"}
    assert success["matched_random"].delta == pytest.approx(16 / 24)
    assert success["matched_random"].delta_ci[0] > 0.0  # interval excludes no effect
    assert all(c.p_corrected >= c.p_value for c in result.contrasts)  # Holm never shrinks a p


def test_h6_null_is_reported_as_a_finding_not_a_failure() -> None:
    # Same success rate everywhere but different step counts, so the arms are not identical and the
    # comparison is a real null rather than the untestable case below.
    records, hashes = _four_arms(dict.fromkeys(GATE_MODES, 6))
    records["active"] = _episodes("active", n=24, n_success=6, steps=11, blocks=2)
    result = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=RQ3bConfig(n_boot=2000))
    assert result.verdict.startswith("H6 NOT SUPPORTED")
    assert "finding about the statistic" in result.verdict


def test_a_floored_grid_is_untestable_rather_than_a_null() -> None:
    # The live risk: job 232980 returned 1/96. Every arm identical means the task produced no
    # outcome variance, which is a statement about the grid and must not read as "the gate failed".
    records, hashes = _four_arms(dict.fromkeys(GATE_MODES, 0))
    for m in GATE_MODES:
        records[m] = _episodes(m, n=24, n_success=0)  # identical blocks too
    result = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=RQ3bConfig(n_boot=2000))
    assert result.verdict.startswith("UNTESTABLE")
    assert "not about the gate" in result.verdict


def test_a_missing_arm_fails_loud() -> None:
    records, hashes = _four_arms(dict.fromkeys(GATE_MODES, 4))
    del records["matched_random"]
    with pytest.raises(ConfigError, match="needs all four arms"):
        analyse_rq3b(records, hashes, statistic_key="cosine")


def test_the_realised_firing_rate_is_measured_not_assumed() -> None:
    records, hashes = _four_arms({"active": 8, "matched_random": 4, "random_trigger": 4, "off": 3})
    result = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=RQ3bConfig(n_boot=2000))
    by_mode = {m.mode: m for m in result.modes}
    assert by_mode["active"].block_rate == pytest.approx(0.2)  # 2 blocked of 10 steps
    assert by_mode["off"].block_rate == 0.0
    assert by_mode["active"].mean_retries == pytest.approx(2.0)


def test_write_rq3b_emits_the_tables_and_the_verdict(tmp_path: Any) -> None:
    records, hashes = _four_arms({"active": 12, "matched_random": 4, "random_trigger": 4, "off": 3})
    result = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=RQ3bConfig(n_boot=2000))
    out = write_rq3b(result, tmp_path / "rq3b")
    for name in ("rq3b.json", "rq3b_modes.csv", "rq3b_contrasts.csv", "verdict.md"):
        assert (out / name).is_file(), name
    assert result.verdict in (out / "verdict.md").read_text()


def test_h6_reproduces_on_the_same_fixture() -> None:
    records, hashes = _four_arms({"active": 14, "matched_random": 5, "random_trigger": 4, "off": 3})
    cfg = RQ3bConfig(n_boot=2000, seed=3)
    a = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=cfg)
    b = analyse_rq3b(records, hashes, statistic_key="cosine", cfg=cfg)
    assert a.model_dump() == b.model_dump()  # seeded bootstrap: identical intervals and p-values


def test_gate_modes_are_exactly_the_integration_modes() -> None:
    # Drift guard: a fifth arm added to GateMode without a home here would be silently untested.
    assert set(GATE_MODES) == set(GateMode.__args__)  # type: ignore[attr-defined]
