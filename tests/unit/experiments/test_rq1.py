"""DSE-020 RQ1 driver: the factorial assembles correctly and the analysis recovers a known gradient.

The synthetic fixture is the mandated known-answer case (CLAUDE.md): a C0->C4 degradation built into
both the outcome (success and per-step progress fall with the condition index) and the messages (C0
and C1 messages name the progress outcome - so CPVI lifts there - while C2-C4 messages are noise). A
correct analysis must surface the success gradient, the CPVI gradient, a negative C4 mixed-model
coefficient, and corrected contrast p-values.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.rq1 import analyse_rq1, rq1_sweep, write_rq1
from preceptx.experiments.sweep import expand
from preceptx.measure.featuriser import EncoderConfig, Featuriser

_MODEL = ModelConfig(name="m", revision="rev", tier="8b")


class _MsgEncoder:
    """dim0 recovers the progress token ('progress'/'stuck'); a 'noise' message carries none."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        rows = []
        for s in sentences:
            flag = 1.0 if "progress" in s else (-1.0 if "stuck" in s else 0.0)
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append([flag, *np.random.default_rng(seed).standard_normal(3).tolist()])
        return np.array(rows, dtype=np.float64)


def test_rq1_sweep_assembles_full_factorial() -> None:
    sweep = rq1_sweep(
        _MODEL, seeds=[1, 2], serialisations=["numeric", "grid"], difficulties=["hard"]
    )
    cells = expand(sweep)
    assert len(cells) == 5 * 2 * 1 * 2  # |C| * |S| * |D| * |seeds|
    assert {c.condition for c in cells} == {"C0", "C1", "C2", "C3", "C4"}


def _gradient_records(n_seeds: int = 6) -> list[HandoffRecord]:
    successes = {"C0": 5, "C1": 4, "C2": 3, "C3": 2, "C4": 1}  # out of n_seeds=6 -> falling rate
    progress = {"C0": 0.9, "C1": 0.7, "C2": 0.5, "C3": 0.3, "C4": 0.1}
    informative = {"C0", "C1"}
    records: list[HandoffRecord] = []
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        c: Condition = cond  # type: ignore[assignment]
        n_handoffs = n_seeds * 3
        n_prog = round(progress[cond] * n_handoffs)
        flags = [1] * n_prog + [0] * (n_handoffs - n_prog)
        h = 0
        for seed in range(n_seeds):
            success = seed < successes[cond]
            for step in range(3):
                ybp = flags[h]
                h += 1
                msg = ("progress" if ybp else "stuck") if cond in informative else "noise"
                records.append(
                    HandoffRecord(
                        episode_id=f"{cond}-s{seed}",
                        step=step,
                        condition=c,
                        serialisation="numeric",
                        difficulty="hard",
                        model="m",
                        seed=seed,
                        state={},
                        state_str=f"state {cond} s{seed} {step}",  # no progress/stuck token
                        message_raw=msg,
                        message_delivered=msg,
                        action={},
                        pre_state={},
                        post_state={},
                        progress=0.0,
                        success=success,
                        collision=False,
                        stuck=False,
                        y_binary_progress=bool(ybp),
                        y_terminal_success=success,
                    )
                )
    return records


def test_analyse_rq1_recovers_the_gradient(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result = analyse_rq1(_gradient_records(), feat, dataset_hash="d0")

    order = [c.condition for c in result.conditions]
    assert order == ["C0", "C1", "C2", "C3", "C4"]
    assert result.conditions[0].success_rate > result.conditions[-1].success_rate  # H1 outcome
    assert result.conditions[0].mean_cpvi > result.conditions[-1].mean_cpvi  # CPVI gradient
    assert all(np.isfinite(c.pvi_cpvi_gap) for c in result.conditions)  # the gap is always reported

    assert result.mixed_model.coef_no_mediator["C4"] < 0.0  # degradation relative to C0
    assert np.isfinite(result.mixed_model.cpvi_coef)
    assert "CPVI mediates" in result.mixed_model.mediation_note

    c4 = next(c for c in result.contrasts if c.condition == "C4")
    assert c4.cliffs_delta < 0.0  # C4 episodes succeed less often than C0
    assert c4.p_corrected >= c4.p_raw - 1e-12  # correction never increases significance
    assert result.seed_sensitivity.n_seeds == 6


def test_write_rq1_emits_table_and_json(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result = analyse_rq1(_gradient_records(), feat, dataset_hash="d0")
    out = write_rq1(result, tmp_path / "rq1")
    assert (out / "rq1.json").exists()
    assert (out / "rq1_results.csv").exists()
    # matplotlib is the optional viz extra; absent it the figures dict stays empty (no crash).
