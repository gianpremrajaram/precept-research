"""DSE-020 RQ1 driver: the factorial assembles correctly and the analysis recovers a known gradient.

The synthetic fixture is the mandated known-answer case (CLAUDE.md), and is built so CPVI genuinely
*mediates* condition -> success rather than being a parallel consequence of it: one per-episode flag
(is this episode's message informative?) sets the message, the per-handoff progress label, and the
terminal outcome together, and the fraction of informative episodes falls C0->C4. So a correct
analysis must surface the success gradient, the CPVI gradient, a negative C4 handoff coefficient,
corrected contrast p-values, AND a positive CPVI->success path with a negative C4 indirect effect.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from preceptx.config import ModelConfig
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.rq1 import RQ1Config, analyse_rq1, rq1_sweep, write_rq1
from preceptx.experiments.sweep import expand
from preceptx.measure.featuriser import EncoderConfig, Featuriser

_MODEL = ModelConfig(name="m", revision="rev", tier="8b")
# Keep the model-refit mediation bootstrap small so the unit suite stays well under its 30s budget.
_FAST = RQ1Config(n_boot=300, n_boot_mediation=50)


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
    # Two falling-with-condition drivers, kept separate:
    #  - true per-handoff progress (rate falls C0->C4) is the handoff outcome y_binary_progress;
    #  - the per-episode "informative" flag (count falls C0->C4) drives the message, CPVI, success.
    # An informative episode's message reveals its true progress ("progress"/"stuck", so CPVI lifts
    # whether progress is 0 or 1); a non-informative one emits constant "noise". Noise rows span
    # conditions, so they carry mixed y - noise is genuinely uninformative and CPVI falls C0->C4. In
    # mixed conditions C1-C3, informative episodes carry high CPVI and succeed while non-informative
    # ones do not - that within-condition spread identifies the positive CPVI->success path.
    informative_count = {"C0": 6, "C1": 5, "C2": 3, "C3": 1, "C4": 0}  # of n_seeds=6 episodes
    progress_rate = {"C0": 0.9, "C1": 0.7, "C2": 0.5, "C3": 0.3, "C4": 0.1}  # per-handoff true rate
    records: list[HandoffRecord] = []
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        c: Condition = cond  # type: ignore[assignment]
        n_handoffs = n_seeds * 3
        n_prog = round(progress_rate[cond] * n_handoffs)
        prog_flags = [1] * n_prog + [0] * (n_handoffs - n_prog)
        h = 0
        for seed in range(n_seeds):
            informative = seed < informative_count[cond]
            for step in range(3):
                y = prog_flags[h]
                h += 1
                msg = ("progress" if y else "stuck") if informative else "noise"
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
                        state_str=f"state {cond} s{seed} {step}",  # no outcome token in the state
                        message_raw=msg,
                        message_delivered=msg,
                        action={},
                        pre_state={},
                        post_state={},
                        progress=0.0,
                        success=informative,  # the working channel is what gets the load home
                        collision=False,
                        stuck=False,
                        y_binary_progress=bool(y),
                        y_terminal_success=informative,
                    )
                )
    return records


def test_analyse_rq1_recovers_the_gradient(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result = analyse_rq1(_gradient_records(), feat, dataset_hash="d0", cfg=_FAST)

    order = [c.condition for c in result.conditions]
    assert order == ["C0", "C1", "C2", "C3", "C4"]
    assert result.conditions[0].success_rate > result.conditions[-1].success_rate  # H1 outcome
    assert result.conditions[0].mean_cpvi > result.conditions[-1].mean_cpvi  # CPVI gradient
    assert all(np.isfinite(c.pvi_cpvi_gap) for c in result.conditions)  # the gap is always reported

    mm = result.mixed_model
    assert mm.coef_no_mediator["C4"] < 0.0  # H1 handoff model: degradation relative to C0
    assert np.isfinite(mm.diagnostic_cpvi_coef)  # within-episode diagnostic still computed
    assert mm.mediation_outcome == "episode_success"  # H2 tests the headline DV, not progress
    assert "mediated by CPVI" in mm.mediation_note

    # H2 episode-level mediation: the channel suppresses success *through* lowered CPVI (a*b < 0).
    assert mm.path_b > 0.0  # more episode-mean CPVI -> more success
    c4_med = next(m for m in mm.mediations if m.condition == "C4")
    assert c4_med.path_a < 0.0  # C4 carries less CPVI than C0
    assert c4_med.indirect < 0.0  # negative indirect effect: degradation flows through CPVI
    assert all(np.isfinite(c4_med.indirect_ci))  # the indirect effect is reported with an interval

    c4 = next(c for c in result.contrasts if c.condition == "C4")
    assert c4.cliffs_delta < 0.0  # C4 episodes succeed less often than C0
    assert c4.p_corrected >= c4.p_raw - 1e-12  # correction never increases significance
    assert result.seed_sensitivity.n_seeds == 6  # per-seed C0-minus-hardest gap, one value per seed
    assert result.seed_sensitivity.mean > 0.0  # the gradient holds on average across seeds


def test_write_rq1_emits_table_and_json(tmp_path: Path) -> None:
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "e"), encoder=_MsgEncoder())
    result = analyse_rq1(_gradient_records(), feat, dataset_hash="d0", cfg=_FAST)
    out = write_rq1(result, tmp_path / "rq1")
    assert (out / "rq1.json").exists()
    assert (out / "rq1_results.csv").exists()
    # matplotlib is the optional viz extra; absent it the figures dict stays empty (no crash).
