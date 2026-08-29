"""RQ3b: the interventional arm - does blocking low-information handoffs improve outcomes (H6)?

RQ1 and RQ2 detect. This asks whether detection becomes *enforcement*: the gate scores each handoff
at the boundary and blocks the weak ones, and the question is whether episodes run under it end
better than episodes run under blocking that is just as frequent but score-blind.

**Four arms, four datasets, never three.** ``SweepConfig.gate`` is inside ``dataset_hash_for``, so
each mode writes its own directory and ``run_grid``'s two guards refuse a gate the sweep does not
declare. That is deliberate: pooling a gated arm into the ungated one would not error, it would
quietly average the treatment into its own control.

**Why two controls rather than one.** ``off`` alone cannot support the causal claim, because the
active gate does two things at once - it blocks, and a block buys agent A another turn to say
something better. ``matched_random`` blocks at the gate's own calibrated firing rate and
``random_trigger`` at a fixed one, both keyed on (episode seed, step) and blind to the score, so
"more retries" is held constant and only "the *right* handoffs were retried" varies. This is the
Lowe et al. (2019) demand made concrete (roadmap section 3.5).

**A null here is a result.** H6 failing - the gate matching its score-blind controls - says the
statistic does not localise the handoffs that matter, which is a finding about the measurement and
is reported as one. ``verdict`` distinguishes it from the different and *un*reportable case where
every arm scored identically because the task produced no outcome variance to move.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.analysis.figures import ci_plot
from preceptx.analysis.stats import bootstrap_ci, cliffs_delta, correct_pvalues
from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord
from preceptx.data.writer import load_records
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for
from preceptx.gate.calibration import CalibrationReport
from preceptx.gate.integration import GateConfig, GateMode, RuntimeGate
from preceptx.gate.statistics import (
    CosineStatistic,
    FailStatistic,
    InfoStatistic,
    Statistic,
    failure_label,
)
from preceptx.measure.featuriser import Featuriser
from preceptx.serving.client import LLMClient

# Fresh factories keyed by the stable ``Statistic.key``, mirroring rq2._statistics.
_STATISTICS: dict[str, type[Statistic]] = {
    "info": InfoStatistic,
    "fail": FailStatistic,
    "cosine": CosineStatistic,
}

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Ordered so every table and figure reads treatment-first, then the two score-blind controls, then
# the no-gate reference. The order is load-bearing for `ci_plot`, which draws `labels` as given.
GATE_MODES: tuple[GateMode, ...] = ("active", "matched_random", "random_trigger", "off")

# The controls H6 is tested against. `off` is included: it is not a *score-blind* control (it blocks
# nothing at all), but "the gate beats no gate" is the weaker claim a reader will ask for first, and
# omitting it would leave the strong claim standing on a comparison nobody had seen the baseline of.
_CONTROLS: tuple[GateMode, ...] = ("matched_random", "random_trigger", "off")

Outcome = Literal["success", "steps"]


class RQ3bConfig(BaseModel):
    """Analysis knobs. Nothing here selects an arm or a threshold - both are fixed upstream."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.05, gt=0, lt=1)
    n_boot: int = Field(default=10_000, ge=1000)
    seed: int = Field(default=0, ge=0)
    # Holm rather than BH: H6 is a small family of three pre-declared contrasts on one treatment,
    # where controlling the family-wise rate is the conventional and the more conservative choice.
    correction: Literal["holm", "bh"] = "holm"


class ModeSummary(BaseModel):
    """One arm's realised behaviour and outcome. ``block_rate`` is measured, not configured."""

    model_config = ConfigDict(extra="forbid")

    mode: GateMode
    dataset_hash: str
    n_episodes: int
    n_handoffs: int
    success_rate: float
    success_ci: tuple[float, float]
    mean_steps: float
    steps_ci: tuple[float, float]
    # The realised firing rate, which is what makes `matched_random` auditable: if it does not land
    # near the active arm's, the control was not matched and the contrast against it is not fair.
    block_rate: float
    mean_retries: float


class H6Contrast(BaseModel):
    """Gate-active against one control, on one outcome."""

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    control: GateMode
    delta: float  # active - control; positive favours the gate on success, disfavours it on steps
    delta_ci: tuple[float, float]
    cliffs_delta: float  # effect size, distribution-free, comparable across the two outcomes
    p_value: float
    p_corrected: float


class RQ3bResult(BaseModel):
    """The full H6 analysis, ready to persist and to drive the figure and the table."""

    model_config = ConfigDict(extra="forbid")

    statistic_key: str
    modes: list[ModeSummary]
    contrasts: list[H6Contrast]
    verdict: str
    figures: dict[str, str] = Field(default_factory=dict)


def rq3b_sweeps(
    base: SweepConfig,
    *,
    statistic_key: str,
    max_retries: int = 1,
    random_rate: float = 0.2,
    gate_seed: int = 0,
) -> dict[GateMode, SweepConfig]:
    """The four arms of H6 as four sweeps, identical but for ``gate``.

    ``base`` is the RQ1 subset the arms run on and must be **ungated** - deriving the arms from an
    already-gated sweep would silently inherit one arm's retry budget or control rate into all four.
    ``off`` keeps ``gate=None`` rather than ``GateConfig(mode="off")``: the two behave alike in the
    loop but hash differently, and ``None`` is what every ungated dataset recorded to date used, so
    the no-gate arm can reuse an existing RQ1 dataset instead of paying to re-run it.
    """
    if base.gate is not None:
        raise ConfigError(
            f"rq3b_sweeps needs an ungated base sweep, got gate={base.gate.mode!r}; the four arms "
            "must differ only in their gate, and a gated base would carry one arm's settings into "
            "the other three"
        )
    gated: tuple[GateMode, ...] = ("active", "matched_random", "random_trigger")
    arms: dict[GateMode, SweepConfig] = {"off": base}
    for mode in gated:
        arms[mode] = base.model_copy(
            update={
                "gate": GateConfig(
                    mode=mode,
                    statistic_key=statistic_key,
                    max_retries=max_retries,
                    random_rate=random_rate,
                    seed=gate_seed,
                )
            }
        )
    hashes = {m: dataset_hash_for(s) for m, s in arms.items()}
    if len(set(hashes.values())) != len(hashes):
        raise ConfigError(f"two RQ3b arms hash to one dataset and would pool: {hashes}")
    return {mode: arms[mode] for mode in GATE_MODES}


def _episode_frame(records: list[HandoffRecord]) -> pd.DataFrame:
    """One row per episode: terminal success, step count, blocks and retries.

    The episode is the unit of analysis because it is the unit of the intervention - the gate acts
    per handoff, but an episode either reaches the goal or does not, and handoffs inside one share
    a start pose and a trajectory.
    """
    rows: dict[str, dict[str, Any]] = {}
    for r in records:
        row = rows.setdefault(
            r.episode_id,
            {"success": False, "steps": 0, "blocks": 0, "retries": 0, "seed": r.seed},
        )
        row["success"] = row["success"] or bool(r.y_terminal_success)
        row["steps"] += 1
        row["blocks"] += int(r.gate_blocked)
        row["retries"] += r.gate_retries
    if not rows:
        raise ConfigError("RQ3b needs at least one episode per arm; got a mode with no records")
    return pd.DataFrame(rows.values())


def _summarise(mode: GateMode, d_hash: str, frame: pd.DataFrame, cfg: RQ3bConfig) -> ModeSummary:
    success = frame["success"].to_numpy(dtype=np.float64)
    steps = frame["steps"].to_numpy(dtype=np.float64)
    return ModeSummary(
        mode=mode,
        dataset_hash=d_hash,
        n_episodes=len(frame),
        n_handoffs=int(steps.sum()),
        success_rate=float(success.mean()),
        success_ci=bootstrap_ci(success, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed),
        mean_steps=float(steps.mean()),
        steps_ci=bootstrap_ci(steps, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed),
        block_rate=float(frame["blocks"].sum() / steps.sum()),
        mean_retries=float(frame["retries"].mean()),
    )


def _diff_draws(a: FloatArray, b: FloatArray, cfg: RQ3bConfig) -> FloatArray:
    """Bootstrap draws of ``mean(a) - mean(b)``, resampling each arm independently.

    Independent resampling because the arms are separate episode sets: the same seed produces the
    same start pose in both, but the trajectories diverge at the first block, so they are not paired
    observations and a paired bootstrap would understate the interval.
    """
    rng = np.random.default_rng(cfg.seed)
    return np.array(
        [
            a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
            for _ in range(cfg.n_boot)
        ],
        dtype=np.float64,
    )


def _contrast(
    outcome: Outcome, control: GateMode, a: FloatArray, b: FloatArray, cfg: RQ3bConfig
) -> H6Contrast:
    draws = _diff_draws(a, b, cfg)
    lo, hi = np.quantile(draws, [cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0])
    # Two-sided bootstrap p: twice the smaller tail mass on the wrong side of zero, floored at one
    # draw's worth so it is never reported as exactly zero, which no finite resample can establish.
    tail = min(float((draws <= 0.0).mean()), float((draws >= 0.0).mean()))
    p = min(1.0, max(2.0 * tail, 1.0 / cfg.n_boot))
    return H6Contrast(
        outcome=outcome,
        control=control,
        delta=float(a.mean() - b.mean()),
        delta_ci=(float(lo), float(hi)),
        cliffs_delta=cliffs_delta(a, b),
        p_value=p,
        p_corrected=p,  # filled in by analyse_rq3b once the whole family is known
    )


_CORRECTION_NAMES: dict[str, str] = {"holm": "Holm", "bh": "Benjamini-Hochberg"}


def _verdict(modes: list[ModeSummary], contrasts: list[H6Contrast], cfg: RQ3bConfig) -> str:
    """H6's plain-language outcome, distinguishing a null from an untestable comparison."""
    # Degeneracy is judged on the PRIMARY outcome alone. Requiring the (success, steps) PAIR to
    # match across arms looked equivalent and was not: every gated arm re-prompts where the ungated
    # one does not, so the arms' step counts differ on any real grid, and a wholly floored run -
    # the live risk, job 232980 returned 1/96 - would have slipped through as "H6 NOT SUPPORTED".
    # That is precisely the reading this branch exists to prevent, so it keys on success alone.
    rates = {m.success_rate for m in modes}
    if rates <= {0.0} or rates <= {1.0}:
        return (
            f"UNTESTABLE: terminal success was {rates.pop():.3f} in every arm, so the task "
            "produced no outcome variance for the gate to move and no contrast on it can carry "
            "evidence "
            "either way. This is a statement about the grid, not about the gate - re-run H6 on a "
            "grid whose control arm is off the floor."
        )
    beaten = {
        c.control
        for c in contrasts
        if c.outcome == "success" and c.p_corrected < cfg.alpha and c.delta > 0.0
    }
    blind = {"matched_random", "random_trigger"}
    if blind <= beaten:
        return (
            "H6 SUPPORTED: gate-active beats both score-blind controls on terminal success after "
            f"{_CORRECTION_NAMES[cfg.correction]} correction, so the improvement is attributable "
            "to blocking the *right* handoffs rather than to blocking, or to the extra sender "
            "turn, per se."
        )
    if beaten & blind:
        only = ", ".join(sorted(beaten & blind))
        return (
            f"H6 PARTIAL: gate-active beats {only} but not the remaining score-blind control, so "
            "the evidence does not separate targeted blocking from blocking at that rate."
        )
    return (
        "H6 NOT SUPPORTED: gate-active does not beat its score-blind controls on terminal success. "
        "Reported as a finding about the statistic - it does not localise the handoffs that decide "
        "the episode - rather than as a failed run."
    )


def analyse_rq3b(
    records: dict[GateMode, list[HandoffRecord]],
    hashes: dict[GateMode, str],
    *,
    statistic_key: str,
    cfg: RQ3bConfig | None = None,
) -> RQ3bResult:
    """Score the four arms and test H6 on both outcomes, Holm-corrected across the family."""
    cfg = cfg or RQ3bConfig()
    missing = [m for m in GATE_MODES if m not in records or not records[m]]
    if missing:
        raise ConfigError(
            f"RQ3b needs all four arms; {missing} have no records. An arm short of episodes is a "
            "missing control, and H6 read without one is the claim the controls exist to refuse"
        )
    frames = {m: _episode_frame(records[m]) for m in GATE_MODES}
    modes = [_summarise(m, hashes[m], frames[m], cfg) for m in GATE_MODES]

    contrasts: list[H6Contrast] = []
    outcomes: tuple[Outcome, ...] = ("success", "steps")
    for outcome in outcomes:
        active = frames["active"][outcome].to_numpy(dtype=np.float64)
        for control in _CONTROLS:
            other = frames[control][outcome].to_numpy(dtype=np.float64)
            contrasts.append(_contrast(outcome, control, active, other, cfg))
    # One family over both outcomes and all three controls: H6 is a single hypothesis and every
    # contrast below is a chance to declare it, so correcting per-outcome would leak six tests in
    # as two families of three.
    corrected = correct_pvalues(
        np.array([c.p_value for c in contrasts], dtype=np.float64), method=cfg.correction
    )
    contrasts = [
        c.model_copy(update={"p_corrected": float(p)})
        for c, p in zip(contrasts, corrected, strict=True)
    ]

    matched = next(m for m in modes if m.mode == "matched_random")
    active_mode = next(m for m in modes if m.mode == "active")
    if abs(matched.block_rate - active_mode.block_rate) > 0.05:
        logger.warning(
            "matched-random fired at %.3f against the gate's %.3f: the rate-matched control is not "
            "matched on this grid, and its contrast is weaker evidence than it appears",
            matched.block_rate,
            active_mode.block_rate,
        )
    return RQ3bResult(
        statistic_key=statistic_key,
        modes=modes,
        contrasts=contrasts,
        verdict=_verdict(modes, contrasts, cfg),
    )


def write_rq3b(result: RQ3bResult, dir: Path | str) -> Path:
    """Persist the analysis JSON, the two tables and the outcome-by-mode figures."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m.model_dump() for m in result.modes]).to_csv(dir / "rq3b_modes.csv", index=False)
    pd.DataFrame([c.model_dump() for c in result.contrasts]).to_csv(
        dir / "rq3b_contrasts.csv", index=False
    )
    (dir / "verdict.md").write_text(
        f"# RQ3b (H6): the causal gate\n\n- **Statistic:** `{result.statistic_key}`\n"
        + "".join(
            f"- **{m.mode}:** success {m.success_rate:.3f} "
            f"[{m.success_ci[0]:.3f}, {m.success_ci[1]:.3f}], "
            f"blocked {m.block_rate:.3f} of {m.n_handoffs} handoffs, "
            f"{m.mean_retries:.2f} retries/episode\n"
            for m in result.modes
        )
        + f"\n{result.verdict}\n"
    )
    labels: list[str] = [m.mode for m in result.modes]
    figs = {
        "success_by_mode": ci_plot(
            labels,
            [m.success_rate for m in result.modes],
            [m.success_ci for m in result.modes],
            ylabel="terminal success rate",
            title=f"RQ3b H6: outcome by gate mode ({result.statistic_key})",
            path=dir / "success_by_mode.png",
        ),
        "steps_by_mode": ci_plot(
            labels,
            [m.mean_steps for m in result.modes],
            [m.steps_ci for m in result.modes],
            ylabel="steps per episode",
            title=f"RQ3b H6: efficiency by gate mode ({result.statistic_key})",
            path=dir / "steps_by_mode.png",
        ),
    }
    result.figures = {k: str(v) for k, v in figs.items() if v is not None}
    (dir / "rq3b.json").write_text(result.model_dump_json(indent=2))
    return dir


def build_gate(
    mode: GateMode,
    sweep: SweepConfig,
    *,
    report: CalibrationReport,
    calibration_records: list[HandoffRecord],
    featuriser: Featuriser,
) -> RuntimeGate | None:
    """Assemble one arm's gate from a persisted calibration. ``None`` for the ungated arm.

    The statistic is **re-fitted on the calibration records**, not on the arm's own episodes, and
    the threshold comes from ``report`` unchanged. Fitting per arm would let each arm choose its own
    operating point from its own outcomes, which is the leakage the calibration step exists to
    prevent - and ``report.target`` is pinned to ``realised_failure``, never CPVI (the R5
    circularity guard), so what is imported here cannot have been tuned against the measurement it
    is about to be compared with.
    """
    if sweep.gate is None:
        return None
    cfg = sweep.gate
    calibration = next((s for s in report.statistics if s.key == cfg.statistic_key), None)
    if calibration is None:
        available = ", ".join(sorted(s.key for s in report.statistics))
        raise ConfigError(
            f"calibration report has no statistic {cfg.statistic_key!r} (has: {available}); a gate "
            "arm run without its threshold fails open and would be recorded as a gated arm"
        )
    # Only the active arm scores. Fitting a probe for a control that never calls it would spend the
    # compute and, worse, imply the control had seen the statistic.
    statistic: Statistic | None = None
    if cfg.mode == "active":
        statistic = _STATISTICS[cfg.statistic_key]()
        e_s, e_m = featuriser.featurise(calibration_records)
        statistic.fit(e_s, e_m, failure_label(calibration_records))
    return RuntimeGate(cfg, statistic=statistic, calibration=calibration, featuriser=featuriser)


def run_rq3b(
    base: SweepConfig,
    client_a: LLMClient,
    *,
    root: Path | str,
    report: CalibrationReport,
    calibration_records: list[HandoffRecord],
    featuriser: Featuriser,
    statistic_key: str = "cosine",
    max_retries: int = 1,
    random_rate: float = 0.2,
    gate_seed: int = 0,
    cfg: RQ3bConfig | None = None,
) -> RQ3bResult:
    """Run all four arms over ``base``'s grid and test H6, each arm to its own dataset.

    Sequential rather than concurrent: the arms share one endpoint, and ``run_grid`` already
    parallelises within an arm at ``sweep.concurrency``, so overlapping them would contend for the
    same GPU without shortening the wall clock.
    """
    sweeps = rq3b_sweeps(
        base,
        statistic_key=statistic_key,
        max_retries=max_retries,
        random_rate=random_rate,
        gate_seed=gate_seed,
    )
    hashes: dict[GateMode, str] = {}
    records: dict[GateMode, list[HandoffRecord]] = {}
    for mode, sweep in sweeps.items():
        d_hash = dataset_hash_for(sweep)
        logger.info("RQ3b arm %r -> dataset %s", mode, d_hash)
        gate = build_gate(
            mode,
            sweep,
            report=report,
            calibration_records=calibration_records,
            featuriser=featuriser,
        )
        run_grid(sweep, client_a, root=root, gate=gate)
        hashes[mode] = d_hash
        records[mode] = load_records(d_hash, root=root)
    return analyse_rq3b(records, hashes, statistic_key=statistic_key, cfg=cfg)
