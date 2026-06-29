"""Pilot go/no-go gates G1/G2/G3 and the fallback report (DSE-019).

The three hard gates that decide whether the headline T-transport task is viable before the RQ1
sweep runs (roadmap §3.1): G1 capability (can self-play even solve the clean channel?), G2 signal
(a measurable C0-to-hard gap in *both* outcome and CPVI?), G3 groundedness (do messages describe
the true geometry, not hallucinate it?). Each returns a ``GateResult``; ``run_pilot`` rolls them
into a ``PilotReport`` whose recommendation is proceed / retune-once / invoke-fallback, with the
fallback ladder spelled out so a failing pilot triggers the documented pivot, not a scramble. The
pilot is allowed exactly one retune (``attempt``), per the roadmap.

Gates are stateless functions on the loaded handoff records (CLAUDE.md function-over-class); only
G2 needs the featuriser (for the CPVI gap).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.config import ConfigError
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.measure.featuriser import Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, cpvi

logger = logging.getLogger(__name__)

Recommendation = Literal["proceed", "retune_once", "fallback"]

# The documented pivot if a gate still fails after the one allowed retune (roadmap G-gate fallback).
FALLBACK_LADDER = (
    "1. Elevate RQ3a (Who&When / MAST external validity) to the headline - it is the pre-planned "
    "fallback and can carry the dissertation alone.\n"
    "2. Simplify the task (wider slits, fewer chambers, or a shorter horizon) and re-pilot once.\n"
    "3. Reframe RQ1 as a diagnostic negative: report the absence of an information gradient as the "
    "finding rather than chasing a positive."
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")  # ints and decimals, signed; the geometry tokens in a message


class PilotConfig(BaseModel):
    """Configurable floors and gaps for the three gates; the pilot may retune these once."""

    model_config = ConfigDict(extra="forbid")

    g1_success_floor: float = Field(default=0.5, ge=0, le=1)  # C0 self-play must clear this
    g2_min_success_gap: float = Field(default=0.1, ge=0)  # C0 minus hard success rate (magnitude)
    # Directional only: CPVI is in bits and its scale is uncalibrated until the pilot runs,
    # so a magnitude floor cannot be honestly pre-specified here. The 0.1 success-gap above carries
    # the magnitude claim; this gate asks the CPVI gradient merely points the right way. Re-set this
    # to a pre-registered positive floor once the pilot reveals the CPVI bit-scale (roadmap G2).
    g2_min_cpvi_gap: float = Field(default=0.0)  # C0 minus hard mean CPVI (bits); see note above
    g3_grounding_floor: float = Field(default=0.8, ge=0, le=1)  # fraction of grounded mentions
    g3_abs_tol: float = Field(default=0.5, gt=0)  # a message number within this of a true one ...
    g3_rel_tol: float = Field(default=0.05, ge=0)  # ... or this fraction of it, is grounded
    min_seeds_for_proceed: int = Field(default=3, ge=1)  # proceed needs >= this many seeds
    cpvi_probe: ProbeConfig = Field(default_factory=ProbeConfig)


class GateResult(BaseModel):
    """One gate's verdict: the headline value against its threshold, plus numeric detail."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    value: float
    threshold: float
    detail: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class PilotReport(BaseModel):
    """The three gates plus the recommended action and the fallback ladder, ready to persist."""

    model_config = ConfigDict(extra="forbid")

    dataset_hash: str
    n_episodes: int
    n_seeds: int
    attempt: int
    gates: list[GateResult]
    recommendation: Recommendation
    recommendation_note: str = ""  # why a verdict was held back (e.g. too few seeds to proceed)
    fallback_ladder: str = FALLBACK_LADDER


def _require_labelled(records: list[HandoffRecord]) -> None:
    # ponytail: local 3-line guard, not an import from gate.statistics - keeps the Phase-1 pilot
    # decoupled from the Phase-5 gate module. Treating None as False would silently fail episodes.
    if any(r.y_terminal_success is None for r in records):
        raise ConfigError("pilot needs labelled episodes; run the DSE-009 labeller first")


def _episode_success(records: list[HandoffRecord]) -> dict[str, bool]:
    """Per-episode terminal success: True if the episode reaches the goal at any handoff."""
    out: dict[str, bool] = {}
    for r in records:
        out[r.episode_id] = out.get(r.episode_id, False) or bool(r.y_terminal_success)
    return out


def _episode_condition(records: list[HandoffRecord]) -> dict[str, Condition]:
    return {r.episode_id: r.condition for r in records}


def _condition_rank(c: Condition) -> int:
    return int(c[1])  # "C3" -> 3; the channel-degradation order C0 < ... < C4


def _hardest_condition(records: list[HandoffRecord]) -> Condition:
    present = {r.condition for r in records}
    hardest = max(present, key=_condition_rank)
    if hardest == "C0":
        raise ConfigError("G2 needs a degraded condition to contrast against C0; only C0 present")
    return hardest


def _groups(records: list[HandoffRecord]) -> NDArray[np.int_]:
    return np.unique([r.episode_id for r in records], return_inverse=True)[1].astype(int)


def _success_per_handoff(records: list[HandoffRecord]) -> NDArray[np.int_]:
    success = _episode_success(records)
    return np.array([1 if success[r.episode_id] else 0 for r in records], dtype=int)


def g1_capability(records: list[HandoffRecord], cfg: PilotConfig) -> GateResult:
    """G1: the C0 self-play episode success rate must clear ``g1_success_floor``."""
    _require_labelled(records)
    success = _episode_success(records)
    condition = _episode_condition(records)
    c0 = [ep for ep, c in condition.items() if c == "C0"]
    if not c0:
        raise ConfigError("G1 needs C0 episodes; none present in the dataset")
    rate = float(np.mean([success[ep] for ep in c0]))
    return GateResult(
        name="G1 capability",
        passed=rate >= cfg.g1_success_floor,
        value=rate,
        threshold=cfg.g1_success_floor,
        detail={
            "n_c0_episodes": float(len(c0)),
            "n_c0_success": float(sum(success[e] for e in c0)),
        },
    )


def g2_signal(records: list[HandoffRecord], featuriser: Featuriser, cfg: PilotConfig) -> GateResult:
    """G2: a C0-to-hard gap in *both* outcome (success rate) and CPVI (held-out, bits)."""
    _require_labelled(records)
    hard = _hardest_condition(records)
    success = _episode_success(records)
    condition = _episode_condition(records)
    c0_eps = [e for e, c in condition.items() if c == "C0"]
    hard_eps = [e for e, c in condition.items() if c == hard]
    if not c0_eps or not hard_eps:
        raise ConfigError("G2 needs both C0 and the hardest condition present")
    c0_rate = float(np.mean([success[e] for e in c0_eps]))
    hard_rate = float(np.mean([success[e] for e in hard_eps]))
    success_gap = c0_rate - hard_rate

    subset = [r for r in records if r.condition in {"C0", hard}]
    y = _success_per_handoff(subset)
    if len(np.unique(y)) < 2:
        return GateResult(
            name="G2 signal",
            passed=False,
            value=success_gap,
            threshold=cfg.g2_min_success_gap,
            detail={"c0_success": c0_rate, "hard_success": hard_rate, "success_gap": success_gap},
            note=f"hard={hard}; CPVI gap unmeasurable (single outcome class in the C0+hard subset)",
        )
    e_s, e_m = featuriser.featurise(subset)
    scores = cpvi(e_s, e_m, y, _groups(subset), cfg.cpvi_probe)
    c0_mask = np.array([r.condition == "C0" for r in subset])
    cpvi_gap = float(np.mean(scores[c0_mask]) - np.mean(scores[~c0_mask]))

    passed = success_gap >= cfg.g2_min_success_gap and cpvi_gap >= cfg.g2_min_cpvi_gap
    return GateResult(
        name="G2 signal",
        passed=passed,
        value=success_gap,
        threshold=cfg.g2_min_success_gap,
        detail={
            "c0_success": c0_rate,
            "hard_success": hard_rate,
            "success_gap": success_gap,
            "cpvi_gap": cpvi_gap,
            "c0_mean_cpvi": float(np.mean(scores[c0_mask])),
            "hard_mean_cpvi": float(np.mean(scores[~c0_mask])),
            "min_cpvi_gap": cfg.g2_min_cpvi_gap,
        },
        note=f"hard={hard}; gate requires both the success gap and the CPVI gap to clear",
    )


def _numeric_leaves(obj: object) -> list[float]:
    """Every numeric scalar in a nested state payload (bools excluded - they are not geometry)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _numeric_leaves(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _numeric_leaves(x)]
    return []


def _record_grounding(rec: HandoffRecord, cfg: PilotConfig) -> float:
    """Fraction of the message's numbers that match a true state number within tolerance.

    1.0 when the message cites no numbers (it claims no geometry, so it cannot hallucinate any).
    """
    mentioned = [float(m) for m in _NUM.findall(rec.message_delivered)]
    if not mentioned:
        return 1.0
    truth = _numeric_leaves(rec.state)
    grounded = sum(
        any(abs(m - t) <= max(cfg.g3_abs_tol, cfg.g3_rel_tol * abs(t)) for t in truth)
        for m in mentioned
    )
    return grounded / len(mentioned)


def g3_groundedness(records: list[HandoffRecord], cfg: PilotConfig) -> GateResult:
    """G3: the mean message-grounding (numbers cited vs the structured state) clears the floor.

    ponytail: number-matching grounds the load-pose mentions the structured ``state`` carries; it
    cannot catch a directional lie or ground the goal/slit (not persisted per-record). Upgrade path:
    per-serialisation entity parsing against the full SceneState if the heuristic proves too coarse.
    """
    grounding = np.array([_record_grounding(r, cfg) for r in records], dtype=np.float64)
    value = float(np.mean(grounding)) if len(grounding) else 1.0
    n_with_numbers = sum(bool(_NUM.search(r.message_delivered)) for r in records)
    return GateResult(
        name="G3 groundedness",
        passed=value >= cfg.g3_grounding_floor,
        value=value,
        threshold=cfg.g3_grounding_floor,
        detail={"n_records": float(len(records)), "n_with_numbers": float(n_with_numbers)},
    )


def _recommendation(
    gates: list[GateResult], attempt: int, n_seeds: int, cfg: PilotConfig
) -> tuple[Recommendation, str]:
    """Verdict plus the reason it was held back, if any. Gates first, then the seed-count floor."""
    if not all(g.passed for g in gates):
        return ("retune_once" if attempt <= 1 else "fallback"), ""  # one retune allowed, then pivot
    if n_seeds < cfg.min_seeds_for_proceed:
        return "retune_once", (
            f"all gates pass but only {n_seeds} seed(s) ran (< {cfg.min_seeds_for_proceed} needed "
            "to proceed): a single-/few-seed pass is LLM noise, not a stable gradient. Add seeds "
            "and re-pilot before greenlighting the full sweep."
        )
    return "proceed", ""


def run_pilot(
    records: list[HandoffRecord],
    featuriser: Featuriser,
    *,
    cfg: PilotConfig | None = None,
    dataset_hash: str = "",
    attempt: int = 1,
) -> PilotReport:
    """Run all three gates over a (small) sweep's records and assemble the go/no-go report."""
    cfg = cfg or PilotConfig()
    if not records:
        raise ConfigError("run_pilot called with no records")
    gates = [
        g1_capability(records, cfg),
        g2_signal(records, featuriser, cfg),
        g3_groundedness(records, cfg),
    ]
    n_seeds = len({r.seed for r in records})
    recommendation, note = _recommendation(gates, attempt, n_seeds, cfg)
    report = PilotReport(
        dataset_hash=dataset_hash,
        n_episodes=len(_episode_success(records)),
        n_seeds=n_seeds,
        attempt=attempt,
        gates=gates,
        recommendation=recommendation,
        recommendation_note=note,
    )
    logger.info("pilot %s: %s", dataset_hash or "(unnamed)", report.recommendation)
    return report


_ACTION_TEXT: dict[Recommendation, str] = {
    "proceed": "PROCEED - all gates pass; run the RQ1 sweep.",
    "retune_once": "RETUNE ONCE - a gate failed; apply the one allowed retune, then re-pilot.",
    "fallback": "INVOKE FALLBACK - a gate still failed after the retune; take the ladder below.",
}


def render_report(report: PilotReport) -> str:
    """A one-page Markdown report: recommendation, a gate table, detail, and the fallback ladder."""
    lines = [
        "# Pilot gate report (G1/G2/G3)",
        "",
        f"- dataset: `{report.dataset_hash or '(unnamed)'}`",
        f"- episodes: {report.n_episodes} | seeds: {report.n_seeds} | attempt: {report.attempt}",
        f"- **recommendation: {_ACTION_TEXT[report.recommendation]}**",
        *([f"- _{report.recommendation_note}_"] if report.recommendation_note else []),
        "",
        "| Gate | Pass | Value | Threshold |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.value:.3f} | {g.threshold:.3f} |"
        for g in report.gates
    ]
    lines.append("")
    for g in report.gates:
        detail = ", ".join(f"{k}={v:.3f}" for k, v in g.detail.items())
        lines.append(f"- **{g.name}**: {detail}" + (f" — {g.note}" if g.note else ""))
    lines += [
        "",
        "## Fallback ladder (if a gate fails after the one retune)",
        "",
        report.fallback_ladder,
    ]
    return "\n".join(lines)


def write_pilot_report(report: PilotReport, dir: Path | str) -> Path:
    """Persist the report as both JSON (machine) and Markdown (the one-pager). Returns the dir."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "pilot.json").write_text(report.model_dump_json(indent=2))
    (dir / "pilot.md").write_text(render_report(report))
    return dir
