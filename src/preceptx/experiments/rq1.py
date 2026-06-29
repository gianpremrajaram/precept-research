"""RQ1 information-gradient driver and analysis (DSE-020): the headline result.

Assemble the factorial (C0-C4 x serialisation x difficulty x seed), run self-play through the
runner, then analyse: per-condition outcome and CPVI (always with the PVI-minus-CPVI gap, never bare
message value), the H1 inferential model of outcome on condition, and the H2 mediation of that
effect through CPVI (roadmap §3.2).

Two models, deliberately at two levels:

- **H1 (handoff level).** A linear probability MixedLM of per-step progress on condition, with seed
  as the group random intercept and episode a variance component within it - the only level where
  random effects for *both* seed and episode fit one model (the episode id encodes the seed). Its
  Holm-corrected condition coefficients back the contrasts.
- **H2 (episode level).** Mediation is tested on the *headline* outcome - episode success - with the
  mediator aggregated to per-episode mean CPVI, so the DV matches H1 and within- vs between-episode
  CPVI variance is not conflated (the "easier episodes happen to carry more CPVI" confound). Full
  Baron-Kenny: path a (condition->CPVI), path b/c' (success~condition+CPVI), total c, and the
  indirect effect a*b per condition with a bootstrap CI. The handoff-level CPVI attenuation is kept
  as a within-episode diagnostic, not the H2 test.

ponytail: both fits are linear probability MixedLMs (statsmodels) on the binary outcomes - the lazy,
AC-satisfying fit; upgrade path b to a logistic GLMM with a delta-method indirect effect if fitted
probabilities stray out of [0,1]. The indirect-effect CI is a percentile bootstrap over episodes
(n_boot_mediation refits); cluster-resample seeds if seed clustering ever dominates.

``analyse_rq1`` is the analysis core (fixture-testable with no runner); ``run_rq1`` is the grid run
plus that analysis.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.analysis.figures import ci_plot
from preceptx.analysis.stats import (
    SeedSensitivity,
    bootstrap_ci,
    cliffs_delta,
    correct_pvalues,
    seed_sensitivity,
)
from preceptx.config import ConfigError, ModelConfig
from preceptx.data.schema import Condition, Difficulty, HandoffRecord, Serialisation
from preceptx.data.writer import dataset_hash, load_records
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, sweep_hash
from preceptx.measure.featuriser import Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, cpvi, pvi
from preceptx.serving.client import LLMClient

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

CONDITION_ORDER: list[Condition] = ["C0", "C1", "C2", "C3", "C4"]


class RQ1Config(BaseModel):
    """Analysis knobs: the CPVI probe, the bootstrap, and the contrast correction."""

    model_config = ConfigDict(extra="forbid")

    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    n_boot: int = Field(default=2000, ge=100)  # for the cheap one-sample/effect-size CIs
    n_boot_mediation: int = Field(default=400, ge=50)  # model-refit bootstrap; costlier per draw
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["holm", "bh"] = "holm"


class ConditionSummary(BaseModel):
    """Per-condition rollup: outcome, efficiency, and CPVI with its PVI gap (all with intervals)."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    n_episodes: int
    n_handoffs: int
    success_rate: float
    success_ci: tuple[float, float]
    mean_steps: float
    mean_collisions: float
    mean_cpvi: float
    cpvi_ci: tuple[float, float]
    mean_pvi: float
    pvi_cpvi_gap: float  # apparent message value that was just an echo of the shared state


class Contrast(BaseModel):
    """A Ck-vs-C0 contrast: effect size with a CI, plus the mixed-model coefficient and its p."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    cliffs_delta: float  # on episode success vs C0
    delta_ci: tuple[float, float]
    mixed_coef: float
    p_raw: float
    p_corrected: float


class EpisodeMediation(BaseModel):
    """One Ck-vs-C0 mediation path set: the channel effect decomposed through episode-mean CPVI."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    path_a: float  # condition -> episode-mean CPVI
    indirect: float  # a * b: the channel effect on success carried *through* CPVI
    indirect_ci: tuple[float, float]  # percentile bootstrap over episodes
    direct: float  # c': condition -> success, controlling for CPVI
    total: float  # c : condition -> success, unadjusted
    prop_mediated: float  # indirect / total (0 when total == 0)


class MixedModelSummary(BaseModel):
    """The H1 handoff model, the H2 episode-level mediation, and the within-episode diagnostic."""

    model_config = ConfigDict(extra="forbid")

    formula: str
    coef_no_mediator: dict[str, float]  # H1 handoff-level condition fixed effects
    converged: bool  # H1 handoff model
    mediation_outcome: str  # the H2 DV - "episode_success"
    path_b: float  # episode-mean CPVI -> success, controlling for condition (shared across Ck)
    mediations: list[EpisodeMediation]
    mediation_converged: bool  # path-a and full-outcome episode models both converged
    diagnostic_cpvi_coef: float  # within-episode: per-handoff CPVI -> progress
    diagnostic_attenuation: float  # mean shrink of handoff condition coefs when CPVI enters
    mediation_note: str


class RQ1Result(BaseModel):
    """The full RQ1 analysis, ready to persist and to drive the figures/table."""

    model_config = ConfigDict(extra="forbid")

    dataset_hash: str
    n_handoffs: int
    conditions: list[ConditionSummary]
    contrasts: list[Contrast]
    mixed_model: MixedModelSummary
    seed_sensitivity: SeedSensitivity
    figures: dict[str, str] = Field(default_factory=dict)


def rq1_sweep(
    model: ModelConfig,
    *,
    seeds: list[int],
    serialisations: list[Serialisation] | None = None,
    difficulties: list[Difficulty] | None = None,
    conditions: list[Condition] | None = None,
    max_steps: int = 12,
) -> SweepConfig:
    """The RQ1 factorial: all of C0-C4 by default, crossed with serialisation/difficulty/seed."""
    return SweepConfig(
        conditions=conditions or CONDITION_ORDER,
        serialisations=serialisations or ["numeric"],
        difficulties=difficulties or ["hard"],
        seeds=seeds,
        model=model,
        max_steps=max_steps,
    )


def _require_progress_labels(records: list[HandoffRecord]) -> IntArray:
    """Per-handoff progress outcome (the mixed-model response); fail loud if unlabelled."""
    if any(r.y_binary_progress is None for r in records):
        raise ConfigError("RQ1 analysis needs y_binary_progress; run the DSE-009 labeller first")
    return np.array([1 if r.y_binary_progress else 0 for r in records], dtype=int)


def _episode_frame(records: list[HandoffRecord]) -> pd.DataFrame:
    """One row per episode: condition, seed, terminal success, step count, collision count."""
    rows: dict[str, dict[str, Any]] = {}
    for r in records:
        row = rows.setdefault(
            r.episode_id,
            {
                "condition": r.condition,
                "seed": r.seed,
                "success": False,
                "steps": 0,
                "collisions": 0,
            },
        )
        row["success"] = row["success"] or bool(r.y_terminal_success)
        row["steps"] += 1
        row["collisions"] += int(r.collision)
    return pd.DataFrame(rows.values())


def _groups(records: list[HandoffRecord]) -> IntArray:
    return np.unique([r.episode_id for r in records], return_inverse=True)[1].astype(int)


def _condition_summary(
    cond: Condition,
    records: list[HandoffRecord],
    cpvi_scores: FloatArray,
    pvi_scores: FloatArray,
    ep_frame: pd.DataFrame,
    cfg: RQ1Config,
) -> ConditionSummary:
    mask = np.array([r.condition == cond for r in records])
    ep = ep_frame[ep_frame["condition"] == cond]
    succ = ep["success"].to_numpy(dtype=np.float64)
    cpvi_c = cpvi_scores[mask]
    return ConditionSummary(
        condition=cond,
        n_episodes=len(ep),
        n_handoffs=int(mask.sum()),
        success_rate=float(succ.mean()),
        success_ci=bootstrap_ci(succ, n_boot=cfg.n_boot, alpha=cfg.alpha),
        mean_steps=float(ep["steps"].mean()),
        mean_collisions=float(ep["collisions"].mean()),
        mean_cpvi=float(cpvi_c.mean()),
        cpvi_ci=bootstrap_ci(cpvi_c, n_boot=cfg.n_boot, alpha=cfg.alpha),
        mean_pvi=float(pvi_scores[mask].mean()),
        pvi_cpvi_gap=float(pvi_scores[mask].mean() - cpvi_c.mean()),
    )


def _delta_ci(a: FloatArray, b: FloatArray, cfg: RQ1Config) -> tuple[float, float]:
    """Two-sample bootstrap CI for Cliff's delta (resample both groups)."""
    rng = np.random.default_rng(0)
    boots = [
        cliffs_delta(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
        for _ in range(cfg.n_boot)
    ]
    lo, hi = np.quantile(boots, [cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0])
    return float(lo), float(hi)


def _fit_mixed(
    df: pd.DataFrame, formula: str, *, vc: dict[str, str] | None = None, quiet: bool = False
) -> tuple[Any, bool]:
    """Fit a seed-grouped MixedLM (optional episode variance component); return (result, converged).

    statsmodels raises convergence (and other) warnings on small or stiff fits; we capture them so
    they surface as WARNING log lines - a degraded mode, not a crash - rather than propagating, and
    thread ``converged`` into the persisted summary so a non-converged fit is auditable, never
    silent. ``quiet`` mutes the log lines for the inner bootstrap refits (expected, and hundreds
    of them); the run still fails loud on real errors.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.mixedlm(formula, df, groups=df["seed"], vc_formula=vc).fit()
    if not quiet:
        for w in caught:
            logger.warning("RQ1 mixed model fit (%s): %s", w.category.__name__, w.message)
    return result, bool(getattr(result, "converged", True))


def _condition_terms(params: Any) -> dict[str, float]:
    """The ``C(condition)[T.Ck]`` fixed-effect coefficients, keyed by the bare condition."""
    return {
        idx.split("T.")[1].rstrip("]"): float(params[idx])
        for idx in params.index
        if idx.startswith("C(condition)[T.")
    }


def _handoff_model(
    df: pd.DataFrame,
) -> tuple[dict[str, float], float, float, bool, dict[str, tuple[float, float]]]:
    """H1 handoff LPM (seed RE, episode VC) plus the within-episode CPVI attenuation diagnostic.

    Returns the condition fixed effects, the per-handoff CPVI coefficient, the mean attenuation of
    those condition coefs when CPVI enters, the H1 convergence flag, and per-condition (coef, p).
    """
    vc = {"episode": "0 + C(episode)"}
    base, converged = _fit_mixed(df, "y ~ C(condition)", vc=vc)
    mediated, _ = _fit_mixed(df, "y ~ C(condition) + cpvi", vc=vc)
    coef_no = _condition_terms(base.params)
    coef_with = _condition_terms(mediated.params)
    pvals = {k: float(base.pvalues[f"C(condition)[T.{k}]"]) for k in coef_no}
    shrink = [abs(coef_with[k]) / abs(coef_no[k]) for k in coef_no if coef_no[k] != 0.0]
    attenuation = 1.0 - float(np.mean(shrink)) if shrink else 0.0
    cpvi_coef = float(mediated.params["cpvi"])
    return coef_no, cpvi_coef, attenuation, converged, {k: (coef_no[k], pvals[k]) for k in coef_no}


def _episode_mediation_frame(records: list[HandoffRecord], cpvi_scores: FloatArray) -> pd.DataFrame:
    """One row per episode: condition, seed, binary terminal success, and mean per-handoff CPVI."""
    rows: dict[str, dict[str, Any]] = {}
    for r, s in zip(records, cpvi_scores, strict=True):
        row = rows.setdefault(
            r.episode_id, {"condition": r.condition, "seed": r.seed, "success": False, "_cpvi": []}
        )
        row["success"] = row["success"] or bool(r.y_terminal_success)
        row["_cpvi"].append(float(s))
    return pd.DataFrame(
        [
            {
                "condition": v["condition"],
                "seed": v["seed"],
                "success": 1 if v["success"] else 0,
                "cpvi": float(np.mean(v["_cpvi"])),
            }
            for v in rows.values()
        ]
    )


def _bootstrap_indirect(
    ep_df: pd.DataFrame, targets: Sequence[str], cfg: RQ1Config
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI of the indirect effect a*b per condition, resampling episodes.

    Each draw refits path a (cpvi~condition) and path b (success~condition+cpvi) on the resampled
    episodes; degenerate draws - a dropped condition (which would shift the C0 reference) or a
    single-class outcome (unfittable) - are skipped. ponytail: plain episode resample,
    n_boot_mediation refits; cluster-resample seeds only if seed clustering dominates the variance.
    """
    rng = np.random.default_rng(0)
    n = len(ep_df)
    n_conditions = ep_df["condition"].nunique()
    draws: dict[str, list[float]] = {c: [] for c in targets}
    for _ in range(cfg.n_boot_mediation):
        boot = ep_df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        if boot["success"].nunique() < 2 or boot["condition"].nunique() < n_conditions:
            continue
        try:
            a_model, _ = _fit_mixed(boot, "cpvi ~ C(condition)", quiet=True)
            y_model, _ = _fit_mixed(boot, "success ~ C(condition) + cpvi", quiet=True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        a = _condition_terms(a_model.params)
        b = float(y_model.params.get("cpvi", 0.0))
        for c in targets:
            draws[c].append(a.get(c, 0.0) * b)
    out: dict[str, tuple[float, float]] = {}
    for c in targets:
        vals = np.array(draws[c], dtype=np.float64)
        if len(vals) < 2:
            out[c] = (float("nan"), float("nan"))
        else:
            lo, hi = np.quantile(vals, [cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0])
            out[c] = (float(lo), float(hi))
    return out


def _episode_mediation(
    ep_df: pd.DataFrame, present: list[Condition], cfg: RQ1Config
) -> tuple[list[EpisodeMediation], float, bool]:
    """Episode-level Baron-Kenny: paths a, b, c, c' and the bootstrapped indirect effect per Ck."""
    targets = [c for c in present if c != "C0"]
    if ep_df["success"].nunique() < 2:  # single-class outcome: mediation unmeasurable (cf. G2)
        nan = float("nan")
        meds = [
            EpisodeMediation(
                condition=c,
                path_a=nan,
                indirect=nan,
                indirect_ci=(nan, nan),
                direct=nan,
                total=nan,
                prop_mediated=nan,
            )
            for c in targets
        ]
        return meds, nan, False
    a_model, a_conv = _fit_mixed(ep_df, "cpvi ~ C(condition)")
    y_full, yf_conv = _fit_mixed(ep_df, "success ~ C(condition) + cpvi")
    y_red, _ = _fit_mixed(ep_df, "success ~ C(condition)")
    a = _condition_terms(a_model.params)
    cprime = _condition_terms(y_full.params)
    total = _condition_terms(y_red.params)
    b = float(y_full.params["cpvi"])
    ci = _bootstrap_indirect(ep_df, targets, cfg)
    meds = [
        EpisodeMediation(
            condition=c,
            path_a=a.get(c, 0.0),
            indirect=a.get(c, 0.0) * b,
            indirect_ci=ci[c],
            direct=cprime.get(c, 0.0),
            total=total.get(c, 0.0),
            prop_mediated=(a.get(c, 0.0) * b / total[c]) if total.get(c, 0.0) != 0.0 else 0.0,
        )
        for c in targets
    ]
    return meds, b, bool(a_conv and yf_conv)


def _mixed_model(
    handoff_df: pd.DataFrame, ep_df: pd.DataFrame, present: list[Condition], cfg: RQ1Config
) -> tuple[MixedModelSummary, dict[str, tuple[float, float]]]:
    """Assemble the H1 handoff model and the H2 episode mediation into the persisted summary."""
    coef_no, cpvi_coef, attenuation, converged, coef_p = _handoff_model(handoff_df)
    mediations, path_b, med_conv = _episode_mediation(ep_df, present, cfg)
    finite_indirect = [m.indirect for m in mediations if np.isfinite(m.indirect)]
    mean_indirect = float(np.mean(finite_indirect)) if finite_indirect else float("nan")
    summary = MixedModelSummary(
        formula="H1: y ~ C(condition) [groups=seed, vc=episode]; "
        "H2: success ~ C(condition) + cpvi_epmean [groups=seed]",
        coef_no_mediator=coef_no,
        converged=converged,
        mediation_outcome="episode_success",
        path_b=path_b,
        mediations=mediations,
        mediation_converged=med_conv,
        diagnostic_cpvi_coef=cpvi_coef,
        diagnostic_attenuation=attenuation,
        mediation_note=(
            f"H2 (episode level): success mediated by CPVI; mean indirect effect a*b over Ck = "
            f"{mean_indirect:.3f} (negative = the channel suppresses success by lowering CPVI). "
            f"Within-episode diagnostic: handoff condition coefs attenuate {attenuation:.0%} "
            f"when per-handoff CPVI enters."
        ),
    )
    return summary, coef_p


def analyse_rq1(
    records: list[HandoffRecord],
    featuriser: Featuriser,
    *,
    dataset_hash: str,
    cfg: RQ1Config | None = None,
) -> RQ1Result:
    """Score CPVI/PVI, summarise per condition, fit the mixed model + mediation, build contrasts."""
    cfg = cfg or RQ1Config()
    if not records:
        raise ConfigError("analyse_rq1 called with no records")
    y = _require_progress_labels(records)
    if len(np.unique(y)) < 2:
        raise ConfigError("RQ1 needs both progress classes to fit the model and estimate CPVI")

    e_s, e_m = featuriser.featurise(records)
    groups = _groups(records)
    cpvi_scores = cpvi(e_s, e_m, y, groups, cfg.probe)
    pvi_scores = pvi(e_m, y, groups, cfg.probe)
    ep_frame = _episode_frame(records)

    present = [c for c in CONDITION_ORDER if any(r.condition == c for r in records)]
    summaries = [
        _condition_summary(c, records, cpvi_scores, pvi_scores, ep_frame, cfg) for c in present
    ]

    model_df = pd.DataFrame(
        {
            "y": y,
            "condition": [r.condition for r in records],
            "seed": [r.seed for r in records],
            "episode": [r.episode_id for r in records],
            "cpvi": cpvi_scores,
        }
    )
    ep_med_df = _episode_mediation_frame(records, cpvi_scores)
    mixed, coef_p = _mixed_model(model_df, ep_med_df, present, cfg)

    contrasts = _contrasts(present, ep_frame, coef_p, cfg)
    # Seed sensitivity on the *gradient*, not a collapsed metric: per-seed C0-minus-hardest success
    # gap, so its spread answers "is the C0->C4 ordering seed-stable?" (the thesis question), rather
    # not "does overall success vary across seeds?" (which it always does, from LLM nondeterminism).
    hardest = present[-1]
    seeds_metric: dict[int, float] = {}
    for s in sorted(ep_frame["seed"].unique()):
        sub = ep_frame[ep_frame["seed"] == s]
        c0 = sub[sub["condition"] == "C0"]["success"]
        hard = sub[sub["condition"] == hardest]["success"]
        if len(c0) and len(hard):
            seeds_metric[int(s)] = float(c0.mean() - hard.mean())
    return RQ1Result(
        dataset_hash=dataset_hash,
        n_handoffs=len(records),
        conditions=summaries,
        contrasts=contrasts,
        mixed_model=mixed,
        seed_sensitivity=seed_sensitivity(seeds_metric),
    )


def _contrasts(
    present: list[Condition],
    ep_frame: pd.DataFrame,
    coef_p: dict[str, tuple[float, float]],
    cfg: RQ1Config,
) -> list[Contrast]:
    """Ck-vs-C0 effect sizes (Cliff's delta on success) with corrected mixed-model p-values."""
    if "C0" not in present:
        raise ConfigError("RQ1 contrasts need C0 as the reference condition")
    c0 = ep_frame[ep_frame["condition"] == "C0"]["success"].to_numpy(dtype=np.float64)
    targets = [c for c in present if c != "C0"]
    raw_p = np.array([coef_p[c][1] for c in targets], dtype=np.float64)
    corrected = correct_pvalues(raw_p, method=cfg.correction)
    out: list[Contrast] = []
    for c, p_corr in zip(targets, corrected, strict=True):
        ck = ep_frame[ep_frame["condition"] == c]["success"].to_numpy(dtype=np.float64)
        out.append(
            Contrast(
                condition=c,
                cliffs_delta=cliffs_delta(ck, c0),  # negative = Ck worse than C0 (degradation)
                delta_ci=_delta_ci(ck, c0, cfg),
                mixed_coef=coef_p[c][0],
                p_raw=coef_p[c][1],
                p_corrected=float(p_corr),
            )
        )
    return out


def run_rq1(
    sweep: SweepConfig,
    client: LLMClient,
    featuriser: Featuriser,
    *,
    root: Path | str,
    cfg: RQ1Config | None = None,
) -> RQ1Result:
    """Run the RQ1 grid and analyse it end to end (full-scale run gated on DSE-005 compute)."""
    run_grid(sweep, client, root=root)
    d_hash = dataset_hash(sweep_hash(sweep))
    return analyse_rq1(load_records(d_hash, root=root), featuriser, dataset_hash=d_hash, cfg=cfg)


def write_rq1(result: RQ1Result, dir: Path | str) -> Path:
    """Persist the analysis JSON, a per-condition results table (CSV), and the two figures."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "rq1.json").write_text(result.model_dump_json(indent=2))
    table = pd.DataFrame([c.model_dump() for c in result.conditions])
    table.to_csv(dir / "rq1_results.csv", index=False)

    labels = [c.condition for c in result.conditions]
    out = ci_plot(
        labels,
        [c.success_rate for c in result.conditions],
        [c.success_ci for c in result.conditions],
        ylabel="episode success rate",
        title="RQ1: outcome vs condition",
        path=dir / "outcome_vs_condition.png",
    )
    cpvi_out = ci_plot(
        labels,
        [c.mean_cpvi for c in result.conditions],
        [c.cpvi_ci for c in result.conditions],
        ylabel="mean CPVI (bits)",
        title="RQ1: CPVI vs condition",
        path=dir / "cpvi_vs_condition.png",
    )
    if out is not None and cpvi_out is not None:  # both render or neither (viz extra present)
        result.figures = {"outcome": str(out), "cpvi": str(cpvi_out)}
        (dir / "rq1.json").write_text(result.model_dump_json(indent=2))  # rewrite with figure paths
    return dir
