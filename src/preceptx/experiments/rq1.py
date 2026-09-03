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
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from preceptx.analysis.figures import ci_plot, series_plot
from preceptx.analysis.stats import (
    AnalysisProvenance,
    OverlapRestrictedContrast,
    SeedSensitivity,
    bootstrap_ci,
    build_provenance,
    cliffs_delta,
    cluster_bootstrap_ci,
    correct_pvalues,
    overlap_restricted_contrast,
    partial_spearman,
    seed_sensitivity,
)
from preceptx.config import ConfigError, ModelConfig
from preceptx.data.schema import Condition, Difficulty, HandoffRecord, Serialisation
from preceptx.data.writer import load_records
from preceptx.experiments.runner import run_grid
from preceptx.experiments.sweep import SweepConfig, dataset_hash_for
from preceptx.measure.featuriser import Featuriser
from preceptx.measure.pvi_cpvi import (
    ProbeConfig,
    control_task_cpvi,
    cpvi_with_sd,
    pvi,
    shuffled_message_cpvi,
)
from preceptx.serving.client import LLMClient
from preceptx.sim.feasibility import STEP_BUDGETS, oracle_action

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

CONDITION_ORDER: list[Condition] = ["C0", "C1", "C2", "C3", "C4"]
# The two macro actions that turn the load. Named once: the oracle wants a rotation on ~95% of
# handoffs, so "did B rotate, and which way" carries almost all of the agreement signal.
_ROTATIONS = ("ROT+", "ROT-")


class RQ1Config(BaseModel):
    """Analysis knobs: the CPVI probe, the bootstrap, and the contrast correction."""

    model_config = ConfigDict(extra="forbid")

    # R = 5 repeated cross-fits (PREREGISTRATION §5); repeat 0 is the canonical fold
    # assignment, so n_repeats=1 still reproduces the unrepeated estimator exactly.
    probe: ProbeConfig = Field(default_factory=lambda: ProbeConfig(n_repeats=5))
    n_boot: int = Field(default=2000, ge=100)  # for the cheap one-sample/effect-size CIs
    n_boot_mediation: int = Field(default=400, ge=50)  # model-refit bootstrap; costlier per draw
    # 200, not 20: the permutation p-value floors at 1/(n+1), so 20 permutations report a real
    # dominance of the null as p = 0.048 - indistinguishable in print from a marginal pass. At
    # 200 the same outcome quotes as p < 0.005, and the null costs ~2 min on a 2.6k-handoff cell.
    n_shuffle: int = Field(default=200, ge=0)  # within-condition perms for the RD-15 null; 0 = off
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["holm", "bh"] = "holm"
    # The overlap-restricted length control (PREREGISTRATION section 5). Three bins with a floor of
    # two episodes per condition per bin: at the E3 cell's six episodes per condition, finer strata
    # keep nothing and a floor of one lets a single episode carry a stratum.
    length_bins: int = Field(default=3, ge=1)
    length_min_per_cell: int = Field(default=2, ge=1)


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
    mean_control_cpvi: float  # CPVI against random labels (DSE-043); expected <= 0
    selectivity: float  # mean_cpvi - mean_control_cpvi
    mean_cpvi_sd: float  # mean across-repeat SD of the per-handoff scores (DSE-044)


class Contrast(BaseModel):
    """A Ck-vs-C0 contrast: effect sizes with CIs, plus the mixed-model coefficient and its p."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    cliffs_delta: float  # on episode success vs C0
    delta_ci: tuple[float, float]
    # Efficiency endpoint (P1-11): Cliff's delta on steps-to-goal vs C0 (positive = Ck slower).
    # Failures sit at steps == budget, so the rank-based delta handles the censoring mass.
    steps_delta: float
    steps_delta_ci: tuple[float, float]
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
    indirect_n_draws: int  # retained (non-degenerate) bootstrap draws behind the CI (P2-6)
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
    path_b_length_controlled: float  # path b with episode-mean message token length added (DSE-044)
    mediations: list[EpisodeMediation]
    mediation_converged: bool  # path-a and full-outcome episode models both converged
    diagnostic_cpvi_coef: float  # within-episode: per-handoff CPVI -> progress
    diagnostic_attenuation: float  # mean shrink of handoff condition coefs when CPVI enters
    mediation_note: str


class LengthMatchedContrast(BaseModel):
    """Ck-vs-C0 on success and CPVI, restricted to overlapping message-length strata (DSE-044).

    The companion to ``partial_spearman_length`` and ``path_b_length_controlled``: those adjust for
    length inside a model, this one refuses to extrapolate outside the region where both conditions
    supply episodes. PREREGISTRATION section 5 pre-registers both, and both are reported.

    Both fields share one stratification - the bins depend only on length and condition, not on
    which outcome is being differenced - so ``success.n_kept`` and ``cpvi.n_kept`` always agree.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    success: OverlapRestrictedContrast
    cpvi: OverlapRestrictedContrast


class ShuffledMessageAudit(BaseModel):
    """Within-condition message-permutation test for CPVI (RD-15).

    Permuting messages within condition decouples each message from its handoff. The criterion is a
    permutation test - the real pooled mean CPVI must exceed *every* permutation's, p = (1 + #{null
    >= real}) / (n_perm + 1). The null is **not** expected to reach zero: permutation preserves the
    condition-level signatures message style carries, and per-handoff progress base rates differ by
    condition, so condition identity alone predicts progress. The null's height is CPVI's *identity*
    component; the real-minus-null excess is the *per-handoff message content* (PREREGISTRATION §8).
    """

    model_config = ConfigDict(extra="forbid")

    n_perm: int
    mean_cpvi: float  # the real pooled per-handoff mean CPVI
    null_mean_cpvi: float  # mean over permutations of the mean CPVI (a structural floor, not 0)
    null_std_cpvi: float  # spread of the permutation null
    null_max_cpvi: float  # the criterion the real score must beat
    p_value: float  # (1 + #{null >= real}) / (n_perm + 1)


class SignalDecomposition(BaseModel):
    """Which failure a condition is having: the sender not encoding, or the receiver not acting.

    Eccles et al. (2019) separate positive *signalling* from positive *listening*; a single
    CPVI-outcome correlation conflates them. Splitting a condition's handoffs at its own median
    CPVI and crossing that with realised progress separates them on data already produced (DSE-046,
    the SocialJax replacement).

    ``absent_signal_rate`` and ``unused_signal_rate`` are rates over *all* of the condition's
    handoffs, so they sum to its no-progress rate - an additive decomposition rather than two
    conditionals, one of which would be one minus the other. The 2x2 counts are carried so any
    other conditional can be recovered without re-running the analysis.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    n_handoffs: int
    # The split threshold, within condition: low = cpvi <= median. Ties go low, which keeps the
    # rule deterministic; with heavy ties the two cells are unequal, which the counts expose.
    median_cpvi: float
    low_cpvi_no_progress: int  # absent signal: nothing was encoded, and the step went nowhere
    low_cpvi_progress: int  # progress without information - the receiver did not need the message
    high_cpvi_no_progress: int  # unused signal: the information was there and was not acted on
    high_cpvi_progress: int  # the intended cell
    absent_signal_rate: float
    absent_signal_ci: tuple[float, float]
    unused_signal_rate: float
    unused_signal_ci: tuple[float, float]


class RQ1Result(BaseModel):
    """The full RQ1 analysis, ready to persist and to drive the figures/table."""

    model_config = ConfigDict(extra="forbid")

    dataset_hash: str
    n_handoffs: int
    provenance: AnalysisProvenance  # encoder + probe + code identity (P1-8)
    conditions: list[ConditionSummary]
    contrasts: list[Contrast]
    mixed_model: MixedModelSummary
    control_mean_cpvi: float  # pooled control-task CPVI (DSE-043); the capacity rule reads this
    selectivity: float  # pooled mean CPVI - control CPVI
    partial_spearman_length: float  # CPVI vs progress, message token length partialled out
    length_matched: list[LengthMatchedContrast]  # the overlap-restricted control (DSE-044)
    signal_decomposition: list[SignalDecomposition]  # DSE-046 secondary analysis
    action_agreement: list[ActionAgreement]  # per-condition receiver-competence diagnostic
    directive_compliance: list[DirectiveCompliance]  # instruction quality vs obedience split
    seed_sensitivity: SeedSensitivity
    shuffled_message_audit: ShuffledMessageAudit | None = None  # RD-15 manipulation check
    figures: dict[str, str] = Field(default_factory=dict)


def rq1_sweep(
    model: ModelConfig,
    *,
    seeds: list[int],
    serialisations: list[Serialisation] | None = None,
    difficulties: list[Difficulty] | None = None,
    conditions: list[Condition] | None = None,
    max_steps: dict[Difficulty, int] | int | None = None,
) -> SweepConfig:
    """The RQ1 factorial: all of C0-C4 by default, crossed with serialisation/difficulty/seed.

    ``max_steps`` defaults to the certified per-difficulty feasibility budgets (P1-4); pass an int
    to broadcast one budget or a dict to override per difficulty.
    """
    return SweepConfig(
        conditions=conditions or CONDITION_ORDER,
        serialisations=serialisations or ["numeric"],
        difficulties=difficulties or ["hard"],
        seeds=seeds,
        model=model,
        max_steps=dict(STEP_BUDGETS) if max_steps is None else max_steps,
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
    *,
    control_scores: FloatArray,
    cpvi_sd: FloatArray,
) -> ConditionSummary:
    mask = np.array([r.condition == cond for r in records])
    ep = ep_frame[ep_frame["condition"] == cond]
    succ = ep["success"].to_numpy(dtype=np.float64)
    cpvi_c = cpvi_scores[mask]
    # Episode-cluster interval for the per-handoff scores: handoffs within an episode are not
    # independent, and the iid handoff resample read ~half the honest width on E3-local. Episode-
    # level quantities (success) keep the plain episode bootstrap.
    groups_c = np.unique(np.array([r.episode_id for r in records])[mask], return_inverse=True)[1]
    return ConditionSummary(
        condition=cond,
        n_episodes=len(ep),
        n_handoffs=int(mask.sum()),
        success_rate=float(succ.mean()),
        success_ci=bootstrap_ci(succ, n_boot=cfg.n_boot, alpha=cfg.alpha),
        mean_steps=float(ep["steps"].mean()),
        mean_collisions=float(ep["collisions"].mean()),
        mean_cpvi=float(cpvi_c.mean()),
        cpvi_ci=cluster_bootstrap_ci(
            cpvi_c, groups_c.astype(int), n_boot=cfg.n_boot, alpha=cfg.alpha
        ),
        mean_pvi=float(pvi_scores[mask].mean()),
        pvi_cpvi_gap=float(pvi_scores[mask].mean() - cpvi_c.mean()),
        mean_control_cpvi=float(control_scores[mask].mean()),
        selectivity=float(cpvi_c.mean() - control_scores[mask].mean()),
        mean_cpvi_sd=float(cpvi_sd[mask].mean()),
    )


class ActionAgreement(BaseModel):
    """Per-condition agreement between B's macro action and the certified scripted policy.

    The same instrument as ``pilot.g3_correctness`` - agreement with
    ``sim.feasibility.oracle_action`` against a **within-episode permutation null on B's own
    actions** - run per condition rather than pooled. Shuffling inside an episode preserves
    that episode's action habits exactly and destroys
    only their link to the pose, so a receiver that never reads the pose scores its own null by
    construction whatever its habits are.

    Deliberately a separate function from the gate. ``g3_correctness`` produced a verdict of record
    on 2026-08-29; re-shaping it to take a condition subset would edit an instrument after it had
    been read. This one is diagnostic and lives with the analysis, where a pooled number cannot say
    *which* condition the receiver was blind in - and on the E3 re-gate the pooled number and the
    per-condition numbers disagree, which is the whole mechanism finding.

    ``rotation_direction_agreement`` is the tie-free supporting read. Exact-action agreement is
    harsh near 90 degrees, where both rotation directions are near ties; restricting to handoffs
    where B rotated and the oracle wanted a rotation asks only "which way", and 0.5 is a coin
    flip. ``rotation_flip_rate`` is the oscillation measure: the share of consecutive rotation
    pairs within an episode that reverse direction, averaged over episodes.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    n_handoffs: int
    agreement: float
    null_mean: float
    null_p95: float  # the one-sided 5% level, in agreement units
    p_value: float
    n_perm: int
    rotation_direction_agreement: float  # NaN when B never rotated where the oracle wanted one
    n_rotations: int
    rotation_flip_rate: float  # NaN when no episode holds two rotations to compare


def action_agreement(
    records: list[HandoffRecord], *, n_perm: int = 2000, seed: int = 0
) -> list[ActionAgreement]:
    """Per-condition oracle agreement, its within-episode null, and the oscillation measures.

    Public for the same reason ``signal_decomposition`` is: it reads only the recorded pose and
    action, so a frozen result can be re-derived without re-fitting a probe.
    """
    conditions = np.array([r.condition for r in records])
    episodes = np.array([r.episode_id for r in records])
    oracle = np.array([str(oracle_action(float(r.pre_state["angle"]))) for r in records])
    taken = np.array([str(r.action["action"]) for r in records])
    steps = np.array([r.step for r in records], dtype=int)
    out: list[ActionAgreement] = []
    for cond in [c for c in CONDITION_ORDER if (conditions == c).any()]:
        # Seeded per condition, not once for the loop: a shared stream makes each condition's null
        # depend on which *other* conditions happen to be in the dataset, so the same arm scored on
        # a four-condition grid and on a two-condition grid would report different p-values. The key
        # is the condition's fixed position in CONDITION_ORDER, so it is grid-independent.
        rng = np.random.default_rng([seed, CONDITION_ORDER.index(cond)])
        m = conditions == cond
        o, a, ep, st = oracle[m], taken[m], episodes[m], steps[m]
        real = float(np.mean(o == a))
        blocks = [np.flatnonzero(ep == e) for e in np.unique(ep)]
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm = a.copy()
            for b in blocks:
                perm[b] = rng.permutation(a[b])
            null[i] = float(np.mean(o == perm))
        rot = np.isin(a, _ROTATIONS) & np.isin(o, _ROTATIONS)
        out.append(
            ActionAgreement(
                condition=cond,
                n_handoffs=int(m.sum()),
                agreement=real,
                null_mean=float(null.mean()),
                null_p95=float(np.quantile(null, 0.95)),
                p_value=(1 + int(np.sum(null >= real))) / (n_perm + 1),
                n_perm=n_perm,
                rotation_direction_agreement=(
                    float(np.mean(o[rot] == a[rot])) if rot.any() else float("nan")
                ),
                n_rotations=int(rot.sum()),
                rotation_flip_rate=_flip_rate(a, ep, st),
            )
        )
    return out


def _flip_rate(taken: NDArray[Any], episodes: NDArray[Any], steps: IntArray) -> float:
    """Mean over episodes of the share of consecutive rotations that reverse direction.

    Episodes with fewer than two rotations contribute nothing - there is no pair to compare - and
    the result is NaN when no episode qualifies, rather than a 0.0 that would read as "never
    oscillates" on a condition that barely rotates at all.
    """
    rates: list[float] = []
    for e in np.unique(episodes):
        m = episodes == e
        seq = taken[m][np.argsort(steps[m])]
        seq = seq[np.isin(seq, _ROTATIONS)]
        if len(seq) >= 2:
            rates.append(float(np.mean(seq[:-1] != seq[1:])))
    return float(np.mean(rates)) if rates else float("nan")


class DirectiveCompliance(BaseModel):
    """Where B's rotation errors come from: a wrong instruction, or a wrong reading of it.

    ``action_agreement`` says *whether* B's actions track the pose. This says *why not*, by
    splitting the same rotation handoffs into two independent links:

    - ``directive_agreement`` - does A's stated turn direction match ``oracle_action``? 0.5 is a
      sender that writes a fluent instruction carrying no information about which way to turn.
    - ``obedience`` - does B do what it was told? 1.0 is a receiver that contributes nothing of its
      own and, crucially, loses nothing either.

    When obedience is high, ``receiver_agreement`` is pinned to ``directive_agreement`` and the
    bottleneck is the *sender*, not the receiver - which is the opposite of how a state-blindness
    result reads without this split. Scored only where A named a direction, the oracle wanted a
    rotation, and B rotated; ``n`` is that subset and ``coverage`` its share of the condition.

    Direction words are read from the DELIVERED message, because that is what B saw: a channel that
    severs the directive must show up here as coverage collapsing towards zero. ``ROT+`` is
    counterclockwise (``apply_macro_action`` adds angular velocity), which is what pins the mapping.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    n: int
    coverage: float  # share of the condition's handoffs carrying a direction word
    directive_agreement: float  # A's instruction vs the oracle; 0.5 = a coin flip
    obedience: float  # B's action vs A's instruction
    receiver_agreement: float  # B's action vs the oracle, on this same subset


# Direction words as they appear in the messages, longest alternative first so that
# "counterclockwise" is never matched as "clockwise" with the negation stripped off.
_DIRECTIVE_RE = re.compile(r"counter-?clockwise|clockwise|ROT\+|ROT-", re.IGNORECASE)


def _directive(message: str | None) -> str | None:
    """The LAST direction A names, as a macro action. Last, not first: the messages state the
    problem before the instruction ("...unless it is rotated. Rotate the load counterclockwise").
    """
    hits = _DIRECTIVE_RE.findall(message or "")
    if not hits:
        return None
    word = hits[-1].lower()
    return "ROT+" if word.startswith("counter") or word == "rot+" else "ROT-"


def _rate(flags: list[bool]) -> float:
    """NaN, never 0.0, on an empty subset: a condition whose channel severs every directive has
    nothing to score, and 0.0 would read as "the sender is always wrong" rather than "never spoke".
    """
    return float(np.mean(flags)) if flags else float("nan")


def directive_compliance(records: list[HandoffRecord]) -> list[DirectiveCompliance]:
    """Per-condition split of rotation agreement into instruction quality and obedience.

    Public for the same reason ``action_agreement`` is: it reads only the recorded message, pose and
    action, so a frozen result can be re-derived without re-fitting a probe.
    """
    out: list[DirectiveCompliance] = []
    for cond in [c for c in CONDITION_ORDER if any(r.condition == c for r in records)]:
        rows = [r for r in records if r.condition == cond]
        triples = [
            (d, oracle_action(float(r.pre_state["angle"])), str(r.action["action"]))
            for r in rows
            for d in [_directive(r.message_delivered)]
            if d is not None
        ]
        scored = [t for t in triples if t[1] in _ROTATIONS and t[2] in _ROTATIONS]
        out.append(
            DirectiveCompliance(
                condition=cond,
                n=len(scored),
                coverage=len(triples) / len(rows) if rows else float("nan"),
                directive_agreement=_rate([d == o for d, o, _ in scored]),
                obedience=_rate([a == d for d, _, a in scored]),
                receiver_agreement=_rate([a == o for _, o, a in scored]),
            )
        )
    return out


def signal_decomposition(
    records: list[HandoffRecord],
    cpvi_scores: FloatArray,
    y: IntArray,
    cfg: RQ1Config | None = None,
) -> list[SignalDecomposition]:
    """The per-condition 2x2 with episode-cluster intervals on the two named rates (DSE-046).

    The median is taken **within** condition (the pre-registered rule): a pooled split would put
    the low-CPVI cell wherever the condition effect already put it, so the decomposition would
    restate the gradient instead of decomposing it. It is computed once on the observed sample and
    held fixed inside the bootstrap - the interval covers the rates given that split rule, not the
    extra variation of re-picking the threshold on every draw.

    Public because ``scores.parquet`` persists per-handoff CPVI: the decomposition can be recomputed
    from a frozen result without re-fitting probes, which would move the scores under it.
    """
    cfg = cfg or RQ1Config()
    conditions = np.array([r.condition for r in records])
    episodes = np.array([r.episode_id for r in records])
    out: list[SignalDecomposition] = []
    for cond in [c for c in CONDITION_ORDER if (conditions == c).any()]:
        mask = conditions == cond
        cpvi_c = cpvi_scores[mask]
        fail = y[mask] == 0
        median = float(np.median(cpvi_c))
        low = cpvi_c <= median
        absent = (low & fail).astype(np.float64)
        unused = (~low & fail).astype(np.float64)
        groups_c = np.unique(episodes[mask], return_inverse=True)[1].astype(int)
        out.append(
            SignalDecomposition(
                condition=cond,
                n_handoffs=int(mask.sum()),
                median_cpvi=median,
                low_cpvi_no_progress=int((low & fail).sum()),
                low_cpvi_progress=int((low & ~fail).sum()),
                high_cpvi_no_progress=int((~low & fail).sum()),
                high_cpvi_progress=int((~low & ~fail).sum()),
                absent_signal_rate=float(absent.mean()),
                absent_signal_ci=cluster_bootstrap_ci(
                    absent, groups_c, n_boot=cfg.n_boot, alpha=cfg.alpha
                ),
                unused_signal_rate=float(unused.mean()),
                unused_signal_ci=cluster_bootstrap_ci(
                    unused, groups_c, n_boot=cfg.n_boot, alpha=cfg.alpha
                ),
            )
        )
    return out


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
    """One row per episode: condition, seed, terminal success, mean CPVI, and mean message length.

    ``msg_len`` is the DSE-044 length covariate: whitespace tokens of the *delivered* message - the
    same unit the C1 cap operates in, so it measures exactly what the channel manipulates.
    """
    rows: dict[str, dict[str, Any]] = {}
    for r, s in zip(records, cpvi_scores, strict=True):
        row = rows.setdefault(
            r.episode_id,
            {"condition": r.condition, "seed": r.seed, "success": False, "_cpvi": [], "_len": []},
        )
        row["success"] = row["success"] or bool(r.y_terminal_success)
        row["_cpvi"].append(float(s))
        row["_len"].append(float(len(r.message_delivered.split())))
    return pd.DataFrame(
        [
            {
                "condition": v["condition"],
                "seed": v["seed"],
                "success": 1 if v["success"] else 0,
                "cpvi": float(np.mean(v["_cpvi"])),
                "msg_len": float(np.mean(v["_len"])),
            }
            for v in rows.values()
        ]
    )


def _bootstrap_indirect(
    ep_df: pd.DataFrame, targets: Sequence[str], cfg: RQ1Config
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Percentile bootstrap CI of the indirect effect a*b per condition, resampling episodes.

    Each draw refits path a (cpvi~condition) and path b (success~condition+cpvi) on the resampled
    episodes; degenerate draws - a dropped condition (which would shift the C0 reference) or a
    single-class outcome (unfittable) - are skipped. The retained-draw count is returned alongside
    the CIs so an interval built from few surviving draws is visibly flagged (P2-6). ponytail:
    plain episode resample, n_boot_mediation refits; cluster-resample seeds only if seed
    clustering dominates the variance.
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
    cis: dict[str, tuple[float, float]] = {}
    counts: dict[str, int] = {}
    for c in targets:
        vals = np.array(draws[c], dtype=np.float64)
        counts[c] = len(vals)
        if len(vals) < 2:
            cis[c] = (float("nan"), float("nan"))
        else:
            lo, hi = np.quantile(vals, [cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0])
            cis[c] = (float(lo), float(hi))
    return cis, counts


def _episode_mediation(
    ep_df: pd.DataFrame, present: list[Condition], cfg: RQ1Config
) -> tuple[list[EpisodeMediation], float, float, bool]:
    """Episode-level Baron-Kenny: paths a, b, c, c' and the bootstrapped indirect effect per Ck.

    Also refits path b with episode-mean message length as a covariate (DSE-044): C1 shortens
    messages by construction, so the uncontrolled path b cannot distinguish "CPVI carries the
    channel effect" from "message length does". Both are reported.
    """
    targets = [c for c in present if c != "C0"]
    if ep_df["success"].nunique() < 2:  # single-class outcome: mediation unmeasurable (cf. G2)
        nan = float("nan")
        meds = [
            EpisodeMediation(
                condition=c,
                path_a=nan,
                indirect=nan,
                indirect_ci=(nan, nan),
                indirect_n_draws=0,
                direct=nan,
                total=nan,
                prop_mediated=nan,
            )
            for c in targets
        ]
        return meds, nan, nan, False
    a_model, a_conv = _fit_mixed(ep_df, "cpvi ~ C(condition)")
    y_full, yf_conv = _fit_mixed(ep_df, "success ~ C(condition) + cpvi")
    y_red, _ = _fit_mixed(ep_df, "success ~ C(condition)")
    a = _condition_terms(a_model.params)
    cprime = _condition_terms(y_full.params)
    total = _condition_terms(y_red.params)
    b = float(y_full.params["cpvi"])
    # Length has no variance to control for when every episode's mean message length is identical;
    # the covariate model is then undefined rather than failed, so report it as such.
    if ep_df["msg_len"].nunique() < 2:
        b_len = float("nan")
    else:
        y_len, _ = _fit_mixed(ep_df, "success ~ C(condition) + cpvi + msg_len")
        b_len = float(y_len.params["cpvi"])
    ci, n_draws = _bootstrap_indirect(ep_df, targets, cfg)
    meds = [
        EpisodeMediation(
            condition=c,
            path_a=a.get(c, 0.0),
            indirect=a.get(c, 0.0) * b,
            indirect_ci=ci[c],
            indirect_n_draws=n_draws[c],
            direct=cprime.get(c, 0.0),
            total=total.get(c, 0.0),
            prop_mediated=(a.get(c, 0.0) * b / total[c]) if total.get(c, 0.0) != 0.0 else 0.0,
        )
        for c in targets
    ]
    return meds, b, b_len, bool(a_conv and yf_conv)


def _length_matched(
    ep_df: pd.DataFrame, present: list[Condition], cfg: RQ1Config
) -> list[LengthMatchedContrast]:
    """Ck-vs-C0 differences taken only inside message-length strata holding both conditions.

    C1 caps message length, so length is confounded with condition by construction and the raw
    contrast cannot separate the channel from the length it imposes. Restricting to the overlap is
    a sensitivity analysis rather than a length-free estimate: where the distributions barely meet,
    the result says so (``interpretable=False``) instead of returning a confident number computed
    from two or three episodes.
    """
    out: list[LengthMatchedContrast] = []
    for c in present:
        if c == "C0":
            continue
        sub = ep_df[ep_df["condition"].isin(["C0", c])]
        treated = (sub["condition"] == c).to_numpy(dtype=bool)
        length = sub["msg_len"].to_numpy(dtype=np.float64)
        # Two calls re-bin identically (the strata depend only on length and condition); at episode
        # N this costs nothing and keeps the estimator a plain two-array function.
        out.append(
            LengthMatchedContrast(
                condition=c,
                success=overlap_restricted_contrast(
                    sub["success"].to_numpy(dtype=np.float64),
                    length,
                    treated,
                    n_bins=cfg.length_bins,
                    min_per_cell=cfg.length_min_per_cell,
                ),
                cpvi=overlap_restricted_contrast(
                    sub["cpvi"].to_numpy(dtype=np.float64),
                    length,
                    treated,
                    n_bins=cfg.length_bins,
                    min_per_cell=cfg.length_min_per_cell,
                ),
            )
        )
    return out


def _mixed_model(
    handoff_df: pd.DataFrame, ep_df: pd.DataFrame, present: list[Condition], cfg: RQ1Config
) -> tuple[MixedModelSummary, dict[str, tuple[float, float]]]:
    """Assemble the H1 handoff model and the H2 episode mediation into the persisted summary.

    Both models regress on ``C(condition)``, so on a single-condition grid the design matrix is
    rank-deficient by construction: there is no contrast to estimate. statsmodels does not refuse -
    it emits ConvergenceWarnings and returns coefficients from a boundary fit, which land in the
    artefact looking like estimates. Refuse here instead, and say why (DSE-067).
    """
    if len(present) < 2:
        note = (
            f"H1/H2 not fitted: only condition {present[0]} is present, so C(condition) is "
            "rank-deficient and there is no contrast to estimate. This is a capability grid, "
            "not a gradient; the mixed model requires at least two conditions."
        )
        return (
            MixedModelSummary(
                formula="not fitted (single-condition grid)",
                coef_no_mediator={},
                converged=False,
                mediation_outcome="episode_success",
                path_b=float("nan"),
                path_b_length_controlled=float("nan"),
                mediations=[],
                mediation_converged=False,
                diagnostic_cpvi_coef=float("nan"),
                diagnostic_attenuation=float("nan"),
                mediation_note=note,
            ),
            {},
        )
    coef_no, cpvi_coef, attenuation, converged, coef_p = _handoff_model(handoff_df)
    mediations, path_b, path_b_len, med_conv = _episode_mediation(ep_df, present, cfg)
    finite_indirect = [m.indirect for m in mediations if np.isfinite(m.indirect)]
    mean_indirect = float(np.mean(finite_indirect)) if finite_indirect else float("nan")
    summary = MixedModelSummary(
        formula="H1: y ~ C(condition) [groups=seed, vc=episode]; "
        "H2: success ~ C(condition) + cpvi_epmean [groups=seed]",
        coef_no_mediator=coef_no,
        converged=converged,
        mediation_outcome="episode_success",
        path_b=path_b,
        path_b_length_controlled=path_b_len,
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


def _shuffle_audit(
    e_s: FloatArray,
    e_m: FloatArray,
    y: IntArray,
    groups: IntArray,
    records: list[HandoffRecord],
    cpvi_scores: FloatArray,
    cfg: RQ1Config,
) -> ShuffledMessageAudit | None:
    """The RD-15 within-condition permutation null (skipped when ``cfg.n_shuffle == 0``)."""
    if cfg.n_shuffle == 0:
        return None
    conditions = np.array([r.condition for r in records])
    null = shuffled_message_cpvi(
        e_s,
        e_m,
        y,
        groups,
        conditions,
        cfg.probe,
        rng=np.random.default_rng(0),
        n_perm=cfg.n_shuffle,
    )
    real = float(np.mean(cpvi_scores))
    return ShuffledMessageAudit(
        n_perm=cfg.n_shuffle,
        mean_cpvi=real,
        null_mean_cpvi=float(np.mean(null)),
        null_std_cpvi=float(np.std(null)),
        null_max_cpvi=float(np.max(null)),
        p_value=float((1 + int(np.sum(null >= real))) / (cfg.n_shuffle + 1)),
    )


def analyse_rq1(
    records: list[HandoffRecord],
    featuriser: Featuriser,
    *,
    dataset_hash: str,
    cfg: RQ1Config | None = None,
) -> tuple[RQ1Result, pd.DataFrame]:
    """Score CPVI/PVI, summarise per condition, fit the mixed model + mediation, build contrasts.

    Returns the result plus the per-handoff score frame (episode_id, step, condition, seed, cpvi,
    pvi), row-aligned to ``records`` - persisted by ``write_rq1`` (P1-17): the methodology promises
    the per-handoff CPVI *distribution*, RQ2 consumes exactly these scores, and re-computing them
    re-fits probes (probe-seed noise makes the recomputation non-identical).
    """
    cfg = cfg or RQ1Config()
    if not records:
        raise ConfigError("analyse_rq1 called with no records")
    y = _require_progress_labels(records)
    if len(np.unique(y)) < 2:
        raise ConfigError("RQ1 needs both progress classes to fit the model and estimate CPVI")

    e_s, e_m = featuriser.featurise(records)
    groups = _groups(records)
    cpvi_scores, cpvi_sd = cpvi_with_sd(e_s, e_m, y, groups, cfg.probe)
    pvi_scores = pvi(e_m, y, groups, cfg.probe)
    # Same features, same splitter, same probe family, random labels: whatever CPVI survives is
    # manufactured by the probe rather than carried by the message (DSE-043, PREREGISTRATION §5).
    control_scores = control_task_cpvi(e_s, e_m, y, groups, cfg.probe)
    msg_tokens = np.array([len(r.message_delivered.split()) for r in records], dtype=np.float64)
    scores = pd.DataFrame(
        {
            "episode_id": [r.episode_id for r in records],
            "step": [r.step for r in records],
            "condition": [r.condition for r in records],
            "seed": [r.seed for r in records],
            "cpvi": cpvi_scores,
            "cpvi_sd": cpvi_sd,
            "pvi": pvi_scores,
            "msg_tokens": msg_tokens,
        }
    )
    ep_frame = _episode_frame(records)

    present = [c for c in CONDITION_ORDER if any(r.condition == c for r in records)]
    summaries = [
        _condition_summary(
            c,
            records,
            cpvi_scores,
            pvi_scores,
            ep_frame,
            cfg,
            control_scores=control_scores,
            cpvi_sd=cpvi_sd,
        )
        for c in present
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
    #
    # On a single-condition grid `hardest is C0` and that gap is C0 minus itself: exactly 0.0 for
    # every seed, an all-zero report that reads as a perfectly seed-stable gradient when the truth
    # is that no gradient exists to be stable (DSE-067; it produced exactly that on the two v9
    # capability arms). Fall back to the per-seed success rate, labelled as such, with the binomial
    # dispersion index carrying the inference - the only across-seed question a one-condition grid
    # can actually answer.
    seeds_metric: dict[int, float] = {}
    if len(present) < 2:
        counts: dict[int, tuple[int, int]] = {}
        for s in sorted(ep_frame["seed"].unique()):
            succ = ep_frame[ep_frame["seed"] == s]["success"]
            seeds_metric[int(s)] = float(succ.mean())
            counts[int(s)] = (int(succ.sum()), len(succ))
        seed_sens = seed_sensitivity(
            seeds_metric,
            metric="success_rate",
            counts=counts,
            reason=(
                f"only condition {present[0]} is present, so the C0-minus-hardest gap is a "
                "self-subtraction and identically zero; reporting the per-seed success rate and "
                "its binomial dispersion instead"
            ),
        )
    else:
        hardest = present[-1]
        for s in sorted(ep_frame["seed"].unique()):
            sub = ep_frame[ep_frame["seed"] == s]
            c0 = sub[sub["condition"] == "C0"]["success"]
            hard = sub[sub["condition"] == hardest]["success"]
            if len(c0) and len(hard):
                seeds_metric[int(s)] = float(c0.mean() - hard.mean())
        seed_sens = seed_sensitivity(seeds_metric)
    result = RQ1Result(
        dataset_hash=dataset_hash,
        n_handoffs=len(records),
        provenance=build_provenance(featuriser.cfg, cfg.probe),
        conditions=summaries,
        contrasts=contrasts,
        mixed_model=mixed,
        control_mean_cpvi=float(control_scores.mean()),
        selectivity=float(cpvi_scores.mean() - control_scores.mean()),
        partial_spearman_length=partial_spearman(cpvi_scores, y.astype(np.float64), msg_tokens),
        length_matched=_length_matched(ep_med_df, present, cfg),
        signal_decomposition=signal_decomposition(records, cpvi_scores, y, cfg),
        action_agreement=action_agreement(records),
        directive_compliance=directive_compliance(records),
        seed_sensitivity=seed_sens,
        shuffled_message_audit=_shuffle_audit(e_s, e_m, y, groups, records, cpvi_scores, cfg),
    )
    return result, scores


def _contrasts(
    present: list[Condition],
    ep_frame: pd.DataFrame,
    coef_p: dict[str, tuple[float, float]],
    cfg: RQ1Config,
) -> list[Contrast]:
    """Ck-vs-C0 effect sizes (Cliff's delta on success and on steps) with corrected p-values."""
    if "C0" not in present:
        raise ConfigError("RQ1 contrasts need C0 as the reference condition")
    c0 = ep_frame[ep_frame["condition"] == "C0"]["success"].to_numpy(dtype=np.float64)
    c0_steps = ep_frame[ep_frame["condition"] == "C0"]["steps"].to_numpy(dtype=np.float64)
    targets = [c for c in present if c != "C0"]
    raw_p = np.array([coef_p[c][1] for c in targets], dtype=np.float64)
    corrected = correct_pvalues(raw_p, method=cfg.correction)
    out: list[Contrast] = []
    for c, p_corr in zip(targets, corrected, strict=True):
        ck = ep_frame[ep_frame["condition"] == c]["success"].to_numpy(dtype=np.float64)
        ck_steps = ep_frame[ep_frame["condition"] == c]["steps"].to_numpy(dtype=np.float64)
        out.append(
            Contrast(
                condition=c,
                cliffs_delta=cliffs_delta(ck, c0),  # negative = Ck worse than C0 (degradation)
                delta_ci=_delta_ci(ck, c0, cfg),
                # Efficiency (P1-11): positive = Ck takes more steps; failures sit at the budget,
                # which the rank-based delta treats as the censored mass (ANALYSIS_PROTOCOL).
                steps_delta=cliffs_delta(ck_steps, c0_steps),
                steps_delta_ci=_delta_ci(ck_steps, c0_steps, cfg),
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
    client_b: LLMClient | None = None,
    root: Path | str,
    cfg: RQ1Config | None = None,
) -> tuple[RQ1Result, pd.DataFrame]:
    """Run the RQ1 grid and analyse it end to end (full-scale run gated on DSE-005 compute).

    ``client_b`` serves agent B (DSE-049); omitted means self-play, the primary cell.
    """
    run_grid(sweep, client, client_b, root=root)
    d_hash = dataset_hash_for(sweep)
    return analyse_rq1(load_records(d_hash, root=root), featuriser, dataset_hash=d_hash, cfg=cfg)


def write_rq1(result: RQ1Result, dir: Path | str, *, scores: pd.DataFrame) -> Path:
    """Persist the analysis JSON, per-handoff scores (Parquet), the results table, and figures."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "rq1.json").write_text(result.model_dump_json(indent=2))
    scores.to_parquet(dir / "scores.parquet", index=False)  # the P1-17 join-key artefact
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
        title=f"RQ1: CPVI vs condition (selectivity {result.selectivity:+.3f} bits)",
        path=dir / "cpvi_vs_condition.png",
    )
    pd.DataFrame([a.model_dump() for a in result.action_agreement]).to_csv(
        dir / "action_agreement.csv", index=False
    )
    pd.DataFrame([d.model_dump() for d in result.directive_compliance]).to_csv(
        dir / "directive_compliance.csv", index=False
    )
    dec = result.signal_decomposition
    dec_table = pd.DataFrame([d.model_dump() for d in dec])
    dec_table.to_csv(dir / "signal_decomposition.csv", index=False)
    dec_out = series_plot(
        [d.condition for d in dec],
        {
            "absent signal (low CPVI, no progress)": (
                [d.absent_signal_rate for d in dec],
                [d.absent_signal_ci for d in dec],
            ),
            "unused signal (high CPVI, no progress)": (
                [d.unused_signal_rate for d in dec],
                [d.unused_signal_ci for d in dec],
            ),
        },
        ylabel="share of handoffs",
        title="RQ1 secondary: absent vs unused signal",
        path=dir / "signal_decomposition.png",
    )
    if out is not None and cpvi_out is not None:  # both render or neither (viz extra present)
        result.figures = {"outcome": str(out), "cpvi": str(cpvi_out)}
        if dec_out is not None:
            result.figures["signal_decomposition"] = str(dec_out)
        (dir / "rq1.json").write_text(result.model_dump_json(indent=2))  # rewrite with figure paths
    return dir
