"""RQ3a localisation, baselines and the judge-agreement audit (DSE-024).

H5 asks whether boundary CPVI localises the responsible step in *real* multi-agent traces better
than cheap surface baselines and the published Who&When attribution methods. This module is the
analysis; the loaders (DSE-041) and the replay outcome labeller (DSE-042) supply its inputs.

Four disciplines are structural here rather than conventions:

* **Scorers cannot read annotations.** Every method consumes :class:`LocalisationStep`, a six-field
  view with no ``annotations`` attribute, exactly as ``rq3a_replay.ReplayStep`` does. The
  annotations are the *evaluation target*; a scorer able to reach them would score a target against
  itself. They enter only through :func:`trace_targets`, on the evaluator's side of the boundary.
* **Both CPVI regimes are reported, neither is silently picked.** Transfer (the simulator-trained
  statistic, applied to logs) and refit (probes refit on held-out traces) answer the same question
  under different assumptions, so each carries a ``status``: ``ok``, ``unavailable`` (the input does
  not exist yet) or ``not_applicable`` (the corpus cannot support it). A table cell reads
  "unavailable - <reason>", never ``0.0`` and never a dropped row.
* **The judge is an open-weight re-implementation, and says so.** The published Who&When methods
  were run against a hosted frontier annotator. Every model call in this project is local or on the
  Myriad allocation, so the three procedures are re-implemented against the served open-weight tier
  and reported as a *replication*, with the model identity and decoding recorded. Their numbers are
  not the published numbers and must not be tabled as if they were.
* **The agreement audit is judge-versus-annotation, not a new human study.** ``kappa`` here compares
  the judge's selected agent with the corpus's existing annotation on a sampled, recorded subset. A
  genuine human double-annotation exercise (two raters, frozen rubric, adjudication) is a separate
  piece of work and is not claimed by this module.

Orientation convention, applied once: every method emits a **risk** score where *higher = more
suspect*. CPVI is information, so its risk is the negation; the transfer regime multiplies by the
calibrated orientation and refuses to run without one, because guessing that sign inverts the
result.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score

from preceptx.analysis.stats import AnalysisProvenance, bootstrap_ci, build_provenance
from preceptx.config import ConfigError
from preceptx.data.logs import LogHandoffRecord, LogTraceRecord
from preceptx.gate.statistics import GateError, load_statistic, resolve_statistic_key
from preceptx.measure.featuriser import Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, cpvi, pvi

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

Status = Literal["ok", "unavailable", "not_applicable"]

# Ranking is by descending risk with **average** ranks for ties, so a method that scores every step
# identically (an abstaining judge, a constant baseline) cannot win by input order. Recorded in the
# result and the manifest because it changes what step accuracy means.
TIE_POLICY = "descending risk, average ranks for ties; top-k inclusive at the cutoff"

# MAST publishes each trace as one unsegmented transcript (schema mapping section 4), so it has no
# observation/message split and therefore no conditioning state. CPVI is undefined there - not
# small, undefined - and the category arm below is reported separately for exactly that reason.
MAST_CPVI_REASON = (
    "MAST publishes traces as one unsegmented transcript with no observation/message split, so "
    "there is no conditioning state and CPVI is undefined (docs/rq3a_schema_mapping.md section 4)"
)

# The MAST failure-mode family for inter-agent misalignment; its modes are keyed "2.x".
MAST_MISALIGNMENT_PREFIX = "2."

_TRANSCRIPT_FIELD_CHARS = 1200  # per-field clip, so a judge prompt stays bounded on long traces


class RQ3aError(RuntimeError):
    """An RQ3a analysis input is unusable. Raised rather than defaulted."""


# --------------------------------------------------------------------------------------------
# The annotation-blind view every scorer consumes
# --------------------------------------------------------------------------------------------


class LocalisationStep(BaseModel):
    """One step, reduced to what a localisation method may see.

    ``LogHandoffRecord.annotations`` is absent rather than empty: the mistake step and agent are
    what this analysis scores methods *against*, so no scoring path may reach them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    step: int = Field(ge=0)
    agent_name: str
    observation: str
    message: str
    is_handoff: bool


def localisation_steps(
    records: Sequence[LogHandoffRecord], *, handoffs_only: bool = True
) -> list[LocalisationStep]:
    """Project log records onto the scoring view, dropping annotations and the trace outcome.

    ``handoffs_only`` keeps the inter-agent boundaries, which is where the H5 claim lives. The
    intra-agent turns stay in the loaded dataset (dropping them there would move the base rate);
    they are filtered here, at scoring time, so the choice is visible in the result.
    """
    return [
        LocalisationStep(
            trace_id=r.trace_id,
            step=r.step,
            agent_name=r.agent_name,
            observation=r.observation,
            message=r.message,
            is_handoff=r.is_handoff,
        )
        for r in records
        if r.is_handoff or not handoffs_only
    ]


def by_trace(steps: Sequence[LocalisationStep]) -> dict[str, list[LocalisationStep]]:
    """Group steps by trace, preserving corpus step order within each trace."""
    out: dict[str, list[LocalisationStep]] = {}
    for s in steps:
        out.setdefault(s.trace_id, []).append(s)
    for trace in out.values():
        trace.sort(key=lambda s: s.step)
    return out


# --------------------------------------------------------------------------------------------
# The evaluator's side: annotations, kept off the scoring path
# --------------------------------------------------------------------------------------------


class TraceTarget(BaseModel):
    """What a corpus says went wrong in one trace. Evaluation only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    decisive_step: int | None
    decisive_agent: str | None


def _as_step(value: object) -> int | None:
    """Corpus step annotations arrive as ints, numeric strings or blanks; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def trace_targets(records: Sequence[LogHandoffRecord]) -> dict[str, TraceTarget]:
    """The per-trace evaluation targets from ``mistake_step`` / ``mistake_agent``.

    Built from the records rather than passed alongside the steps so the annotations never travel
    with the object the scorers hold.
    """
    out: dict[str, TraceTarget] = {}
    for r in records:
        if r.trace_id in out:
            continue
        agent = r.annotations.get("mistake_agent")
        out[r.trace_id] = TraceTarget(
            trace_id=r.trace_id,
            decisive_step=_as_step(r.annotations.get("mistake_step")),
            decisive_agent=str(agent) if isinstance(agent, str) and agent.strip() else None,
        )
    return out


# --------------------------------------------------------------------------------------------
# The refit regime's outcome, and the census that says whether one exists
# --------------------------------------------------------------------------------------------


class OutcomeCensus(BaseModel):
    """What annotation-free outcome the corpus actually supplies for the refit regime.

    DSE-042 provides for the refit arm to fall back on the cheap trace-level outcome if
    counterfactual replay is cut. Whether that fallback is *available* is an empirical property of
    the corpus, not a design choice, so it is counted and recorded rather than assumed. ``usable``
    is false whenever the labels are absent or single-class - either way there is nothing for a
    probe to separate, and the difference between "not run yet" and "cannot be run" matters.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    traces_failed: int
    traces_succeeded: int
    traces_unlabelled: int
    steps_labelled: int
    usable: bool
    reason: str


def trace_outcome_labels(records: Sequence[LogHandoffRecord]) -> dict[tuple[str, int], bool]:
    """Broadcast each trace's annotation-free outcome onto its steps (DSE-042's cheap fallback).

    ``trace_failed is None`` contributes no label: the corpus records no harness outcome for that
    trace, and defaulting it either way would invent the class the refit arm is short of.
    """
    return {
        (r.trace_id, r.step): bool(r.trace_failed) for r in records if r.trace_failed is not None
    }


def outcome_census(records: Sequence[LogHandoffRecord], *, source: str) -> OutcomeCensus:
    """Count the refit regime's available outcome classes, and say why a refit can or cannot run."""
    per_trace = {r.trace_id: r.trace_failed for r in records}
    failed = sum(1 for v in per_trace.values() if v is True)
    succeeded = sum(1 for v in per_trace.values() if v is False)
    labelled = sum(1 for r in records if r.trace_failed is not None)
    if failed and succeeded:
        return OutcomeCensus(
            source=source,
            traces_failed=failed,
            traces_succeeded=succeeded,
            traces_unlabelled=len(per_trace) - failed - succeeded,
            steps_labelled=labelled,
            usable=True,
            reason="",
        )
    present = "failures" if failed else "non-failures" if succeeded else "no labelled traces"
    return OutcomeCensus(
        source=source,
        traces_failed=failed,
        traces_succeeded=succeeded,
        traces_unlabelled=len(per_trace) - failed - succeeded,
        steps_labelled=labelled,
        usable=False,
        reason=(
            f"the corpus supplies {present} only ({failed} failed, {succeeded} succeeded, "
            f"{len(per_trace) - failed - succeeded} unlabelled of {len(per_trace)} traces), so the "
            "trace-level fallback is single-class and only counterfactual replay (DSE-042) can "
            "define an outcome here"
        ),
    )


# --------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------


class StepScore(BaseModel):
    """One method's risk for one step. Higher = more suspect (the module-wide convention)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    step: int
    risk: float


class MethodScores(BaseModel):
    """What one method produced, including why it produced nothing.

    One type for baselines, judges and both CPVI regimes, so the results table has one shape and an
    unavailable method keeps its row instead of vanishing from the comparison.
    """

    model_config = ConfigDict(extra="forbid")

    method: str
    status: Status = "ok"
    reason: str | None = None
    scores: list[StepScore] = Field(default_factory=list)
    n_abstained: int = 0  # traces the method declined to rank (a judge failure, never a fallback)
    model_calls: int = 0


Scorer = Callable[[Sequence[LocalisationStep]], MethodScores]


def schema_validity_scores(steps: Sequence[LocalisationStep]) -> MethodScores:
    """Baseline: a step is suspect when what it emitted is empty or unbalanced.

    The cheapest deployable check a practitioner already has - if it localises as well as CPVI, the
    measurement is not earning its keep. Kept deliberately crude: emptiness plus unbalanced
    brackets/braces, which catches truncated tool calls without becoming a parser.
    """
    scores = [
        StepScore(trace_id=s.trace_id, step=s.step, risk=float(_invalid(s.message))) for s in steps
    ]
    return MethodScores(method="schema_validity", scores=scores)


def _invalid(message: str) -> bool:
    text = message.strip()
    if not text:
        return True
    return any(text.count(a) != text.count(b) for a, b in ("{}", "[]", "()"))


def cosine_scores(steps: Sequence[LocalisationStep], featuriser: Featuriser) -> MethodScores:
    """Baseline: cosine(observation, message) - a message that only restates the state is suspect.

    Probe-free, so it answers the "your localisation is an artefact of the probe" objection the same
    way ``CosineStatistic`` does in the runtime stack.
    """
    if not steps:
        return MethodScores(method="mean_cosine", status="unavailable", reason="no steps")
    e_s = featuriser.embed_texts([s.observation for s in steps])
    e_m = featuriser.embed_texts([s.message for s in steps])
    num = np.einsum("ij,ij->i", e_s, e_m)
    den = np.linalg.norm(e_s, axis=1) * np.linalg.norm(e_m, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return MethodScores(
        method="mean_cosine",
        scores=[
            StepScore(trace_id=s.trace_id, step=s.step, risk=float(c))
            for s, c in zip(steps, cos, strict=True)
        ],
    )


# --------------------------------------------------------------------------------------------
# The published Who&When procedures, re-implemented against an open-weight judge
# --------------------------------------------------------------------------------------------


class JudgeBackend(ABC):
    """A judge that answers one attribution question at a time.

    Three narrow methods rather than one free-text call: the *procedure* (all-at-once, binary
    search, step-by-step) lives here in the module, the backend only answers, and no answer parsing
    leaks into the analysis. ``None`` is a first-class return - the judge failed or refused - and is
    recorded as an abstention. A backend must never fall back to the annotation.
    """

    model_name: str
    model_revision: str
    decoding: str

    @abstractmethod
    def select_step(self, transcript: str, n_steps: int) -> int | None:
        """All-at-once: index (0-based, into the trace's steps) of the decisive step."""

    @abstractmethod
    def contains_error(self, transcript: str) -> bool | None:
        """Binary search: does this contiguous segment contain the decisive step?"""

    @abstractmethod
    def is_error(self, transcript: str, step_text: str) -> bool | None:
        """Step-by-step: is this one step, in its preceding context, the decisive step?"""


class JudgeIdentity(BaseModel):
    """What was actually asked, recorded so the replication is auditable."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_revision: str
    decoding: str
    replication_of: str = "Who&When all-at-once / binary-search / step-by-step"
    published_result: str = (
        "the published baselines were run against a hosted frontier annotator; these are an "
        "open-weight re-implementation and are not comparable to the published figures as if the "
        "same annotator had been used"
    )


def _clip(text: str) -> str:
    return text[:_TRANSCRIPT_FIELD_CHARS]


def _transcript(steps: Sequence[LocalisationStep], *, offset: int = 0) -> str:
    """Render steps for a judge prompt: numbered, agent-attributed, per-field clipped."""
    return "\n\n".join(
        f"[step {offset + i}] agent={s.agent_name}\n"
        f"context: {_clip(s.observation)}\n"
        f"emitted: {_clip(s.message)}"
        for i, s in enumerate(steps)
    )


def _pointed(trace: Sequence[LocalisationStep], selected: int | None) -> list[StepScore]:
    """One-hot risk over a trace; an abstention is all-zero, which cannot rank first on ties."""
    return [
        StepScore(trace_id=s.trace_id, step=s.step, risk=float(selected == i))
        for i, s in enumerate(trace)
    ]


def judge_all_at_once(steps: Sequence[LocalisationStep], backend: JudgeBackend) -> MethodScores:
    """One call per trace: show the whole transcript, ask for the decisive step."""
    out: list[StepScore] = []
    abstained = calls = 0
    for trace in by_trace(steps).values():
        pick = backend.select_step(_transcript(trace), len(trace))
        calls += 1
        if pick is None or not 0 <= pick < len(trace):
            abstained += 1
            pick = None
        out.extend(_pointed(trace, pick))
    return MethodScores(
        method="judge_all_at_once", scores=out, n_abstained=abstained, model_calls=calls
    )


def judge_binary_search(steps: Sequence[LocalisationStep], backend: JudgeBackend) -> MethodScores:
    """Bisect the trace, asking which half holds the decisive step; ~log2(n) calls per trace."""
    out: list[StepScore] = []
    abstained = calls = 0
    for trace in by_trace(steps).values():
        lo, hi = 0, len(trace) - 1
        failed = False
        while lo < hi:
            mid = (lo + hi) // 2
            answer = backend.contains_error(_transcript(trace[lo : mid + 1], offset=lo))
            calls += 1
            if answer is None:
                failed = True
                break
            hi, lo = (mid, lo) if answer else (hi, mid + 1)
        if failed:
            abstained += 1
            out.extend(_pointed(trace, None))
        else:
            out.extend(_pointed(trace, lo))
    return MethodScores(
        method="judge_binary_search", scores=out, n_abstained=abstained, model_calls=calls
    )


def judge_step_by_step(steps: Sequence[LocalisationStep], backend: JudgeBackend) -> MethodScores:
    """Walk the trace, asking of each step in its preceding context; the first yes is the pick."""
    out: list[StepScore] = []
    abstained = calls = 0
    for trace in by_trace(steps).values():
        pick: int | None = None
        failed = False
        for i, s in enumerate(trace):
            answer = backend.is_error(_transcript(trace[:i], offset=0), _transcript([s], offset=i))
            calls += 1
            if answer is None:
                failed = True
                break
            if answer:
                pick = i
                break
        if failed:
            abstained += 1
            pick = None
        out.extend(_pointed(trace, pick))
    return MethodScores(
        method="judge_step_by_step", scores=out, n_abstained=abstained, model_calls=calls
    )


# --------------------------------------------------------------------------------------------
# The two CPVI regimes
# --------------------------------------------------------------------------------------------


def transfer_scores(
    steps: Sequence[LocalisationStep],
    featuriser: Featuriser,
    *,
    key: str,
    dir: Path | str | None,
    orientation: float | None,
) -> MethodScores:
    """Apply a frozen simulator-trained statistic to the logs, unchanged (the transfer regime).

    ``orientation`` is the sign ``StatisticCalibration`` recorded when the statistic was calibrated
    against realised outcomes (``oriented = orientation * raw``, higher = more failure-risk). It is
    required rather than defaulted: getting the sign wrong silently inverts every localisation
    number, so an absent orientation is ``unavailable``, not an assumption.

    The calibrated *threshold* is deliberately not transferred, only the orientation. A threshold is
    an operating point on the arena's score distribution; these corpora are free-text traces from a
    different distribution, so the cut-point does not carry even where the ordering does. Every
    metric downstream is rank-based for that reason, and this arm makes no pass/fail claim.
    """
    method = "cpvi_transfer"
    if dir is None:
        return MethodScores(
            method=method,
            status="unavailable",
            reason="no frozen simulator statistic: the arena track has not produced one yet",
        )
    if orientation is None:
        return MethodScores(
            method=method,
            status="unavailable",
            reason="statistic present but no calibrated orientation; refusing to guess the sign",
        )
    try:
        stat = load_statistic(resolve_statistic_key(key), dir=dir)
    except GateError as exc:
        return MethodScores(method=method, status="unavailable", reason=str(exc))
    e_s = featuriser.embed_texts([s.observation for s in steps])
    e_m = featuriser.embed_texts([s.message for s in steps])
    raw = stat.score(e_s, e_m)
    return MethodScores(
        method=method,
        scores=[
            StepScore(trace_id=s.trace_id, step=s.step, risk=float(orientation * v))
            for s, v in zip(steps, raw, strict=True)
        ],
    )


def refit_scores(
    steps: Sequence[LocalisationStep],
    featuriser: Featuriser,
    labels: Mapping[tuple[str, int], bool],
    cfg: ProbeConfig,
    *,
    label_source: str = "replay",
) -> MethodScores:
    """Refit probes on the logs themselves, cross-fit by trace (the refit regime).

    ``labels`` is the per-step outcome, keyed ``(trace_id, step)`` - from the counterfactual replay
    labeller (DSE-042) or, where replay has not run, from DSE-042's cheap trace-level fallback.
    ``label_source`` names which, because "no labels" and "labels that cannot separate" are
    different findings and a reason string that conflates them misreports the corpus. Either way it
    is *not* the corpus annotation: fitting on the annotation and then scoring localisation against
    it is the circularity this whole design avoids. Risk is the negated CPVI - a step whose message
    carries little conditional information is the suspect one.

    Both single-class labels and a single trace return ``not_applicable``: an all-failure corpus
    (Who&When is 184/184) has nothing for a probe to separate, and one trace cannot be held out.
    """
    method = "cpvi_refit"
    labelled = [s for s in steps if (s.trace_id, s.step) in labels]
    if not labelled:
        return MethodScores(
            method=method,
            status="unavailable",
            reason=f"no {label_source} outcome labels for these steps",
        )
    y = np.array([int(labels[(s.trace_id, s.step)]) for s in labelled], dtype=int)
    trace_ids = np.array([s.trace_id for s in labelled])
    groups = np.unique(trace_ids, return_inverse=True)[1].astype(int)
    if len(np.unique(y)) < 2:
        return MethodScores(
            method=method,
            status="not_applicable",
            reason=(
                f"{label_source} labels are single-class ({int(y[0])}) on this corpus; "
                "nothing to fit"
            ),
        )
    if len(np.unique(groups)) < 2:
        return MethodScores(
            method=method,
            status="not_applicable",
            reason="one trace only; cross-fitting requires at least two trace groups",
        )
    e_s = featuriser.embed_texts([s.observation for s in labelled])
    e_m = featuriser.embed_texts([s.message for s in labelled])
    scores = cpvi(e_s, e_m, y, groups, cfg)
    return MethodScores(
        method=method,
        scores=[
            StepScore(trace_id=s.trace_id, step=s.step, risk=float(-v))
            for s, v in zip(labelled, scores, strict=True)
        ],
        reason=(
            None
            if len(labelled) == len(steps)
            else f"scored the {len(labelled)} of {len(steps)} steps carrying a {label_source} label"
        ),
    )


def replay_labels(labellings: Sequence[object]) -> dict[tuple[str, int], bool]:
    """Flatten ``rq3a_replay.ReplayLabelling`` objects into the ``labels`` mapping refit wants.

    Steps whose replay did not complete (``outcome_failed is None``) are dropped rather than
    defaulted; a below-floor agreement label is *kept* and flagged upstream, as DSE-042 decided.
    """
    out: dict[tuple[str, int], bool] = {}
    for labelling in labellings:
        for label in getattr(labelling, "labels", []):
            if label.outcome_failed is not None:
                out[(label.trace_id, label.step)] = bool(label.outcome_failed)
    return out


# --------------------------------------------------------------------------------------------
# Localisation metrics
# --------------------------------------------------------------------------------------------


class LocalisationMetrics(BaseModel):
    """One method's localisation performance, with intervals and its eligibility bookkeeping."""

    model_config = ConfigDict(extra="forbid")

    method: str
    status: Status
    reason: str | None = None
    n_traces_scored: int = 0
    n_traces_evaluated: int = 0  # traces whose annotated step is inside the scored set
    n_traces_target_off_boundary: int = 0  # annotated step exists but was not scored
    n_abstained: int = 0
    model_calls: int = 0
    top_k: int = 0
    step_accuracy: float | None = None
    step_accuracy_ci: tuple[float, float] | None = None
    agent_accuracy: float | None = None
    agent_accuracy_ci: tuple[float, float] | None = None
    top_k_accuracy: float | None = None
    top_k_accuracy_ci: tuple[float, float] | None = None
    mrr: float | None = None
    mrr_ci: tuple[float, float] | None = None


def _ranks(risk: FloatArray) -> FloatArray:
    """Ranks with 1 = most suspect; ties take the average rank (see ``TIE_POLICY``)."""
    ranked: FloatArray = rankdata(-risk, method="average").astype(np.float64)
    return ranked


def evaluate(
    scores: MethodScores,
    steps: Sequence[LocalisationStep],
    targets: Mapping[str, TraceTarget],
    *,
    top_k: int = 3,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> LocalisationMetrics:
    """Score one method against the corpus annotations, one value per trace, intervals over traces.

    Traces are the resampling unit: steps inside a trace share a transcript, so an iid step
    bootstrap would read an interval far narrower than the data supports. Each trace contributes
    exactly one value per metric, so the trace bootstrap *is* the cluster bootstrap here.
    """
    if scores.status != "ok":
        return LocalisationMetrics(
            method=scores.method, status=scores.status, reason=scores.reason, top_k=top_k
        )
    risk = {(s.trace_id, s.step): s.risk for s in scores.scores}
    grouped = by_trace(steps)
    hit_step: list[float] = []
    hit_agent: list[float] = []
    hit_topk: list[float] = []
    recip: list[float] = []
    off_boundary = 0
    for trace_id, trace in grouped.items():
        target = targets.get(trace_id)
        if target is None or target.decisive_step is None:
            continue
        positions = {s.step: i for i, s in enumerate(trace)}
        if target.decisive_step not in positions:
            off_boundary += 1
            continue
        if any((s.trace_id, s.step) not in risk for s in trace):
            continue  # the method did not score this trace at all
        ranks = _ranks(np.array([risk[(s.trace_id, s.step)] for s in trace], dtype=np.float64))
        rank = float(ranks[positions[target.decisive_step]])
        hit_step.append(float(rank == 1.0))
        hit_topk.append(float(rank <= top_k))
        recip.append(1.0 / rank)
        if target.decisive_agent is not None:
            top = trace[int(np.argmin(ranks))]
            hit_agent.append(float(top.agent_name.strip() == target.decisive_agent.strip()))
    if not hit_step:
        return LocalisationMetrics(
            method=scores.method,
            status="not_applicable",
            reason="no trace carries an annotated decisive step inside the scored steps",
            n_traces_scored=len({s.trace_id for s in scores.scores}),
            n_traces_target_off_boundary=off_boundary,
            n_abstained=scores.n_abstained,
            model_calls=scores.model_calls,
            top_k=top_k,
        )

    def _mean_ci(values: list[float]) -> tuple[float | None, tuple[float, float] | None]:
        if not values:
            return None, None
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), bootstrap_ci(arr, n_boot=n_boot, alpha=alpha)

    step_acc, step_ci = _mean_ci(hit_step)
    agent_acc, agent_ci = _mean_ci(hit_agent)
    topk_acc, topk_ci = _mean_ci(hit_topk)
    mrr, mrr_ci = _mean_ci(recip)
    return LocalisationMetrics(
        method=scores.method,
        status="ok",
        reason=scores.reason,
        n_traces_scored=len({s.trace_id for s in scores.scores}),
        n_traces_evaluated=len(hit_step),
        n_traces_target_off_boundary=off_boundary,
        n_abstained=scores.n_abstained,
        model_calls=scores.model_calls,
        top_k=top_k,
        step_accuracy=step_acc,
        step_accuracy_ci=step_ci,
        agent_accuracy=agent_acc,
        agent_accuracy_ci=agent_ci,
        top_k_accuracy=topk_acc,
        top_k_accuracy_ci=topk_ci,
        mrr=mrr,
        mrr_ci=mrr_ci,
    )


# --------------------------------------------------------------------------------------------
# The MAST trace-level arm and the agreement audit
# --------------------------------------------------------------------------------------------


class MastCategoryResult(BaseModel):
    """MAST category prediction: the bits its text carries about inter-agent misalignment.

    Reported apart from the localisation table and never pooled with it. MAST has no per-step
    boundaries and no observation/message split, so it supports neither localisation nor CPVI
    (``cpvi_status``); what it does supply is the one genuinely two-class trace outcome in the RQ3a
    substrate, which is why the arm exists at all.
    """

    model_config = ConfigDict(extra="forbid")

    status: Status
    reason: str | None = None
    n_traces: int = 0
    n_positive: int = 0
    category_information_bits: float | None = None
    category_information_ci: tuple[float, float] | None = None
    cpvi_status: Literal["not_applicable"] = "not_applicable"
    cpvi_reason: str = MAST_CPVI_REASON


def mast_category(
    traces: Sequence[LogTraceRecord],
    featuriser: Featuriser,
    cfg: ProbeConfig,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> MastCategoryResult:
    """Cross-fit pointwise information of the trace text about the inter-agent-misalignment label.

    Not CPVI and not presented as it: there is no conditioning state to hold fixed, so this is the
    unconditional quantity against the class prior, cross-fit by trace and reported in bits with an
    interval. It answers "is this failure family visible in the transcript at all" on the only RQ3a
    corpus with both classes present - a substrate check for the localisation claim, not evidence
    for it.
    """
    labelled = [t for t in traces if t.annotations]
    if not labelled:
        return MastCategoryResult(status="unavailable", reason="no MAST annotations present")
    y = np.array(
        [
            int(any(v for k, v in t.annotations.items() if k.startswith(MAST_MISALIGNMENT_PREFIX)))
            for t in labelled
        ],
        dtype=int,
    )
    if len(np.unique(y)) < 2:
        return MastCategoryResult(
            status="not_applicable",
            reason="the inter-agent-misalignment label is single-class on this sample",
            n_traces=len(labelled),
            n_positive=int(y.sum()),
        )
    e = featuriser.embed_texts([t.trace_text for t in labelled])
    groups = np.arange(len(labelled), dtype=int)  # one trace per group: no trace trains on itself
    bits = pvi(e, y, groups, cfg)
    return MastCategoryResult(
        status="ok",
        n_traces=len(labelled),
        n_positive=int(y.sum()),
        category_information_bits=float(bits.mean()),
        category_information_ci=bootstrap_ci(bits, n_boot=n_boot, alpha=alpha),
    )


class AgreementAudit(BaseModel):
    """Cohen's kappa between the judge's selected agent and the corpus annotation.

    Named for what it is. Both label series are automated-versus-existing-annotation, so this is a
    judge-validity check on a sampled subset, **not** a newly collected human double-annotation
    study; a genuine one (two raters, frozen rubric, adjudication policy) remains outstanding and is
    not claimed here.
    """

    model_config = ConfigDict(extra="forbid")

    comparison: str = "judge-selected agent vs the corpus's existing annotation"
    rule: str = "uniform sample of annotated traces, without replacement, seeded"
    seed: int = 0
    n_sampled: int = 0
    n_categories: int = 0
    kappa: float | None = None
    reason: str | None = None


def judge_agreement(
    scores: MethodScores,
    steps: Sequence[LocalisationStep],
    targets: Mapping[str, TraceTarget],
    *,
    n_sample: int = 30,
    seed: int = 0,
) -> AgreementAudit:
    """Sample annotated traces and compare the judge's top-ranked agent with the annotation."""
    if scores.status != "ok":
        return AgreementAudit(seed=seed, reason=f"judge unavailable: {scores.reason}")
    risk = {(s.trace_id, s.step): s.risk for s in scores.scores}
    eligible = [
        (tid, trace)
        for tid, trace in sorted(by_trace(steps).items())
        if targets.get(tid) is not None
        and targets[tid].decisive_agent is not None
        and all((s.trace_id, s.step) in risk for s in trace)
    ]
    if not eligible:
        return AgreementAudit(seed=seed, reason="no annotated trace was scored by the judge")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))[: min(n_sample, len(eligible))]
    judged: list[str] = []
    annotated: list[str] = []
    for i in sorted(idx.tolist()):
        tid, trace = eligible[i]
        ranks = _ranks(np.array([risk[(s.trace_id, s.step)] for s in trace], dtype=np.float64))
        judged.append(trace[int(np.argmin(ranks))].agent_name.strip())
        agent = targets[tid].decisive_agent
        annotated.append(agent.strip() if agent is not None else "")
    categories = sorted(set(judged) | set(annotated))
    if len(categories) < 2:
        return AgreementAudit(
            seed=seed,
            n_sampled=len(judged),
            n_categories=len(categories),
            reason="one agent category in the sample; kappa is undefined",
        )
    return AgreementAudit(
        seed=seed,
        n_sampled=len(judged),
        n_categories=len(categories),
        kappa=float(cohen_kappa_score(annotated, judged, labels=categories)),
    )


# --------------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------------


class RQ3aConfig(BaseModel):
    """Analysis knobs. The probe family is RQ1's, unchanged - *V* is frozen across substrates."""

    model_config = ConfigDict(extra="forbid")

    probe: ProbeConfig = Field(default_factory=lambda: ProbeConfig(n_repeats=5))
    top_k: int = Field(default=3, ge=1)
    n_boot: int = Field(default=2000, ge=100)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    handoffs_only: bool = True
    audit_sample: int = Field(default=30, ge=1)
    audit_seed: int = Field(default=0, ge=0)
    # Transfer regime inputs; absent means the regime reports "unavailable" rather than guessing.
    # The key is a `Statistic.key` ("fail"/"cosine"), not the paper's s_* notation; it defaulted to
    # "s_info" until DSE-024, which no `load_statistic` could ever have resolved.
    transfer_key: str = "fail"
    transfer_dir: Path | None = None
    transfer_orientation: float | None = None


class RQ3aResult(BaseModel):
    """The RQ3a comparison: every method against the annotations, plus MAST and the audit."""

    model_config = ConfigDict(extra="forbid")

    corpus: str
    n_traces: int
    n_steps_scored: int
    handoffs_only: bool
    tie_policy: str = TIE_POLICY
    provenance: AnalysisProvenance
    methods: list[LocalisationMetrics]
    outcomes: OutcomeCensus
    judge: JudgeIdentity | None = None
    mast: MastCategoryResult | None = None
    agreement: AgreementAudit | None = None


def analyse_rq3a(
    records: Sequence[LogHandoffRecord],
    featuriser: Featuriser,
    *,
    cfg: RQ3aConfig | None = None,
    judge: JudgeBackend | None = None,
    labels: Mapping[tuple[str, int], bool] | None = None,
    mast_traces: Sequence[LogTraceRecord] | None = None,
) -> RQ3aResult:
    """Score every method on one corpus and evaluate them against its annotations.

    ``judge`` absent skips the three published-method replications (they are the only methods that
    cost model calls); ``mast_traces`` absent omits the trace-level arm. Each omission is visible in
    the result rather than silently changing what the table compares.

    ``labels`` absent does *not* leave the refit regime unavailable by default: it falls back to the
    corpus's own trace-level outcome, which is what DSE-042 provides for when replay is cut. The
    census on the result then records whether that fallback had two classes to separate, so an
    unavailable refit row carries the measured reason instead of an assumed one.
    """
    cfg = cfg or RQ3aConfig()
    if not records:
        raise ConfigError("analyse_rq3a called with no records")
    steps = localisation_steps(records, handoffs_only=cfg.handoffs_only)
    if not steps:
        raise RQ3aError("no steps to score: the corpus has no inter-agent handoffs")
    targets = trace_targets(records)
    label_source = "replay" if labels is not None else "trace-outcome"
    refit_labels = labels if labels is not None else trace_outcome_labels(records)
    census = outcome_census(records, source=label_source)

    produced = [
        schema_validity_scores(steps),
        cosine_scores(steps, featuriser),
        transfer_scores(
            steps,
            featuriser,
            key=cfg.transfer_key,
            dir=cfg.transfer_dir,
            orientation=cfg.transfer_orientation,
        ),
        refit_scores(steps, featuriser, refit_labels, cfg.probe, label_source=label_source),
    ]
    judge_scores: MethodScores | None = None
    if judge is not None:
        judge_scores = judge_all_at_once(steps, judge)
        produced.extend(
            [judge_scores, judge_binary_search(steps, judge), judge_step_by_step(steps, judge)]
        )

    methods = [
        evaluate(s, steps, targets, top_k=cfg.top_k, n_boot=cfg.n_boot, alpha=cfg.alpha)
        for s in produced
    ]
    return RQ3aResult(
        corpus=records[0].corpus,
        n_traces=len({s.trace_id for s in steps}),
        n_steps_scored=len(steps),
        handoffs_only=cfg.handoffs_only,
        provenance=build_provenance(featuriser.cfg, cfg.probe),
        methods=methods,
        outcomes=census,
        judge=None
        if judge is None
        else JudgeIdentity(
            model_name=judge.model_name,
            model_revision=judge.model_revision,
            decoding=judge.decoding,
        ),
        mast=None
        if mast_traces is None
        else mast_category(mast_traces, featuriser, cfg.probe, n_boot=cfg.n_boot, alpha=cfg.alpha),
        agreement=None
        if judge_scores is None
        else judge_agreement(
            judge_scores, steps, targets, n_sample=cfg.audit_sample, seed=cfg.audit_seed
        ),
    )


def results_table(result: RQ3aResult) -> pd.DataFrame:
    """The comparison table: one row per method, unavailable rows retained with their reason."""
    return pd.DataFrame([m.model_dump() for m in result.methods])


def write_rq3a(result: RQ3aResult, dir: Path | str) -> Path:
    """Persist the analysis JSON and the comparison table."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "rq3a.json").write_text(result.model_dump_json(indent=2))
    results_table(result).to_csv(dir / "rq3a_localisation.csv", index=False)
    return dir


def manifest_metrics(result: RQ3aResult) -> dict[str, object]:
    """The RQ3a block for ``RunManifest.metrics``, carrying the judge substitution and tie policy.

    In ``metrics`` rather than as new manifest fields: the manifest schema is this repo's stable
    reproducibility contract and one experiment's metadata does not belong on every run.
    """
    return {
        "rq3a": {
            "corpus": result.corpus,
            "n_traces": result.n_traces,
            "n_steps_scored": result.n_steps_scored,
            "handoffs_only": result.handoffs_only,
            "tie_policy": result.tie_policy,
            "judge": None if result.judge is None else result.judge.model_dump(mode="json"),
            "statuses": {m.method: m.status for m in result.methods},
            "unavailable": {
                m.method: m.reason for m in result.methods if m.status != "ok" and m.reason
            },
            "outcomes": result.outcomes.model_dump(mode="json"),
            "agreement": None
            if result.agreement is None
            else result.agreement.model_dump(mode="json"),
        }
    }
