"""Counterfactual-replay outcome labelling for real multi-agent logs (DSE-042).

CPVI needs a per-handoff outcome that is **not** the corpus's attribution annotation, or the
localisation claim is circular. Replay defines one interventionally: re-run the system from step
*t* with that step's output substituted, and record whether the trace outcome changes.

This is load-bearing rather than an upgrade path. ``docs/rq3a_schema_mapping.md`` §5 records why:
TraceElephant is 220/220 failures at trace level and Who&When 184/184, so Y1 (trace success) is
degenerate on both, and Y2 (annotation-as-Y) is forbidden for circularity. Replay is the only route
to a within-trace two-class target on a corpus that also supplies the per-step conditioning state.

Three disciplines are structural here, not conventions:

* **The labeller cannot read annotations.** It consumes ``ReplayStep``, a four-field view built by
  ``replay_steps``; the field does not exist on the type, so no call path can reach it. This
  mirrors ``measure.twin.prospective_twin``, which cannot reach Y for the same reason.
* **A dry run issues no backend calls.** ``project`` takes no backend argument, so "dry run" is a
  property of the signature rather than of a flag someone must remember to pass.
* **The cap is enforced at the call site, not on the forecast.** A projection is a forecast; a
  mis-wired loop is what actually burns an allocation. ``_Meter`` is consulted immediately before
  every backend invocation, so an accidental full-corpus replay stops at the cap.

Budgets are counted in **model calls and elapsed seconds** - never currency. Every call this repo
makes is local or on the Myriad allocation, so a monetary figure would be an invented number
dressed as a measurement. GPU seconds appear only as an advisory estimate carrying its calibration
source.

Running replay at corpus scale is a budgeted experiment, not part of this module (DSE-042 scope).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.logs import LogHandoffRecord

logger = logging.getLogger(__name__)

# The stratification key: the trace-level outcome, whose balance the sample must not disturb.
# ``None`` is a stratum in its own right and by far the largest on TraceElephant (176 of 220), so
# folding it into either class would silently reweight the corpus.
Stratum = bool | None


class ReplayError(RuntimeError):
    """A replay plan or backend result is unusable. Raised rather than defaulted."""


# --------------------------------------------------------------------------------------------
# The annotation-free view the labeller consumes
# --------------------------------------------------------------------------------------------


class ReplayStep(BaseModel):
    """One step, reduced to what a replay needs: where it sits, what was seen, what was said.

    ``LogHandoffRecord.annotations`` is deliberately absent rather than empty. The labeller's
    output becomes the outcome the localisation analysis scores the annotations *against*, so a
    labeller that could read them would be scoring a target against itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    step: int = Field(ge=0)
    observation: str
    message: str


def replay_steps(records: Sequence[LogHandoffRecord]) -> list[ReplayStep]:
    """Project log records onto the replay view, dropping annotations and the outcome."""
    return [
        ReplayStep(trace_id=r.trace_id, step=r.step, observation=r.observation, message=r.message)
        for r in records
    ]


def trace_success_labels(records: Sequence[LogHandoffRecord]) -> dict[str, bool | None]:
    """The cheap label: per-trace failure as the corpus records it, computed for every trace.

    Kept unconditional so the refit arm survives replay being cut for budget. It is degenerate on
    both per-step corpora (§5 of the schema mapping) - that is precisely why replay exists - but a
    degenerate fallback that is present beats a missing one, and MAST is genuinely two-class.
    """
    return {r.trace_id: r.trace_failed for r in records}


# --------------------------------------------------------------------------------------------
# Budget: what the run may spend, and what it is forecast to spend
# --------------------------------------------------------------------------------------------


class ReplayBudget(BaseModel):
    """The hard limits. Enforced at the call site; recorded verbatim in the manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = Field(ge=1)
    # Optional wall-clock guard, for a queue slot rather than an allocation. None disables it.
    max_elapsed_s: float | None = Field(default=None, gt=0)


class ReplayPlan(BaseModel):
    """The planning assumptions a projection rests on, declared so the forecast is auditable.

    ``calls_per_replay`` is a (min, max) band because one replay is not one call: an agent run can
    branch, retry, or invoke tools. The band is a declared assumption about the system under
    replay, not a measurement, which is why the projection reports both ends rather than a point.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    replays_per_step: int = Field(default=5, ge=1)
    calls_per_replay: tuple[int, int] = (1, 1)
    # Advisory only. None when no throughput calibration exists yet - the honest default, since a
    # made-up seconds/call would read as measured once it reached a table.
    seconds_per_call: float | None = Field(default=None, gt=0)
    calibration_source: str | None = None


class ReplayProjection(BaseModel):
    """What a dry run reports. Produced without touching a backend."""

    model_config = ConfigDict(extra="forbid")

    selected_steps: int
    replays_per_step: int
    projected_calls_min: int
    projected_calls_max: int
    max_model_calls: int
    permitted: bool
    # Non-empty exactly when ``permitted`` is False.
    refusal: str = ""
    estimated_gpu_seconds: float | None = None
    calibration_source: str | None = None


def project(selected_steps: int, plan: ReplayPlan, budget: ReplayBudget) -> ReplayProjection:
    """Forecast the spend of a replay run. Takes no backend, so it cannot issue a call.

    Refusal is decided on the projected **minimum**: if even the cheapest possible execution
    exceeds the cap, the run is refused before anything is sent, rather than started and killed
    part-way with a half-labelled corpus to explain.
    """
    if selected_steps < 0:
        raise ReplayError(f"selected_steps must be non-negative, got {selected_steps}")
    lo, hi = plan.calls_per_replay
    if lo < 1 or hi < lo:
        raise ReplayError(
            f"calls_per_replay must satisfy 1 <= min <= max, got {plan.calls_per_replay}"
        )

    replays = selected_steps * plan.replays_per_step
    calls_min, calls_max = replays * lo, replays * hi
    permitted = calls_min <= budget.max_model_calls
    refusal = (
        ""
        if permitted
        else (
            f"projected minimum {calls_min} model calls exceeds the cap "
            f"{budget.max_model_calls}; no replay was executed"
        )
    )
    seconds = None if plan.seconds_per_call is None else calls_max * plan.seconds_per_call
    return ReplayProjection(
        selected_steps=selected_steps,
        replays_per_step=plan.replays_per_step,
        projected_calls_min=calls_min,
        projected_calls_max=calls_max,
        max_model_calls=budget.max_model_calls,
        permitted=permitted,
        refusal=refusal,
        estimated_gpu_seconds=seconds,
        calibration_source=plan.calibration_source,
    )


class _Meter:
    """Runtime budget state. Mutable, private, and consulted before every backend call."""

    def __init__(self, budget: ReplayBudget) -> None:
        self.budget = budget
        self.calls = 0
        self._t0 = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    def may_call(self) -> bool:
        """Is there room for at least one more backend invocation?

        Checked *before* sending. A single replay may consume several calls, so the final count can
        exceed the cap by at most one replay's worth - the alternative is reserving a worst-case
        band up front and refusing runs that would in fact have fitted.
        """
        if self.calls >= self.budget.max_model_calls:
            return False
        limit = self.budget.max_elapsed_s
        return limit is None or self.elapsed_s < limit


# --------------------------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------------------------


class StratifiedSample(BaseModel):
    """A subset of steps to replay, plus the rule that produced it, for the manifest."""

    model_config = ConfigDict(extra="forbid")

    steps: list[tuple[str, int]]  # (trace_id, step)
    rule: str
    seed: int
    # Stratum key rendered as a string ("True"/"False"/"None") so it survives JSON round-tripping.
    allocated: dict[str, int]
    available: dict[str, int]


def stratified_sample(
    records: Sequence[LogHandoffRecord], n: int, *, seed: int
) -> StratifiedSample:
    """Sample ``n`` steps stratified on the trace-level outcome, preserving its balance.

    Proportional allocation with largest-remainder rounding, so the strata sum to exactly ``n``
    instead of drifting by a step or two per stratum. Sampling is over whatever records are passed
    in: filter to ``is_handoff`` first when the analysis wants handoffs only, since which steps are
    eligible is the caller's question and the base rate depends on the answer.
    """
    if n < 0:
        raise ReplayError(f"n must be non-negative, got {n}")
    if not records:
        raise ReplayError("no records to sample")

    strata: dict[Stratum, list[tuple[str, int]]] = {}
    for r in records:
        strata.setdefault(r.trace_failed, []).append((r.trace_id, r.step))
    total = len(records)
    n = min(n, total)

    # Largest remainder: floor everywhere, then hand the leftover places to the biggest fractions.
    exact = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    leftover = n - sum(alloc.values())
    for k in sorted(strata, key=lambda k: exact[k] - alloc[k], reverse=True)[:leftover]:
        alloc[k] += 1

    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, int]] = []
    for key in sorted(strata, key=str):
        pool = strata[key]
        take = alloc[key]
        idx: list[int] = (
            [int(i) for i in rng.choice(len(pool), size=take, replace=False)] if take else []
        )
        chosen.extend(pool[i] for i in idx)

    return StratifiedSample(
        steps=chosen,
        rule="proportional allocation over trace_failed (True/False/None), largest-remainder "
        "rounding, uniform without replacement within each stratum",
        seed=seed,
        allocated={str(k): v for k, v in alloc.items()},
        available={str(k): len(v) for k, v in strata.items()},
    )


# --------------------------------------------------------------------------------------------
# The backend seam and the labeller
# --------------------------------------------------------------------------------------------


class ReplayOutcome(BaseModel):
    """One replay's result: did the trace fail, and what did it cost to find out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failed: bool
    model_calls: int = Field(ge=1)


class ReplayBackend(ABC):
    """Re-runs a system from one step. One concrete per environment; a stub serves the tests.

    Kept behind an interface because executing TraceElephant's environments is a budgeted
    experiment outside this ticket, while the labelling logic above it must be testable today.
    """

    @abstractmethod
    def replay(self, step: ReplayStep, attempt: int) -> ReplayOutcome:
        """Re-run from ``step`` once. ``attempt`` distinguishes repeats of the same step."""


class StepLabel(BaseModel):
    """The per-step outcome a replay produced, with the agreement that qualifies it."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    step: int
    # None when no replay completed - the budget ran out before this step was reached.
    outcome_failed: bool | None
    agreement: float  # majority share over completed replays, 0.5-1.0
    n_replays: int  # completed, which is not necessarily the number requested
    below_floor: bool  # flagged, never dropped
    budget_exhausted: bool


class ReplayLabelling(BaseModel):
    """Everything a replay run produced, including what it actually spent."""

    model_config = ConfigDict(extra="forbid")

    labels: list[StepLabel]
    model_calls: int
    elapsed_s: float
    stopped_on_budget: bool
    budget: ReplayBudget
    agreement_floor: float


def label_by_replay(
    trace: Sequence[ReplayStep],
    step_ids: Sequence[int],
    n_replays: int,
    budget: ReplayBudget,
    *,
    backend: ReplayBackend,
    agreement_floor: float = 0.6,
) -> ReplayLabelling:
    """Label the named steps of one trace by majority vote over ``n_replays`` counterfactuals.

    Replay is non-deterministic - the systems are LLM-driven - so determinism here means the same
    thing it means for the simulator: low-variance and agreement-reported, never bit-exact. Steps
    whose agreement falls below ``agreement_floor`` are **flagged, not dropped**: a silently
    removed step changes the base rate of every count downstream, and the disagreement is itself
    the signal that the step's outcome is not well defined.
    """
    if n_replays < 1:
        raise ReplayError(f"n_replays must be at least 1, got {n_replays}")
    if not 0.0 < agreement_floor <= 1.0:
        raise ReplayError(f"agreement_floor must lie in (0, 1], got {agreement_floor}")

    by_step = {s.step: s for s in trace}
    missing = sorted(set(step_ids) - set(by_step))
    if missing:
        raise ReplayError(f"step_ids not present in the trace: {missing}")

    meter = _Meter(budget)
    labels: list[StepLabel] = []
    stopped = False

    for step_id in step_ids:
        step = by_step[step_id]
        failures = 0
        done = 0
        for attempt in range(n_replays):
            if not meter.may_call():
                stopped = True
                break
            outcome = backend.replay(step, attempt)
            meter.calls += outcome.model_calls
            failures += outcome.failed
            done += 1

        if done == 0:
            labels.append(
                StepLabel(
                    trace_id=step.trace_id,
                    step=step_id,
                    outcome_failed=None,
                    agreement=0.0,
                    n_replays=0,
                    below_floor=True,
                    budget_exhausted=True,
                )
            )
            continue

        # Strict majority for `failed`; a tie therefore reads False and scores 0.5 agreement, which
        # is below any sensible floor and so arrives flagged rather than silently resolved.
        majority_failed = failures * 2 > done
        agreement = max(failures, done - failures) / done
        labels.append(
            StepLabel(
                trace_id=step.trace_id,
                step=step_id,
                outcome_failed=majority_failed,
                agreement=agreement,
                n_replays=done,
                below_floor=agreement < agreement_floor,
                budget_exhausted=done < n_replays,
            )
        )

    if stopped:
        logger.warning(
            "replay stopped on budget after %d model calls (%.1fs); %d steps labelled",
            meter.calls,
            meter.elapsed_s,
            len(labels),
        )

    return ReplayLabelling(
        labels=labels,
        model_calls=meter.calls,
        elapsed_s=meter.elapsed_s,
        stopped_on_budget=stopped,
        budget=budget,
        agreement_floor=agreement_floor,
    )


def manifest_metrics(
    projection: ReplayProjection,
    sample: StratifiedSample,
    plan: ReplayPlan,
    labelling: ReplayLabelling | None = None,
) -> dict[str, object]:
    """The replay block for ``RunManifest.metrics``: forecast, sampling rule, and realised spend.

    Written into ``metrics`` rather than as new ``RunManifest`` fields. The manifest schema is this
    repo's stable reproducibility contract, and widening it is a result-affecting change that
    re-keys nothing here - a replay run is one experiment's metadata, not a new mandatory field on
    every run in the project.
    """
    block: dict[str, object] = {
        "plan": plan.model_dump(mode="json"),
        "projection": projection.model_dump(mode="json"),
        "sampling": {
            "rule": sample.rule,
            "seed": sample.seed,
            "allocated": sample.allocated,
            "available": sample.available,
            "n_selected": len(sample.steps),
        },
    }
    if labelling is not None:
        block["realised"] = {
            "model_calls": labelling.model_calls,
            "elapsed_s": round(labelling.elapsed_s, 3),
            "stopped_on_budget": labelling.stopped_on_budget,
            "n_labelled": len(labelling.labels),
            "n_below_floor": sum(x.below_floor for x in labelling.labels),
            "agreement_floor": labelling.agreement_floor,
        }
    return {"replay": block}


def render_projection(projection: ReplayProjection, sample: StratifiedSample) -> str:
    """The dry run's human-readable report - the thing a person reads before spending an hour."""
    p = projection
    strata = ", ".join(
        f"{k}={sample.allocated.get(k, 0)}/{v}" for k, v in sorted(sample.available.items())
    )
    gpu = (
        "unknown (no throughput calibration)"
        if p.estimated_gpu_seconds is None
        else (f"{p.estimated_gpu_seconds:,.0f}")
    )
    lines = [
        "Replay dry run",
        "",
        f"Eligible steps:              {sum(sample.available.values()):,}",
        f"Selected steps:              {p.selected_steps:,}",
        f"Sampling rule:               {sample.rule}",
        f"Sampling seed:               {sample.seed}",
        f"Strata (allocated/available):  {strata}",
        f"Replays per selected step:   {p.replays_per_step}",
        f"Projected model calls:       {p.projected_calls_min:,}-{p.projected_calls_max:,}",
        f"Hard model-call cap:         {p.max_model_calls:,}",
        f"Estimated GPU seconds:       {gpu}  (advisory, not a guarantee)",
        f"Calibration:                 {p.calibration_source or 'none declared'}",
        "",
        f"Decision:                    {'PERMITTED' if p.permitted else 'REFUSED - ' + p.refusal}",
    ]
    return "\n".join(lines)
