"""The per-step record schema for real multi-agent logs (RQ3a).

Deliberately separate from :mod:`preceptx.data.schema`. ``HandoffRecord`` is the simulator's
reproducibility contract and carries physics; a log record is a different contract with no physics
at all, so it is versioned on its own counter rather than widening the simulator schema with
nullable fields. Physics fields here are **absent, not nullable** - a log row cannot accidentally
be read as a degraded episode row.

``trace_id`` is the grouping key for cross-fitting, the exact analogue of ``episode_id``: no trace
may straddle a train/test split. Native corpus annotations ride along in ``annotations`` so the
localisation analysis can score against them, and are **never** featurised - the labeller and the
probe both take ``observation`` and ``message`` only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LOG_SCHEMA_VERSION = 1
"""Bump on any breaking change to ``LogHandoffRecord`` or ``LogTraceRecord``.

Independent of ``data.schema.SCHEMA_VERSION``: the simulator and log substrates are versioned
separately so a change to one never re-keys datasets of the other.
"""

Corpus = Literal["traceelephant", "who_and_when", "mast"]


class LogHandoffRecord(BaseModel):
    """One step of a real multi-agent trace, in the shape the CPVI estimator consumes.

    ``observation`` is the receiver-observed context at the step (the conditioning state *s*) and
    ``message`` is what the acting component emitted (the message *m*). Together they are the only
    two fields the featuriser ever sees.
    """

    model_config = ConfigDict(extra="forbid")

    log_schema_version: int = LOG_SCHEMA_VERSION
    corpus: Corpus

    # Identity. ``trace_id`` groups cross-fit folds; ``step`` is the corpus's own step ordinal.
    trace_id: str
    step: int = Field(ge=0)

    # The acting component at this step, and the one that acts next. ``receiver`` is None on the
    # final step, where there is no successor to hand off to.
    agent_name: str
    agent_id: str | None = None
    receiver: str | None = None

    # True when the acting component changes between this step and the next - an inter-agent
    # handoff. False marks an intra-agent tool turn, which stays in the dataset (dropping it would
    # bias the base rate) but is separable at analysis time.
    is_handoff: bool

    # The CPVI pair.
    observation: str
    message: str

    # True only on the Who&When path, where no per-step input context exists and the observation is
    # rebuilt from the preceding messages. Downstream analysis must never pool reconstructed and
    # true conditioning state without reporting the split.
    reconstructed_observation: bool = False

    # Trace-level outcome where the corpus supplies one that does not read the annotations.
    # None where the corpus records no annotation-free outcome.
    trace_failed: bool | None = None

    # Native corpus labels (mistake agent/step/reason, MAST modes). Evaluation targets only.
    annotations: dict[str, Any] = Field(default_factory=dict)


class LogTraceRecord(BaseModel):
    """A whole trace as one row, for corpora that publish traces as an unsegmented transcript.

    MAST-Data is the trace-level secondary: its ``trajectory`` is a single formatted string whose
    layout differs per multi-agent system, so per-step extraction would need seven bespoke parsers
    and would invent step boundaries the corpus does not record. Reporting it at trace level is the
    honest resolution, not a lesser one.
    """

    model_config = ConfigDict(extra="forbid")

    log_schema_version: int = LOG_SCHEMA_VERSION
    corpus: Corpus
    trace_id: str
    system_name: str
    model_name: str | None = None
    benchmark: str | None = None
    trace_text: str
    trace_failed: bool | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
