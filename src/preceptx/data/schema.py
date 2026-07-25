"""The per-handoff record schema - the stable contract every downstream ticket imports.

A ``HandoffRecord`` captures one A->B coordination handoff: the physics state either side of the
acted step, the raw and post-channel message, the structured action, and outcome flags. The four
``Y`` labels are placeholders here, populated by the post-episode labeller (DSE-009); downstream
code imports this schema and never redefines its fields. Treat the schema as versioned: a breaking
change bumps ``SCHEMA_VERSION`` and the loader handles both shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 2
"""Bump on any breaking change to ``HandoffRecord``; ``data.writer.load_dataset`` keys off it.

v2: adds ``observation`` (the receiver-conditioned state, P0-1), the DSE-018 gate fields
(``gate_blocked``/``gate_retries``/``message_blocked``, P0-4), and ``y_window_truncated`` (P1-12),
bundled as one deliberate bump before any real dataset exists. No v1 loader shim: no v1 data
was ever recorded.
"""

Condition = Literal["C0", "C1", "C2", "C3", "C4"]
Serialisation = Literal["numeric", "grid", "nl"]
Difficulty = Literal["easy", "medium", "hard"]

# Structured physics/action payloads. Values are JSON scalars, lists, or nested dicts thereof; the
# exact physics keys are owned by the simulator (DSE-006+), not pinned here.
StatePayload = dict[str, Any]


class HandoffRecord(BaseModel):
    """One A->B handoff: state, message (raw and delivered), action, and outcome flags."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION

    # Identity and experimental cell.
    episode_id: str
    step: int = Field(ge=0)
    condition: Condition
    serialisation: Serialisation
    difficulty: Difficulty
    model: str
    seed: int = Field(ge=0)

    # State as seen at the handoff: structured physics dict plus its serialised prompt form. Both
    # are kept so the serialisation A/B (DSE-008) is recoverable from the dataset alone.
    state: StatePayload
    state_str: str

    # B's delivered view at the handoff - the CPVI conditioning state s (P0-1 semantics): the state
    # observable to the RECEIVER, which under C3 is the restricted window, by design. Equals
    # ``state_str`` in C0/C1/C2/C4. Keeping both gives the C3 dual-baseline diagnostic (CPVI vs the
    # receiver's window vs the full state) for free.
    observation: str

    # The A->B message, before and after the channel degrades it (DSE-011).
    message_raw: str
    message_delivered: str

    # Runtime-gate outcome at this handoff (DSE-018; defaults = no gate in the loop). Reserved in
    # the v2 bump so the gate lands without a second schema change.
    gate_blocked: bool = False
    gate_retries: int = Field(default=0, ge=0)
    message_blocked: str | None = None

    # B's structured action and the physics either side of applying it.
    action: StatePayload
    pre_state: StatePayload
    post_state: StatePayload

    # Per-step outcome signals.
    progress: float
    success: bool
    collision: bool
    stuck: bool

    # The four Y labels, filled by the post-episode labeller (DSE-009). None until then.
    y_binary_progress: bool | None = None
    y_continuous_displacement: float | None = None
    y_discrete_config: int | None = None
    y_terminal_success: bool | None = None
    # True when the forward window for the progress labels was clamped at episode end (P1-12), so
    # truncated-horizon labels are visible to the analysis rather than silently shortened.
    y_window_truncated: bool | None = None
