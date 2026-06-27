"""The A->B communication channel: the degradation ladder C0-C4 (plus a C5 stub) (DSE-011).

``apply_channel`` touches the A->B message and B's observation only - never physics or the action
path (CLAUDE.md): any outcome difference between conditions is then attributable to the channel.
C0 passes through; C1 caps length; C2 delays delivery by exactly one step; C3 restricts B's
observation to a local window (the message is untouched - this is what forces the message to carry
non-state information); C4 applies seeded token dropout. The delivered message is recorded as
``message_delivered``; every degradation is deterministic given the seed.
"""

from __future__ import annotations

from typing import NamedTuple

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from preceptx.data.schema import Condition, Serialisation

_C2_SENTINEL = "(no message yet)"  # B's delivered message at step 0 under the one-step delay


class ChannelConfig(BaseModel):
    """Degradation parameters for the channel ladder; recorded in the run manifest."""

    model_config = ConfigDict(extra="forbid")

    c1_max_tokens: int = Field(default=8, gt=0)  # C1 whitespace-token cap on the message
    c3_window_rows: int = Field(default=2, ge=0)  # C3 grid rows kept either side of the load
    c4_dropout: float = Field(default=0.4, ge=0.0, le=1.0)  # C4 per-token drop probability
    c5_enabled: bool = False  # C5 supervisor relay (full impl DSE-026); stubbed off for now


class ChannelResult(NamedTuple):
    """What the channel yields: the message B reads, an optional observation override (C3), and the
    message to buffer for next step (C2 only)."""

    message_delivered: str
    observation_override: str | None = None
    new_buffer: str | None = None


def apply_channel(
    message_raw: str,
    condition: Condition,
    *,
    serialisation: Serialisation,
    observation: str,
    cfg: ChannelConfig,
    rng: Generator,
    buffered: str | None,
) -> ChannelResult:
    """Degrade the A->B message / B's observation for ``condition``. Pure and seed-deterministic."""
    if condition == "C0":
        return ChannelResult(message_delivered=message_raw)
    if condition == "C1":
        return ChannelResult(message_delivered=_cap_tokens(message_raw, cfg.c1_max_tokens))
    if condition == "C2":
        delivered = buffered if buffered is not None else _C2_SENTINEL
        return ChannelResult(message_delivered=delivered, new_buffer=message_raw)
    if condition == "C3":
        return ChannelResult(
            message_delivered=message_raw,
            observation_override=_restrict(observation, serialisation, cfg.c3_window_rows),
        )
    return ChannelResult(message_delivered=_drop_tokens(message_raw, cfg.c4_dropout, rng))  # C4


def _cap_tokens(text: str, max_tokens: int) -> str:
    # ponytail: whitespace-token cap; swap to the model tokenizer if exact token budgets matter.
    return " ".join(text.split()[:max_tokens])


def _drop_tokens(text: str, p: float, rng: Generator) -> str:
    return " ".join(t for t in text.split() if rng.random() >= p)


def _restrict(observation: str, serialisation: Serialisation, window_rows: int) -> str:
    """Restrict B's observation to a local window: a band around the load (grid) or self-state only
    (numeric / nl), so the goal and global layout must come from A's message - the C3 asymmetry."""
    if serialisation == "grid":
        return _window_grid(observation, window_rows)
    if serialisation == "numeric":
        return "\n".join(line for line in observation.splitlines() if not line.startswith("goal="))
    # nl: keep only the first sentence (the load-pose clause), dropping the goal / slit clauses.
    head = observation.split(". ", 1)[0]
    return head if head.endswith(".") else head + "."


def _window_grid(grid: str, window_rows: int) -> str:
    rows = grid.splitlines()
    load_rows = [i for i, row in enumerate(rows) if "T" in row]
    if not load_rows:  # load off-grid: nothing to window (cannot happen in a valid scene)
        return grid
    lo = max(0, min(load_rows) - window_rows)
    hi = min(len(rows), max(load_rows) + window_rows + 1)
    return "\n".join(rows[lo:hi])
