"""Versioned prompts for the two-agent negotiation loop (DSE-010).

A (the navigator) sees the full serialised state and emits one natural-language handoff for B.
B (the actuator) sees its - possibly channel-degraded - observation plus A's message and chooses one
macro-action via guided JSON decoding. The templates are versioned: a wording change is
result-affecting, so ``PROMPT_VERSION`` is bumped and recorded in the run manifest. The frozen
``HandoffRecord`` schema has no prompt field, so the prompt version lives with the run, not the
record (``SweepManifest.prompt_version``).
"""

from __future__ import annotations

from preceptx.serving.client import ChatMessage

PROMPT_VERSION = "v1"

_SYSTEM_A = (
    "You are agent A, the navigator in a two-agent cooperative-transport task. A T-shaped load "
    "must be pushed rightward through narrow slits between chambers to a goal region. You can see "
    "the scene but cannot act. Send agent B one short, concrete instruction (one or two "
    "sentences): the direction to push, or whether to rotate the load to clear a slit. Be brief."
)

_SYSTEM_B = (
    "You are agent B, the actuator in a two-agent cooperative-transport task. You receive a "
    "partial observation of the scene and one instruction from agent A. Choose exactly one "
    "macro-action that best advances the T-load toward the goal."
)

_ACTION_HINT = "Actions: N/S/E/W push the load; ROT+/ROT- rotate it; WAIT does nothing. Choose one."


def prompt_a(state_str: str) -> list[ChatMessage]:
    """A's chat: observe the full serialised state, emit a natural-language handoff to B."""
    return [
        ChatMessage(role="system", content=_SYSTEM_A),
        ChatMessage(role="user", content=f"Current scene:\n{state_str}\n\nYour instruction to B:"),
    ]


def prompt_b(observation: str, message: str) -> list[ChatMessage]:
    """B's chat: observe its (possibly degraded) view plus A's message, then choose one action."""
    return [
        ChatMessage(role="system", content=_SYSTEM_B),
        ChatMessage(
            role="user",
            content=(
                f"Your observation:\n{observation}\n\nMessage from A:\n{message}\n\n{_ACTION_HINT}"
            ),
        ),
    ]
