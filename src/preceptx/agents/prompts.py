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
from preceptx.sim.actions import ROTATION_STEP_DEG

# v2: serialisation bump (P1-5 + RD-7) - the grid gained its constant legend/axis header and the
# numeric form dropped the dead vel line. Prompt templates themselves are unchanged from v1; the
# serialised state is part of the prompt surface, so the recorded version moves with it.
# v3 (E1, first live read): the numeric form gained the wall/slit geometry it had never carried, and
# both system prompts were rewritten. A was emitting near-constant boilerplate (7 distinct messages
# across 75 handoffs) because it had no obstacle to describe, and B chose the same action 75/75
# because nothing told it to act on the message. Both are prerequisites for any information gradient
# to exist at all. One of the three budgeted pre-E3 bumps.
# v4 (E3-local): the numeric form gained `load_size` and A's system prompt now states that the WHOLE
# load must fit the slit, not its centre. The state named the gap's extent and never the object's,
# so "aligned with the slit" was underdetermined: the pilot watched A call com_y=2.0074 aligned with
# a (2.1, 3.9) gap and push into the wall for the rest of the budget. The dimensions are constants
# of the load, not a derived pass band. Bump two of the three budgeted pre-E3 bumps.
# v5 (E3 attempt 1, the Myriad bf16 re-gate - THE ONE PERMITTED RETUNE): the state gained a
# `recent=` line naming the last four actions and what each gained, and A's system prompt names it.
# Attempt 1 failed G1 with 24 of its 34 failed episodes in a limit cycle: N,S,N,S...; ROT+,ROT-,...;
# or E into a wall for the whole budget. Nothing in the v4 prompt surface carried an action history,
# so at temperature 0 a state that maps to an action which returns the state to itself is a fixed
# point of the policy and the pair cannot escape by construction. The fix is observability, not
# instruction: the line reports what happened, and the inference stays the agent's. Bump three of
# the three budgeted pre-E3 bumps.
# v6 (DSE-058, the successor task): the load is a convex 1.4 x 0.3 bar, so every prompt-surface
# sentence calling it a T is now simply false, and each internal wall is a CHANNEL of depth
# `wall_depth` rather than a threshold, which the state forms never named. Both system prompts and
# the NL serialiser described a T; the numeric and NL forms gave one x per wall; the grid drew a
# one-cell stripe where 1.5 world-units of solid geometry stand. This does NOT consume a fourth
# retune of the T task - it is the prompt surface of a DIFFERENT benchmark, and describing the T
# arena's object and obstacle to agents manipulating a bar through a channel would reintroduce the
# exact grounded-but-inferentially-wrong defect DSE-057 was spent falsifying. Landed before the
# first successor model call, so no dataset is re-keyed.
PROMPT_VERSION = "v7"

_SYSTEM_A = (
    "You are agent A, the navigator in a two-agent cooperative-transport task. A straight "
    "bar-shaped load must be pushed rightward (+x) through a slit in each wall to reach the goal "
    "region. +y is north, -y is south. Each wall is a channel with depth along x, not a thin line, "
    "so the load must be aligned before it enters and stay aligned all the way through. The load "
    "only passes a wall when the WHOLE load fits inside that wall's slit - its centre being inside "
    "the slit's y-range is not enough, because the load has size of its own (see load_size) and "
    "juts out either side of its centre. You can see the whole "
    "scene; B acts but sees less than you. Send B one or two sentences saying where the load is "
    "relative to the next slit and what to do now - which way to push, or whether to rotate first "
    "so the load fits through. Use the actual numbers in front of you; do not give generic advice. "
    "The `recent` line lists the last few actions and how much ground each one gained toward the "
    "goal, oldest first; read it before deciding, because an action that gained nothing last time "
    "will gain nothing again from the same position."
)

# Realised world-units per N/S/E/W push, for the action hint. Deterministic (sd 1e-15), so it is a
# constant rather than an average - and asserted against the live actuator in test_prompts.py, since
# a number in a prompt that describes the physics must not be able to disagree with the physics.
_LINEAR_STEP = 1.03


_SYSTEM_B = (
    "You are agent B, the actuator in a two-agent cooperative-transport task. You receive a "
    "partial observation of the scene and one instruction from agent A, who can see more of the "
    "scene than you. Choose exactly one macro-action that best advances the bar-shaped load "
    "toward the goal, following A's instruction unless your own observation plainly contradicts it."
)

# The action QUANTA are stated, not left to be inferred from action history (DSE-059, D26). Since
# the corrected actuator, hard tolerates no miscount - it needs exactly seven rotations - so leaving
# the step size implicit would make the rung a constant-DISCOVERY task and the capability arm would
# measure the wrong thing. Declaring it is also the standard in embodied spatial-reasoning
# benchmarks: REM supplies "rotate right 15 deg" alongside the observation, and still finds that
# models collapse under full rotation, which is the capability this task is meant to stress.
# The numbers are interpolated from the live StepConfig so the prompt cannot drift from the physics.
_ACTION_HINT = (
    "Actions: N pushes the load north (+y), S south (-y), E east (+x), W west (-x), each by about "
    f"{_LINEAR_STEP:.2f} units; ROT+/ROT- rotate it by exactly {ROTATION_STEP_DEG:.0f} degrees "
    "(anticlockwise/clockwise); WAIT does nothing. Choose one."
)


# The gate's retry feedback (DSE-045). Versioned separately from PROMPT_VERSION because it is part
# of the RQ3b *treatment*, not of the base task: it only ever reaches a model on a blocked retry, so
# a wording change re-shapes the causal arm while leaving every ungated dataset untouched.
#
# Why a feedback template at all: under greedy decoding a re-prompt is a fixed point. The same
# prompt yields the same message, the same statistic and the same block, for every bounded retry, so
# a gate that merely re-asks is vacuous - it would pass its unit tests and change nothing live.
# The retry prompt must differ in content, which is what this template supplies.
#
# Rejected alternative - raise the temperature on the retry. It escapes the fixed point, but it
# breaks the determinism story mid-episode (the run is greedy except at exactly the handoffs the
# gate touched) and it confounds the gate's effect with an increase in sampling entropy: a
# post-retry improvement could not be attributed to the feedback rather than to having sampled
# twice. The four-arm contrast in H6 needs the arms to differ in one thing, so retries stay greedy
# and the *content* is what changes.
GATE_FEEDBACK_VERSION = "v1"

GATE_FEEDBACK = (
    "Your previous instruction was blocked: it did not carry enough information for B to act on. "
    "Write a different one. Using the actual numbers in front of you, state explicitly: which way "
    "to push now (north, south, east or west); whether the load must rotate first to fit through "
    "the slit and if so which way; and which direction the goal lies in. Do not reuse your "
    "previous wording."
)


def prompt_a(state_str: str, *, gate_feedback: bool = False) -> list[ChatMessage]:
    """A's chat: observe the full serialised state, emit a natural-language handoff to B.

    ``gate_feedback`` appends ``GATE_FEEDBACK`` for a gate-blocked retry (DSE-045); the default
    False path is byte-identical to the ungated prompt, so no existing dataset shifts.
    """
    feedback = f"\n\n{GATE_FEEDBACK}" if gate_feedback else ""
    return [
        ChatMessage(role="system", content=_SYSTEM_A),
        ChatMessage(
            role="user",
            content=f"Current scene:\n{state_str}{feedback}\n\nYour instruction to B:",
        ),
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
