"""Prompt-surface tests, and the gate retry-feedback template (DSE-045)."""

from __future__ import annotations

import pytest

from preceptx.agents.prompts import (
    _ACTION_HINT,
    _LINEAR_STEP,
    GATE_FEEDBACK,
    GATE_FEEDBACK_VERSION,
    prompt_a,
    prompt_b,
)
from preceptx.sim.actions import ROTATION_STEP_DEG, StepConfig, apply_macro_action
from preceptx.sim.arena import make_scenario


def test_ungated_prompt_is_unchanged_by_the_feedback_parameter() -> None:
    """The default path must be byte-identical, or adding the gate would shift every dataset."""
    assert prompt_a("scene") == prompt_a("scene", gate_feedback=False)


def test_blocked_retry_issues_a_different_prompt() -> None:
    """The load-bearing DSE-045 claim: under greedy decoding an identical re-prompt is a fixed
    point, so the gate is vacuous unless the retry prompt actually differs."""
    original = prompt_a("scene")
    retry = prompt_a("scene", gate_feedback=True)
    assert original[1].content != retry[1].content
    assert GATE_FEEDBACK in str(retry[1].content)
    assert original[0] == retry[0]  # only the user turn changes; the system prompt is fixed


def test_retry_prompt_keeps_the_state_and_the_instruction_cue() -> None:
    retry = str(prompt_a("com_y=2.0", gate_feedback=True)[1].content)
    assert "com_y=2.0" in retry
    assert retry.endswith("Your instruction to B:")


def test_feedback_names_the_three_things_a_must_state() -> None:
    """The template is the treatment: it has to ask for the push direction, the rotation decision
    and the goal direction explicitly, not just say 'try again' (DSE-045 acceptance criteria)."""
    text = GATE_FEEDBACK.lower()
    assert "push" in text
    assert all(d in text for d in ("north", "south", "east", "west"))
    assert "rotate" in text and "which way" in text
    assert "goal" in text


def test_feedback_version_is_pinned() -> None:
    assert GATE_FEEDBACK_VERSION == "v1"


def test_prompt_b_carries_observation_and_message() -> None:
    content = str(prompt_b("obs-here", "msg-here")[1].content)
    assert "obs-here" in content and "msg-here" in content


def test_the_action_hint_matches_the_actual_actuator() -> None:
    """A prompt that describes the physics must not be able to disagree with the physics.

    The DSE-059 failure in prompt form: the rotation quantum is now stated to the agents, and hard
    tolerates no miscount, so a stale number here would mislead every episode rather than merely
    document something wrong. The rotation figure is interpolated from ``ROTATION_STEP_DEG``; the
    linear figure is a constant, so it is the one that needs asserting.
    """
    scenario = make_scenario("easy")
    body = scenario.load
    before = body.position.x
    apply_macro_action(scenario.space, body, "E", StepConfig())
    assert body.position.x - before == pytest.approx(_LINEAR_STEP, abs=0.01)
    assert f"{ROTATION_STEP_DEG:.0f} degrees" in _ACTION_HINT
