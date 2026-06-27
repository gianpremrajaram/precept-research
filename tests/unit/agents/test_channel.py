from __future__ import annotations

import numpy as np

from preceptx.agents.channel import ChannelConfig, ChannelResult, apply_channel
from preceptx.data.schema import Condition, Serialisation

CFG = ChannelConfig()


def _ch(
    message: str,
    condition: Condition,
    *,
    serialisation: Serialisation = "nl",
    observation: str = "o",
    cfg: ChannelConfig = CFG,
    seed: int = 0,
    buffered: str | None = None,
) -> ChannelResult:
    return apply_channel(
        message,
        condition,
        serialisation=serialisation,
        observation=observation,
        cfg=cfg,
        rng=np.random.default_rng(seed),
        buffered=buffered,
    )


def test_c0_passes_message_through_unchanged() -> None:
    r = _ch("push east now", "C0")
    assert r.message_delivered == "push east now"
    assert r.observation_override is None


def test_c1_caps_to_token_budget() -> None:
    r = _ch("one two three four", "C1", cfg=ChannelConfig(c1_max_tokens=2))
    assert r.message_delivered == "one two"


def test_c2_delays_delivery_by_exactly_one_step() -> None:
    first = _ch("msg-0", "C2", buffered=None)  # step 0: nothing buffered yet
    assert first.message_delivered == "(no message yet)"
    assert first.new_buffer == "msg-0"
    second = _ch("msg-1", "C2", buffered=first.new_buffer)  # step 1: receives step-0 message
    assert second.message_delivered == "msg-0"
    assert second.new_buffer == "msg-1"


def test_c3_windows_grid_observation_and_leaves_message_intact() -> None:
    grid = "\n".join(["....", "....", ".T..", "....", "...."])  # load on row index 2
    r = _ch(
        "full instruction",
        "C3",
        serialisation="grid",
        observation=grid,
        cfg=ChannelConfig(c3_window_rows=1),
    )
    assert r.message_delivered == "full instruction"  # C3 never touches the message
    assert r.observation_override == "\n".join(["....", ".T..", "...."])  # band of +/-1 row


def test_c3_numeric_hides_the_goal_line() -> None:
    obs = "load=(1, 2, 3)\nvel=(0, 0, 0)\ncontact=False\ngoal=(9, 9, 1)"
    r = _ch("m", "C3", serialisation="numeric", observation=obs)
    assert r.observation_override is not None
    assert "goal=" not in r.observation_override and "load=" in r.observation_override


def test_c3_nl_keeps_only_the_self_state_sentence() -> None:
    obs = "The load is at (1, 2). The goal is east beyond 1 slit. The load is clear of a wall."
    r = _ch("m", "C3", serialisation="nl", observation=obs)
    assert r.observation_override == "The load is at (1, 2)."  # goal / slit clauses dropped


def test_c4_dropout_is_seed_deterministic() -> None:
    cfg = ChannelConfig(c4_dropout=0.5)
    a = _ch("alpha beta gamma delta epsilon", "C4", cfg=cfg, seed=7)
    b = _ch("alpha beta gamma delta epsilon", "C4", cfg=cfg, seed=7)
    assert a.message_delivered == b.message_delivered  # same seed -> identical dropout
    assert len(a.message_delivered.split()) <= 5


def test_c5_supervisor_relay_is_disabled_by_default() -> None:
    assert ChannelConfig().c5_enabled is False
