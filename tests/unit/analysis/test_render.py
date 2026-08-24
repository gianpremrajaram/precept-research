"""Episode renderer + transcript tests (review section 7-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from preceptx.analysis.render import render_episode, render_transcript
from preceptx.data.schema import HandoffRecord


def _record(
    step: int,
    action: str,
    *,
    com_x: float = 2.0,
    success: bool = False,
    msg_raw: str = "push east",
    msg_delivered: str = "push east",
) -> HandoffRecord:
    payload = {"com_x": com_x, "com_y": 3.0, "angle": 0.3, "vx": 0.0, "vy": 0.0, "omega": 0.0}
    return HandoffRecord(
        episode_id="ep-demo",
        step=step,
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model="m",
        seed=0,
        state=payload,
        state_str="s",
        observation="s",
        message_raw=msg_raw,
        message_delivered=msg_delivered,
        action={"action": action},
        pre_state=payload,
        post_state=payload,
        progress=0.5,
        success=success,
        collision=False,
        stuck=False,
    )


def test_transcript_has_header_and_a_row_per_step() -> None:
    records = [_record(0, "E"), _record(1, "ROT+", success=True)]
    text = render_transcript(records)
    assert "# Episode `ep-demo`" in text
    assert "terminal success: **True**" in text
    assert "| 0 | E |" in text
    assert "| 1 | ROT+ |" in text
    assert "goal" in text  # the success flag on the last row


def test_transcript_shows_channel_degradation_and_escapes_pipes() -> None:
    # Raw != delivered renders as "raw -> delivered"; a literal pipe is escaped so the table holds.
    text = render_transcript([_record(0, "E", msg_raw="go | east", msg_delivered="go")])
    assert "go \\| east -> go" in text


def test_transcript_empty_fails_loud() -> None:
    with pytest.raises(ValueError):
        render_transcript([])


def test_render_episode_noops_without_viz_or_writes_a_png(tmp_path: Path) -> None:
    # Robust to both CI (no viz extra -> None) and a viz-installed env (-> a written PNG).
    out = render_episode([_record(0, "E"), _record(1, "E", com_x=3.0)], path=tmp_path / "ep.png")
    assert out is None or out.exists()


def test_render_episode_empty_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_episode([], path=tmp_path / "ep.png")


def test_transcript_gives_each_episode_its_own_section() -> None:
    # One table under one header would attribute every step and the whole handoff count to the
    # first episode - the exact thing the E1 transcript read must not be misled by.
    a = [_record(0, "E"), _record(1, "N", success=True)]
    b = [_record(0, "W")]
    b[0] = b[0].model_copy(update={"episode_id": "ep-other"})
    text = render_transcript([*a, *b])
    assert text.count("# Episode ") == 2
    assert "# Episode `ep-demo`" in text and "# Episode `ep-other`" in text
    assert text.count("- 2 handoffs") == 1 and text.count("- 1 handoffs") == 1
    assert text.count("terminal success: **True**") == 1  # only ep-demo reached the goal
