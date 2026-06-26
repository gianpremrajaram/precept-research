from __future__ import annotations

import pytest
from pydantic import ValidationError

from preceptx.data.schema import SCHEMA_VERSION, HandoffRecord


def _minimal_record() -> HandoffRecord:
    return HandoffRecord(
        episode_id="ep-1",
        step=0,
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model="Qwen/Qwen3-14B-Instruct",
        seed=0,
        state={"x": 1.0},
        state_str="x=1.0",
        message_raw="push right",
        message_delivered="push right",
        action={"dx": 1.0},
        pre_state={"x": 1.0},
        post_state={"x": 1.1},
        progress=0.1,
        success=False,
        collision=False,
        stuck=False,
    )


def test_record_validates_and_defaults_y_to_none() -> None:
    record = _minimal_record()
    assert record.schema_version == SCHEMA_VERSION
    assert record.y_binary_progress is None
    assert record.y_continuous_displacement is None
    assert record.y_discrete_config is None
    assert record.y_terminal_success is None


def test_record_rejects_unknown_condition() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"condition": "C9"})


def test_record_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"surprise": 1})


def test_record_rejects_negative_step() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"step": -1})
