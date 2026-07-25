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
        model="Qwen/Qwen3-14B",
        seed=0,
        state={"x": 1.0},
        state_str="x=1.0",
        observation="x=1.0",
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
    assert record.y_window_truncated is None


def test_record_defaults_gate_fields_to_no_gate() -> None:
    record = _minimal_record()
    assert record.gate_blocked is False
    assert record.gate_retries == 0
    assert record.message_blocked is None


def test_record_requires_observation() -> None:
    payload = _minimal_record().model_dump()
    del payload["observation"]
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(payload)


def test_record_rejects_unknown_condition() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"condition": "C9"})


def test_record_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"surprise": 1})


def test_record_rejects_negative_step() -> None:
    with pytest.raises(ValidationError):
        HandoffRecord.model_validate(_minimal_record().model_dump() | {"step": -1})
