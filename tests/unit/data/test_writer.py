from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from preceptx.data.schema import HandoffRecord
from preceptx.data.writer import (
    DatasetError,
    dataset_hash,
    load_dataset,
    load_records,
    register_dataset,
    write_handoffs,
)

# JSON-round-trippable nested payloads: string keys, finite scalar values.
_scalars = st.one_of(
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=8),
)
_payloads = st.dictionaries(st.text(min_size=1, max_size=6), _scalars, max_size=4)

_records = st.builds(
    HandoffRecord,
    episode_id=st.text(min_size=1, max_size=8),
    step=st.integers(min_value=0, max_value=100),
    condition=st.sampled_from(["C0", "C1", "C2", "C3", "C4"]),
    serialisation=st.sampled_from(["numeric", "grid", "nl"]),
    difficulty=st.sampled_from(["easy", "medium", "hard"]),
    model=st.text(min_size=1, max_size=12),
    seed=st.integers(min_value=0, max_value=1000),
    state=_payloads,
    state_str=st.text(max_size=16),
    message_raw=st.text(max_size=16),
    message_delivered=st.text(max_size=16),
    action=_payloads,
    pre_state=_payloads,
    post_state=_payloads,
    progress=st.floats(allow_nan=False, allow_infinity=False, width=32),
    success=st.booleans(),
    collision=st.booleans(),
    stuck=st.booleans(),
    y_binary_progress=st.none() | st.booleans(),
    y_continuous_displacement=st.none()
    | st.floats(allow_nan=False, allow_infinity=False, width=32),
    y_discrete_config=st.none() | st.integers(min_value=0, max_value=9),
    y_terminal_success=st.none() | st.booleans(),
)


def _record() -> HandoffRecord:
    return HandoffRecord(
        episode_id="ep",
        step=0,
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model="m",
        seed=0,
        state={"x": 1.0},
        state_str="x",
        message_raw="r",
        message_delivered="d",
        action={"dx": 1.0},
        pre_state={"x": 1.0},
        post_state={"x": 2.0},
        progress=0.5,
        success=True,
        collision=False,
        stuck=False,
    )


def test_write_then_load_dataset_typed_columns(tmp_path: Path) -> None:
    write_handoffs([_record()], root=tmp_path, dataset_hash="abc")
    frame = load_dataset("abc", root=tmp_path)
    assert frame["progress"].dtype == "float64"
    assert frame["success"].dtype == "bool"
    assert frame["step"].dtype == "int64"
    assert frame.loc[0, "state"] == {"x": 1.0}  # nested decoded back to a dict


def test_append_creates_new_part_without_clobbering(tmp_path: Path) -> None:
    write_handoffs([_record()], root=tmp_path, dataset_hash="abc")
    write_handoffs([_record()], root=tmp_path, dataset_hash="abc")
    parts = sorted((tmp_path / "abc").glob("part-*.parquet"))
    assert [p.name for p in parts] == ["part-00000.parquet", "part-00001.parquet"]
    assert len(load_records("abc", root=tmp_path)) == 2


def test_truncated_temp_part_does_not_poison_reads_or_part_index(tmp_path: Path) -> None:
    write_handoffs([_record()], root=tmp_path, dataset_hash="abc")
    # Simulate a crash mid-write: a leftover, truncated temp part (garbage, not valid parquet).
    (tmp_path / "abc" / ".part-00001.parquet.tmp").write_bytes(b"truncated junk")
    assert len(load_records("abc", root=tmp_path)) == 1  # read ignores the temp, no parse error
    write_handoffs([_record()], root=tmp_path, dataset_hash="abc")  # resume
    parts = sorted((tmp_path / "abc").glob("part-*.parquet"))
    assert [p.name for p in parts] == ["part-00000.parquet", "part-00001.parquet"]  # temp uncounted
    assert len(load_records("abc", root=tmp_path)) == 2


def test_empty_write_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        write_handoffs([], root=tmp_path, dataset_hash="abc")


def test_load_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_dataset("nope", root=tmp_path)


def test_register_dataset_writes_index(tmp_path: Path) -> None:
    register_dataset(
        root=tmp_path,
        dataset_hash="abc",
        config_hash="cfg123",
        manifest_path="runs/x/manifest.json",
    )
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["abc"]["config_hash"] == "cfg123"
    assert index["abc"]["manifest_path"] == "runs/x/manifest.json"


def test_dataset_hash_is_stable_and_short() -> None:
    h = dataset_hash("cfg123")
    assert h == dataset_hash("cfg123")
    assert len(h) == 16
    assert dataset_hash("cfg123") != dataset_hash("cfg456")


@settings(max_examples=50)
@given(records=st.lists(_records, min_size=1, max_size=5))
def test_records_round_trip_identically(
    tmp_path_factory: pytest.TempPathFactory, records: list[HandoffRecord]
) -> None:
    root = tmp_path_factory.mktemp("ds")
    write_handoffs(records, root=root, dataset_hash="rt")
    assert load_records("rt", root=root) == records
