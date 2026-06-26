"""Handoff dataset: the schema contract, persistence, and OTel capture."""

from __future__ import annotations

from preceptx.data.otel_capture import emit_handoff
from preceptx.data.schema import SCHEMA_VERSION, HandoffRecord
from preceptx.data.writer import (
    DatasetError,
    dataset_hash,
    load_dataset,
    load_records,
    register_dataset,
    write_handoffs,
)

__all__ = [
    "SCHEMA_VERSION",
    "DatasetError",
    "HandoffRecord",
    "dataset_hash",
    "emit_handoff",
    "load_dataset",
    "load_records",
    "register_dataset",
    "write_handoffs",
]
