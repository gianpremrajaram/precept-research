"""Append-safe handoff persistence: hash-stamped Parquet parts plus a dataset index.

Each ``write_handoffs`` call appends a new ``part-*.parquet`` under ``data/<dataset_hash>/`` rather
than rewriting, so concurrent episode writes never clobber each other. The nested physics/action
dicts are stored as JSON strings (Parquet schemas would otherwise churn as physics keys evolve);
``load_dataset`` returns those columns decoded as a frame, and ``load_records`` reconstructs exact
``HandoffRecord`` objects. ``register_dataset`` maps each dataset hash to the config + manifest that
produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preceptx.data.schema import SCHEMA_VERSION, HandoffRecord

logger = logging.getLogger(__name__)

# Nested dict fields serialised as JSON strings in Parquet (see module docstring).
_NESTED_FIELDS = ("state", "action", "pre_state", "post_state")
_INDEX_NAME = "index.json"

# Explicit Arrow schema so every part file shares one schema - otherwise per-part type inference
# (e.g. an all-None Y column inferred as null) would break reading the parts back as one dataset.
_ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("episode_id", pa.string()),
        ("step", pa.int64()),
        ("condition", pa.string()),
        ("serialisation", pa.string()),
        ("difficulty", pa.string()),
        ("model", pa.string()),
        ("seed", pa.int64()),
        ("state", pa.string()),
        ("state_str", pa.string()),
        ("message_raw", pa.string()),
        ("message_delivered", pa.string()),
        ("action", pa.string()),
        ("pre_state", pa.string()),
        ("post_state", pa.string()),
        ("progress", pa.float64()),
        ("success", pa.bool_()),
        ("collision", pa.bool_()),
        ("stuck", pa.bool_()),
        ("y_binary_progress", pa.bool_()),
        ("y_continuous_displacement", pa.float64()),
        ("y_discrete_config", pa.int64()),
        ("y_terminal_success", pa.bool_()),
    ]
)


class DatasetError(RuntimeError):
    """A dataset read or write failed, or a dataset hash was not found in the index."""


def dataset_hash(config_hash: str, *, schema_version: int = SCHEMA_VERSION) -> str:
    """Derive a stable dataset hash from the run's config hash and the schema version."""
    payload = f"{config_hash}:v{schema_version}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _record_to_row(record: HandoffRecord) -> dict[str, Any]:
    row = record.model_dump()
    for field in _NESTED_FIELDS:
        row[field] = json.dumps(row[field], sort_keys=True)
    return row


def write_handoffs(
    records: Sequence[HandoffRecord], *, root: Path | str, dataset_hash: str
) -> Path:
    """Append ``records`` as a new Parquet part under ``root/<dataset_hash>/``; return its path."""
    if not records:
        raise DatasetError("write_handoffs called with no records")
    dataset_dir = Path(root) / dataset_hash
    dataset_dir.mkdir(parents=True, exist_ok=True)
    part_index = len(list(dataset_dir.glob("part-*.parquet")))
    part_path = dataset_dir / f"part-{part_index:05d}.parquet"
    # Write to a hidden temp, then atomically rename in. A crash mid-write leaves a ".part-*.tmp"
    # that both the part-*.parquet glob (wrong prefix and suffix) and pyarrow's directory read
    # (ignores "."-prefixed files) skip, so a truncated part can never poison a resume or read.
    # The temp name is keyed on the part index, so a resume overwrites any stale leftover.
    tmp_path = dataset_dir / f".part-{part_index:05d}.parquet.tmp"
    table = pa.Table.from_pylist([_record_to_row(r) for r in records], schema=_ARROW_SCHEMA)
    pq.write_table(table, tmp_path)
    tmp_path.replace(part_path)  # os.replace: atomic within one directory
    logger.info("wrote %d handoffs to %s", len(records), part_path)
    return part_path


def register_dataset(
    *, root: Path | str, dataset_hash: str, config_hash: str, manifest_path: Path | str
) -> None:
    """Record ``dataset_hash -> {config_hash, manifest_path, ...}`` in ``root/index.json``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / _INDEX_NAME
    index: dict[str, Any] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    index[dataset_hash] = {
        "config_hash": config_hash,
        "manifest_path": str(manifest_path),
        "schema_version": SCHEMA_VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(),
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))


def _read_rows(dataset_hash: str, root: Path | str) -> list[dict[str, Any]]:
    dataset_dir = Path(root) / dataset_hash
    if not list(dataset_dir.glob("part-*.parquet")):
        raise DatasetError(f"no parquet parts found for dataset {dataset_hash!r} under {root}")
    # to_pylist gives pure-Python scalars with None preserved (pandas would coerce None -> NaN).
    rows: list[dict[str, Any]] = pq.read_table(dataset_dir).to_pylist()
    for row in rows:
        for field in _NESTED_FIELDS:
            row[field] = json.loads(row[field])
    return rows


def load_dataset(dataset_hash: str, *, root: Path | str) -> pd.DataFrame:
    """Load all parts for ``dataset_hash`` into a typed frame; nested columns decoded to dicts."""
    return pd.DataFrame(_read_rows(dataset_hash, root))


def load_records(dataset_hash: str, *, root: Path | str) -> list[HandoffRecord]:
    """Load all parts back into exact ``HandoffRecord`` objects (round-trip identity)."""
    return [HandoffRecord.model_validate(row) for row in _read_rows(dataset_hash, root)]
