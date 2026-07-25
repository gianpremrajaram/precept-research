# Handoff dataset schema

The `HandoffRecord` (`src/preceptx/data/schema.py`) is the **stable contract** the measurement, gate,
and experiment tickets all consume. Downstream code imports it; it never redefines these fields. The
schema is versioned via `SCHEMA_VERSION` — a breaking change bumps it and `load_dataset` keys off it.

## Fields

| Field | Type | Filled by | Meaning |
|---|---|---|---|
| `schema_version` | `int` | writer | Schema version of this record. |
| `episode_id` | `str` | runner | Stable id for the episode this handoff belongs to. |
| `step` | `int ≥ 0` | runner | Step index within the episode. |
| `condition` | `C0`–`C4` | config | Communication condition (channel degradation). |
| `serialisation` | `numeric`/`grid`/`nl` | config | How state is written into the prompt. |
| `difficulty` | `easy`/`medium`/`hard` | config | Task difficulty (slit width). |
| `model` | `str` | config | Served model identity. |
| `seed` | `int ≥ 0` | config | Run seed. |
| `state` | `dict` | sim | Structured physics state at the handoff. |
| `state_str` | `str` | serialiser | Serialised (prompt) form of `state` — A's full view. |
| `observation` | `str` | channel | **v2.** B's delivered view at the handoff — the CPVI conditioning state. Equals `state_str` except under C3 (the restricted window, by design). |
| `message_raw` | `str` | agent A | A→B message before the channel. |
| `message_delivered` | `str` | channel | Message after the channel degrades it. |
| `gate_blocked` | `bool` | gate (DSE-018) | **v2.** Whether the runtime gate blocked the message. |
| `gate_retries` | `int ≥ 0` | gate (DSE-018) | **v2.** Regeneration attempts before delivery. |
| `message_blocked` | `str \| None` | gate (DSE-018) | **v2.** The blocked message, when the gate fired. |
| `action` | `dict` | agent B | Structured action B took. |
| `pre_state` | `dict` | sim | Physics state before the action. |
| `post_state` | `dict` | sim | Physics state after the action. |
| `progress` | `float` | sim | Signed progress over this step. |
| `success` | `bool` | sim | Whether the goal is reached at this step. |
| `collision` | `bool` | sim | Whether a collision occurred this step. |
| `stuck` | `bool` | sim | Whether the load is stuck this step. |
| `y_binary_progress` | `bool \| None` | DSE-009 | Net progress over the next *k* steps. |
| `y_continuous_displacement` | `float \| None` | DSE-009 | Continuous displacement label. |
| `y_discrete_config` | `int \| None` | DSE-009 | Bucketed pose-region label (chamber at the window end). |
| `y_terminal_success` | `bool \| None` | DSE-009 | Whether the episode ultimately succeeded from here. |
| `y_window_truncated` | `bool \| None` | DSE-009 | **v2.** The forward *k*-window was clamped at episode end. |

Both `state` (structured) and `state_str` (serialised) are kept so the serialisation A/B (DSE-008) is
recoverable from the dataset alone. Keeping `state_str` *alongside* `observation` gives C3 a free
dual-baseline diagnostic (condition on what B saw vs what A saw). The `Y` labels are `None` until the
post-episode labeller (DSE-009) fills them.

**v2 (2026-07):** adds `observation` (P0-1 — CPVI must condition on the receiver-observed state), the
three DSE-018 gate fields (bundled so the schema changes once), and `y_window_truncated` (P1-12).
No v1 loader is kept — no real dataset was ever written at v1.

## Storage

`data.writer` persists records as **append-safe, hash-stamped Parquet**:

- `write_handoffs(records, root=..., dataset_hash=...)` appends a new `part-NNNNN.parquet` under
  `data/<dataset_hash>/` — appending never rewrites existing parts.
- The nested dict fields (`state`, `action`, `pre_state`, `post_state`) are stored as JSON strings;
  `load_dataset` returns them decoded, and `frame_to_records` reconstructs exact `HandoffRecord`s.
- `dataset_hash(config_hash)` derives the dataset hash from the run's config hash + schema version.
- `register_dataset(...)` maps `dataset_hash → {config_hash, manifest_path, schema_version, created}`
  in `data/index.json`.

Prompts are **not** stored (size); only the message and state are. Prompt templates are versioned in
the repo, not the dataset.

## Telemetry

`data.otel_capture.emit_handoff(record)` emits each record as a vanilla OpenTelemetry `handoff` span.
It is **fail-open**: with no exporter configured it no-ops via the default no-op tracer and never
raises. This is in-repo capture — **precept is not a dependency**.
