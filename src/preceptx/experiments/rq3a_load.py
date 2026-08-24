"""Loaders that normalise real multi-agent logs into the RQ3a record schema (DSE-041).

Three corpora, one interface. Each loader takes a **local path** and returns validated records;
nothing here touches the network, so the unit tests run offline and a cluster job cannot stall on a
download. ``scripts/fetch_rq3a.sh`` does the fetching, once, out of band.

Substrate roles, as measured by the E9 spike rather than assumed:

* **TraceElephant** - primary. Per-trace ``step_records.json`` carries ``input`` (the receiver-
  observed message list) and ``output`` (the acting component's completion), which is exactly the
  conditional construct CPVI needs. ``trace_metadata.json`` carries ``tests_status``, an
  annotation-free trace outcome.
* **Who&When** - secondary, and observability-caveated. It records a flat message ``history`` with
  no per-step input context, so the observation is rebuilt from the preceding messages and every
  row it emits carries ``reconstructed_observation=True``.
* **MAST-Data** - trace-level secondary only. Its ``trajectory`` is one unsegmented string per
  trace, formatted differently by each of the seven systems, so it yields ``LogTraceRecord`` and
  never a per-step record.

See ``docs/rq3a_schema_mapping.md`` for the field-by-field mapping and the counts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from preceptx.data.logs import Corpus, LogHandoffRecord, LogTraceRecord

logger = logging.getLogger(__name__)


class CorpusError(RuntimeError):
    """A corpus file is missing or does not have the shape the loader was written against.

    Raised rather than skipped: a silently dropped trace changes every count downstream, and the
    counts are the artefact this ticket exists to produce.
    """


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------


def mark_handoffs(agents: Sequence[str]) -> list[tuple[str | None, bool]]:
    """Per step, the next acting component and whether the step ends an inter-agent handoff.

    A step is a handoff when the component acting at ``i + 1`` differs from the one acting at
    ``i``; otherwise it is an intra-agent tool turn. The final step has no successor, so its
    receiver is ``None`` and it is not a handoff.
    """
    out: list[tuple[str | None, bool]] = []
    for i, a in enumerate(agents):
        nxt = agents[i + 1] if i + 1 < len(agents) else None
        out.append((nxt, nxt is not None and nxt != a))
    return out


def _text(content: Any) -> str:
    """Flatten an OpenAI message ``content`` field, which is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                parts.append(t if isinstance(t, str) else json.dumps(p, sort_keys=True))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return "" if content is None else str(content)


def render_messages(messages: Sequence[Any]) -> str:
    """Render a chat message list as the role-prefixed transcript the component actually saw.

    The whole prefix is the observation, not just the last turn: the receiver's usable state at a
    step is everything in its context window, and truncating to the last message would understate
    the state-only baseline and inflate CPVI.
    """
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            raise CorpusError(f"message is {type(m).__name__}, expected a dict")
        role = str(m.get("name") or m.get("role") or "unknown")
        lines.append(f"{role}: {_text(m.get('content'))}")
    return "\n\n".join(lines)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise CorpusError(f"expected corpus file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# TraceElephant - the primary substrate
# --------------------------------------------------------------------------------------------


def _completion_text(output: Any) -> str:
    """The acting component's emitted message: assistant content plus any tool calls.

    Tool calls are part of the message, not metadata: on a tool-using system the call *is* what the
    next component receives, and a step whose content is ``""`` with a populated ``tool_calls`` is
    the common case, not an error.
    """
    if not isinstance(output, dict):
        raise CorpusError(f"step output is {type(output).__name__}, expected a dict")
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CorpusError("step output has no choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise CorpusError("step output choice has no message")
    parts = [_text(msg.get("content"))]
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        parts.append(f"[tool_call {fn.get('name')}] {fn.get('arguments')}")
    return "\n".join(p for p in parts if p)


def _tests_failed(tests_status: Any) -> bool | None:
    """Annotation-free trace outcome from SWE-bench test results.

    A trace failed when any test in either bucket is in its ``failure`` list. This reads only the
    harness's own test outcome and never ``mistake_agent``/``mistake_step``/``mistake_reason``,
    which is what keeps the outcome usable as *Y* rather than circular with the labels.
    """
    if not isinstance(tests_status, dict) or not tests_status:
        return None
    for bucket in tests_status.values():
        if isinstance(bucket, dict) and bucket.get("failure"):
            return True
    return False


def load_traceelephant(root: Path) -> list[LogHandoffRecord]:
    """Load every trace under ``root`` (the unzipped ``data/`` directory).

    Layout is ``<root>/<run family>/<task id>/{step_records,trace_metadata}.json``; the trace id is
    ``family/task`` so two families sharing a task id stay distinct groups at cross-fit time.
    """
    if not root.is_dir():
        raise CorpusError(f"expected the unzipped TraceElephant data directory at {root}")
    out: list[LogHandoffRecord] = []
    for steps_path in sorted(root.glob("*/*/step_records.json")):
        trace_dir = steps_path.parent
        trace_id = f"{trace_dir.parent.name}/{trace_dir.name}"
        meta = _load_json(trace_dir / "trace_metadata.json")
        if not isinstance(meta, dict):
            raise CorpusError(f"{trace_id}: trace_metadata.json is not an object")
        steps = _load_json(steps_path)
        if not isinstance(steps, list):
            raise CorpusError(f"{trace_id}: step_records.json is not a list")
        failed = _tests_failed(meta.get("tests_status"))
        annotations = {k: meta.get(k) for k in ("mistake_agent", "mistake_step", "mistake_reason")}
        agents = [str(s.get("agent_name", "")) for s in steps]
        for i, (s, (receiver, is_handoff)) in enumerate(
            zip(steps, mark_handoffs(agents), strict=True)
        ):
            inp = s.get("input")
            if not isinstance(inp, dict) or not isinstance(inp.get("messages"), list):
                raise CorpusError(f"{trace_id} step {i}: input.messages missing")
            agent_id = s.get("agent_id")
            out.append(
                LogHandoffRecord(
                    corpus="traceelephant",
                    trace_id=trace_id,
                    step=int(s.get("step_id", i)),
                    agent_name=agents[i],
                    agent_id=None if agent_id is None else str(agent_id),
                    receiver=receiver,
                    is_handoff=is_handoff,
                    observation=render_messages(inp["messages"]),
                    message=_completion_text(s.get("output")),
                    trace_failed=failed,
                    annotations=annotations,
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# Who&When - reconstructed observations only
# --------------------------------------------------------------------------------------------

_WHOWHEN_FILES = ("Algorithm-Generated.parquet", "Hand-Crafted.parquet")


def load_who_and_when(root: Path) -> list[LogHandoffRecord]:
    """Load the Who&When parquets, flagging every row as a reconstructed observation.

    The corpus stores a flat ``history`` of chat messages and no per-step input context, so the
    observation at step *i* is rebuilt as the render of messages ``0..i-1`` - an approximation of
    what the component saw, never the recorded thing. The flag makes that unmissable downstream.
    """
    out: list[LogHandoffRecord] = []
    for name in _WHOWHEN_FILES:
        path = root / name
        if not path.is_file():
            raise CorpusError(f"expected corpus file at {path}")
        df = pd.read_parquet(path)
        # The two splits disagree on the spelling of the correctness column.
        correct_col = "is_correct" if "is_correct" in df.columns else "is_corrected"
        for row_idx, row in df.iterrows():
            trace_id = f"{Path(name).stem}/{row.get('question_ID', row_idx)}"
            history = list(row["history"])
            # Hand-Crafted carries no ``name``; its ``role`` doubles as the component identity.
            agents = [str(m.get("name") or m.get("role") or "unknown") for m in history]
            annotations = {
                k: row[k] for k in ("mistake_agent", "mistake_step", "mistake_reason") if k in df
            }
            for i, (m, (receiver, is_handoff)) in enumerate(
                zip(history, mark_handoffs(agents), strict=True)
            ):
                out.append(
                    LogHandoffRecord(
                        corpus="who_and_when",
                        trace_id=trace_id,
                        step=i,
                        agent_name=agents[i],
                        receiver=receiver,
                        is_handoff=is_handoff,
                        observation=render_messages(history[:i]),
                        message=_text(m.get("content")),
                        reconstructed_observation=True,
                        trace_failed=not bool(row[correct_col]),
                        annotations=annotations,
                    )
                )
    return out


# --------------------------------------------------------------------------------------------
# MAST-Data - the trace-level secondary
# --------------------------------------------------------------------------------------------


def load_mast(path: Path) -> list[LogTraceRecord]:
    """Load MAD_full_dataset.json as trace-level rows.

    A trace counts as a failure when any of the 14 MAST modes is flagged; an all-zero annotation
    row is the non-failure class the refit arm needs. That proportion is **counted here**, not
    assumed from a dataset preview.
    """
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise CorpusError(f"{path} is not a list of traces")
    out: list[LogTraceRecord] = []
    for r in rows:
        ann = r.get("mast_annotation")
        trace = r.get("trace")
        if not isinstance(trace, dict) or "trajectory" not in trace:
            raise CorpusError(f"trace {r.get('trace_id')}: no trace.trajectory")
        out.append(
            LogTraceRecord(
                corpus="mast",
                trace_id=f"{r.get('mas_name')}/{r.get('benchmark_name')}/{r.get('trace_id')}",
                system_name=str(r.get("mas_name")),
                model_name=r.get("llm_name"),
                benchmark=r.get("benchmark_name"),
                trace_text=str(trace["trajectory"]),
                trace_failed=None if not isinstance(ann, dict) else any(ann.values()),
                annotations=ann if isinstance(ann, dict) else {},
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# Counts - the E9 artefact
# --------------------------------------------------------------------------------------------


class CorpusCounts(BaseModel):
    """What a corpus actually contains. Reported per corpus, never pooled."""

    model_config = ConfigDict(extra="forbid")

    corpus: Corpus
    traces: int
    steps: int
    handoffs: int
    failures: int
    non_failures: int
    outcome_unknown: int
    reconstructed_observations: bool


def count_handoff_corpus(records: Sequence[LogHandoffRecord]) -> CorpusCounts:
    """Counts over per-step records. Failure counts are per *trace*, not per step."""
    if not records:
        raise CorpusError("no records to count")
    outcomes = {r.trace_id: r.trace_failed for r in records}
    return CorpusCounts(
        corpus=records[0].corpus,
        traces=len(outcomes),
        steps=len(records),
        handoffs=sum(r.is_handoff for r in records),
        failures=sum(v is True for v in outcomes.values()),
        non_failures=sum(v is False for v in outcomes.values()),
        outcome_unknown=sum(v is None for v in outcomes.values()),
        reconstructed_observations=any(r.reconstructed_observation for r in records),
    )


def count_trace_corpus(records: Sequence[LogTraceRecord]) -> CorpusCounts:
    """Counts over trace-level records. ``steps`` and ``handoffs`` are 0: none are recorded."""
    if not records:
        raise CorpusError("no records to count")
    return CorpusCounts(
        corpus=records[0].corpus,
        traces=len(records),
        steps=0,
        handoffs=0,
        failures=sum(r.trace_failed is True for r in records),
        non_failures=sum(r.trace_failed is False for r in records),
        outcome_unknown=sum(r.trace_failed is None for r in records),
        reconstructed_observations=False,
    )


def _main() -> None:
    """Print the E9 counts table from a local corpus root (dev tool, not an experiment)."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="RQ3a corpus counts (E9)")
    ap.add_argument(
        "--root", type=Path, required=True, help="directory holding the fetched corpora"
    )
    args = ap.parse_args()

    counts: list[CorpusCounts] = []
    counts.append(count_handoff_corpus(load_traceelephant(args.root / "traceelephant" / "data")))
    counts.append(count_handoff_corpus(load_who_and_when(args.root / "who_and_when")))
    counts.append(count_trace_corpus(load_mast(args.root / "mast" / "MAD_full_dataset.json")))

    logger.info(
        "%-14s %8s %8s %9s %9s %13s %9s %s",
        "corpus",
        "traces",
        "steps",
        "handoffs",
        "failures",
        "non-failures",
        "unknown",
        "recon",
    )
    for c in counts:
        logger.info(
            "%-14s %8d %8d %9d %9d %13d %9d %s",
            c.corpus,
            c.traces,
            c.steps,
            c.handoffs,
            c.failures,
            c.non_failures,
            c.outcome_unknown,
            c.reconstructed_observations,
        )


if __name__ == "__main__":
    _main()
