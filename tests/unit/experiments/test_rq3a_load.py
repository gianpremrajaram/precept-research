"""Offline tests for the RQ3a corpus loaders (DSE-041).

Every fixture here is hand-built to mirror the field layout verified against the real corpora
during the E9 spike (see docs/rq3a_schema_mapping.md). Nothing touches the network: a loader that
needed a download to be testable would be untestable in CI and on a compute node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from preceptx.data.logs import LOG_SCHEMA_VERSION, LogHandoffRecord
from preceptx.experiments.rq3a_load import (
    CorpusError,
    count_handoff_corpus,
    count_trace_corpus,
    load_mast,
    load_traceelephant,
    load_who_and_when,
    mark_handoffs,
    render_messages,
)

# ---------------------------------------------------------------------------------------------
# Fixtures mirroring the verified on-disk layouts
# ---------------------------------------------------------------------------------------------


def _te_step(step_id: int, agent: str, prompt: str, reply: str, tool: str | None = None) -> Any:
    calls = [{"function": {"name": tool, "arguments": '{"a":1}'}}] if tool else []
    return {
        "step_id": step_id,
        "agent_id": 1 if agent == "editor" else 2,
        "agent_name": agent,
        "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]},
        "output": {"choices": [{"message": {"content": reply, "tool_calls": calls}}]},
        "tool_logs": [],
    }


def _write_traceelephant(root: Path, *, failing: bool = True) -> Path:
    """Two agents over four steps: editor, editor, bash, editor -> two handoffs."""
    d = root / "data" / "swe-agent-runs-swe-bench" / "psf__requests-1724"
    d.mkdir(parents=True)
    steps = [
        _te_step(1, "editor", "look at /testbed", "", tool="str_replace_editor"),
        _te_step(2, "editor", "still looking", "found it"),
        _te_step(3, "bash", "run the tests", "3 failed"),
        _te_step(4, "editor", "patch it", "done"),
    ]
    (d / "step_records.json").write_text(json.dumps(steps))
    (d / "trace_metadata.json").write_text(
        json.dumps(
            {
                "task_id": "psf__requests-1724",
                "task_instruction": "fix the bug",
                "system_name": "swe-agent",
                "mistake_agent": "editor",
                "mistake_step": "2",
                "mistake_reason": "wrong file",
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": ["t::a"] if failing else []},
                    "PASS_TO_PASS": {"success": ["t::b"], "failure": []},
                },
            }
        )
    )
    return root / "data"


def _write_who_and_when(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "is_correct": [False],
            "question": ["q"],
            "question_ID": ["abc"],
            "ground_truth": ["42"],
            "history": [
                [
                    {"content": "plan it", "name": "Planner", "role": "assistant"},
                    {"content": "verify it", "name": "Verifier", "role": "user"},
                ]
            ],
            "mistake_agent": ["Verifier"],
            "mistake_step": ["1"],
            "mistake_reason": ["missed the price"],
        }
    ).to_parquet(root / "Algorithm-Generated.parquet")
    pd.DataFrame(
        {
            "history": [[{"content": "go", "role": "Orchestrator"}]],
            "question": ["q2"],
            "groundtruth": ["0.2"],
            "is_corrected": [False],
            "mistake_agent": ["WebSurfer"],
            "mistake_step": ["3"],
            "mistake_reason": ["no access"],
            "question_ID": ["def"],
            "mistake_type": [None],
        }
    ).to_parquet(root / "Hand-Crafted.parquet")
    return root


def _write_mast(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = {f"{a}.{b}": 0 for a in (1, 2, 3) for b in (1, 2)}
    path.write_text(
        json.dumps(
            [
                {
                    "mas_name": "ChatDev",
                    "llm_name": "GPT-4o",
                    "benchmark_name": "ProgramDev",
                    "trace_id": 0,
                    "trace": {"key": "k", "index": 0, "trajectory": "[INFO] ChatDev Starts"},
                    "mast_annotation": {**modes, "1.1": 1},
                },
                {
                    "mas_name": "AG2",
                    "llm_name": "GPT-4o",
                    "benchmark_name": "GSM",
                    "trace_id": 1,
                    "trace": {"key": "k2", "index": 1, "trajectory": "solved"},
                    "mast_annotation": dict(modes),
                },
            ]
        )
    )
    return path


# ---------------------------------------------------------------------------------------------
# Handoff extraction
# ---------------------------------------------------------------------------------------------


def test_mark_handoffs_flags_only_component_changes() -> None:
    assert mark_handoffs(["a", "a", "b", "a"]) == [
        ("a", False),  # intra-agent tool turn
        ("b", True),  # a -> b
        ("a", True),  # b -> a
        (None, False),  # no successor
    ]


def test_mark_handoffs_on_a_single_step_trace() -> None:
    assert mark_handoffs(["solo"]) == [(None, False)]


def test_mark_handoffs_on_an_empty_trace() -> None:
    assert mark_handoffs([]) == []


# ---------------------------------------------------------------------------------------------
# TraceElephant
# ---------------------------------------------------------------------------------------------


def test_traceelephant_extracts_true_observations_and_handoffs(tmp_path: Path) -> None:
    rows = load_traceelephant(_write_traceelephant(tmp_path))
    assert len(rows) == 4
    assert [r.is_handoff for r in rows] == [False, True, True, False]
    assert [r.receiver for r in rows] == ["editor", "bash", "editor", None]
    assert all(r.trace_id == "swe-agent-runs-swe-bench/psf__requests-1724" for r in rows)
    # The observation is the recorded input context, never reconstructed.
    assert not any(r.reconstructed_observation for r in rows)
    assert "look at /testbed" in rows[0].observation
    # An empty content with a tool call is a real message, not a dropped one.
    assert rows[0].message == '[tool_call str_replace_editor] {"a":1}'


def test_traceelephant_outcome_reads_tests_not_annotations(tmp_path: Path) -> None:
    failing = load_traceelephant(_write_traceelephant(tmp_path / "f", failing=True))
    passing = load_traceelephant(_write_traceelephant(tmp_path / "p", failing=False))
    assert all(r.trace_failed is True for r in failing)
    assert all(r.trace_failed is False for r in passing)
    # The annotations ride along for evaluation but are not what produced trace_failed.
    assert failing[0].annotations["mistake_agent"] == "editor"


def test_traceelephant_raises_on_a_missing_metadata_file(tmp_path: Path) -> None:
    root = _write_traceelephant(tmp_path)
    (root / "swe-agent-runs-swe-bench" / "psf__requests-1724" / "trace_metadata.json").unlink()
    with pytest.raises(CorpusError, match=r"trace_metadata\.json"):
        load_traceelephant(root)


def test_traceelephant_raises_on_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="unzipped"):
        load_traceelephant(tmp_path / "nope")


# ---------------------------------------------------------------------------------------------
# Who&When
# ---------------------------------------------------------------------------------------------


def test_who_and_when_flags_every_row_as_reconstructed(tmp_path: Path) -> None:
    rows = load_who_and_when(_write_who_and_when(tmp_path / "ww"))
    assert rows, "fixture produced no rows"
    assert all(r.reconstructed_observation for r in rows)
    assert all(r.corpus == "who_and_when" for r in rows)
    # Both splits are failures; the two spellings of the correctness column both resolve.
    assert all(r.trace_failed is True for r in rows)


def test_who_and_when_rebuilds_the_observation_from_the_prefix(tmp_path: Path) -> None:
    rows = load_who_and_when(_write_who_and_when(tmp_path / "ww"))
    alg = [r for r in rows if r.trace_id.startswith("Algorithm-Generated")]
    assert alg[0].observation == ""  # nothing precedes the first message
    assert "plan it" in alg[1].observation  # step 1 sees step 0
    assert alg[1].message == "verify it"
    assert [r.agent_name for r in alg] == ["Planner", "Verifier"]
    # Hand-Crafted has no ``name``; ``role`` carries the component identity.
    hand = [r for r in rows if r.trace_id.startswith("Hand-Crafted")]
    assert hand[0].agent_name == "Orchestrator"


def test_who_and_when_raises_on_a_missing_split(tmp_path: Path) -> None:
    root = _write_who_and_when(tmp_path / "ww")
    (root / "Hand-Crafted.parquet").unlink()
    with pytest.raises(CorpusError, match="Hand-Crafted"):
        load_who_and_when(root)


def test_only_who_and_when_sets_the_reconstruction_flag(tmp_path: Path) -> None:
    te = load_traceelephant(_write_traceelephant(tmp_path / "te"))
    ww = load_who_and_when(_write_who_and_when(tmp_path / "ww"))
    assert {r.reconstructed_observation for r in te} == {False}
    assert {r.reconstructed_observation for r in ww} == {True}


# ---------------------------------------------------------------------------------------------
# MAST
# ---------------------------------------------------------------------------------------------


def test_mast_loads_at_trace_level_and_counts_the_non_failure_class(tmp_path: Path) -> None:
    rows = load_mast(_write_mast(tmp_path / "mast" / "MAD_full_dataset.json"))
    assert [r.trace_failed for r in rows] == [True, False]
    counts = count_trace_corpus(rows)
    assert (counts.traces, counts.failures, counts.non_failures) == (2, 1, 1)
    # Trace-level means no steps are claimed, rather than steps being invented.
    assert (counts.steps, counts.handoffs) == (0, 0)


def test_mast_raises_on_a_trace_without_a_trajectory(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"trace_id": 7, "trace": {}, "mast_annotation": {}}]))
    with pytest.raises(CorpusError, match="trajectory"):
        load_mast(p)


# ---------------------------------------------------------------------------------------------
# Counts, rendering, and the schema contract
# ---------------------------------------------------------------------------------------------


def test_counts_are_per_trace_for_outcomes_and_per_step_for_steps(tmp_path: Path) -> None:
    counts = count_handoff_corpus(load_traceelephant(_write_traceelephant(tmp_path)))
    assert (counts.traces, counts.steps, counts.handoffs) == (1, 4, 2)
    assert (counts.failures, counts.non_failures) == (1, 0)
    assert counts.reconstructed_observations is False


def test_counting_nothing_fails_loud() -> None:
    with pytest.raises(CorpusError, match="no records"):
        count_handoff_corpus([])


def test_render_messages_flattens_typed_content_parts() -> None:
    rendered = render_messages(
        [
            {"role": "system", "content": "be helpful"},
            {"name": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
    )
    assert rendered == "system: be helpful\n\nuser: hello"


def test_render_messages_rejects_a_non_dict_message() -> None:
    with pytest.raises(CorpusError, match="expected a dict"):
        render_messages(["not a message"])


def test_log_record_forbids_physics_fields() -> None:
    """Physics is absent, not nullable: a log row can never be read as a degraded episode row."""
    with pytest.raises(ValueError, match=r"extra_forbidden|Extra inputs"):
        LogHandoffRecord(
            corpus="traceelephant",
            trace_id="t",
            step=0,
            agent_name="a",
            is_handoff=False,
            observation="s",
            message="m",
            state={"x": 1.0},  # type: ignore[call-arg]
        )


def test_log_schema_version_is_independent_of_the_simulator_schema() -> None:
    from preceptx.data.schema import SCHEMA_VERSION

    rec = LogHandoffRecord(
        corpus="mast",
        trace_id="t",
        step=0,
        agent_name="a",
        is_handoff=False,
        observation="s",
        message="m",
    )
    assert rec.log_schema_version == LOG_SCHEMA_VERSION
    assert "schema_version" not in rec.model_dump()
    assert SCHEMA_VERSION  # the simulator contract still exists and is not what logs key off


@given(
    agents=st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=12),
    text=st.text(max_size=40),
)
def test_every_emitted_row_validates(agents: list[str], text: str) -> None:
    """Property: whatever the trace shape, the rows the extractor builds are valid records."""
    rows = [
        LogHandoffRecord(
            corpus="traceelephant",
            trace_id="t",
            step=i,
            agent_name=a,
            receiver=receiver,
            is_handoff=is_handoff,
            observation=text,
            message=text,
        )
        for i, (a, (receiver, is_handoff)) in enumerate(
            zip(agents, mark_handoffs(agents), strict=True)
        )
    ]
    assert len(rows) == len(agents)
    assert sum(r.is_handoff for r in rows) == sum(
        agents[i] != agents[i + 1] for i in range(len(agents) - 1)
    )
    assert rows[-1].receiver is None
