from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from preceptx.data.otel_capture import emit_handoff
from preceptx.data.schema import HandoffRecord


def _record() -> HandoffRecord:
    return HandoffRecord(
        episode_id="ep",
        step=2,
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model="m",
        seed=0,
        state={"x": 1.0},
        state_str="x=1.0",
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


def test_emit_handoff_no_ops_without_exporter() -> None:
    # Default global tracer provider has no exporter: must not raise.
    emit_handoff(_record())


def test_emit_handoff_records_span_with_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    emit_handoff(_record(), tracer=tracer)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "handoff"
    assert span.attributes is not None
    assert span.attributes["condition"] == "C0"
    assert span.attributes["step"] == 2
    assert span.attributes["state"] == '{"x": 1.0}'  # nested dict as JSON string
