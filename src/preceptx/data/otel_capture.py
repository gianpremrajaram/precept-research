"""Emit each handoff as an OpenTelemetry span via the vanilla OTel SDK - fail-open by design.

This is in-repo capture with no precept dependency. If no tracer provider/exporter is configured,
the default OTel API returns a no-op tracer, so ``emit_handoff`` does nothing and never raises:
telemetry is observability, not a result, and must not crash a run by its absence. (The OTel API is
no-throw by contract, so no defensive catch is needed - and a broad ``except`` is banned anyway.)
Nested physics/action dicts attach as JSON-string attributes (OTel attributes must be scalars).
"""

from __future__ import annotations

import json

from opentelemetry import trace
from opentelemetry.trace import Tracer

from preceptx.data.schema import HandoffRecord

# Scalar fields attached directly as span attributes; nested dicts go on as JSON strings.
_SCALAR_FIELDS = (
    "schema_version",
    "episode_id",
    "step",
    "condition",
    "serialisation",
    "difficulty",
    "model",
    "seed",
    "state_str",
    "message_raw",
    "message_delivered",
    "progress",
    "success",
    "collision",
    "stuck",
)
_NESTED_FIELDS = ("state", "action", "pre_state", "post_state")


def emit_handoff(record: HandoffRecord, *, tracer: Tracer | None = None) -> None:
    """Emit ``record`` as a ``handoff`` span. No-ops cleanly when no exporter is configured."""
    tracer = tracer or trace.get_tracer(__name__)
    with tracer.start_as_current_span("handoff") as span:
        for field in _SCALAR_FIELDS:
            span.set_attribute(field, getattr(record, field))
        for field in _NESTED_FIELDS:
            span.set_attribute(field, json.dumps(getattr(record, field), sort_keys=True))
