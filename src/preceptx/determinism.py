"""Quantify LLM run-to-run variance at a fixed seed - the honest input to the limitations section.

Batched inference is not bit-exact, so a seed-pinned call is not guaranteed identical across
repeats. This harness issues the *same* structured action request ``k`` times against a served model
and reports how often the answer agrees and how much its numeric fields vary. It probes determinism
at the single structured-call granularity; full-episode determinism (a fixed-seed trajectory) lands
with the episode runner (DSE-012) and reuses this report. The point is to *measure and report* the
residual variance, never to suppress it or claim exact reproducibility.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict

from preceptx.serving.client import ChatMessage, LLMClient

logger = logging.getLogger(__name__)


class DeterminismReport(BaseModel):
    """Result of repeating one fixed-seed structured call ``k`` times."""

    model_config = ConfigDict(extra="forbid")

    k: int
    distinct_outputs: int
    agreement_rate: float
    """Fraction of the ``k`` outputs equal to the most common output (1.0 = full agreement)."""
    numeric_variance: dict[str, float]
    """Population variance of each numeric action field across the ``k`` outputs."""


def _numeric_variance(outputs: list[dict[str, Any]]) -> dict[str, float]:
    if not outputs:
        return {}
    variances: dict[str, float] = {}
    for key in outputs[0]:
        values = [o[key] for o in outputs if isinstance(o.get(key), int | float)]
        if len(values) == len(outputs):  # numeric in every run
            mean = sum(values) / len(values)
            variances[key] = sum((v - mean) ** 2 for v in values) / len(values)
    return variances


def run_determinism_check(
    client: LLMClient,
    messages: list[ChatMessage],
    schema: dict[str, Any],
    *,
    k: int,
) -> DeterminismReport:
    """Issue the same structured request ``k`` times and report agreement and numeric variance."""
    if k < 2:
        raise ValueError(f"determinism check needs k >= 2 repeats, got {k}")
    outputs = [client.structured(messages, schema) for _ in range(k)]
    keys = [tuple(sorted(o.items())) for o in outputs]
    counts = Counter(keys)
    agreement_rate = counts.most_common(1)[0][1] / k
    report = DeterminismReport(
        k=k,
        distinct_outputs=len(counts),
        agreement_rate=agreement_rate,
        numeric_variance=_numeric_variance(outputs),
    )
    logger.info(
        "determinism: k=%d distinct=%d agreement=%.3f", k, report.distinct_outputs, agreement_rate
    )
    return report
