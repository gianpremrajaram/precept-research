"""Determinism harness against a mocked endpoint (the real-model 8B run needs Myriad/GPU)."""

from __future__ import annotations

import itertools
import json

import httpx
import respx

from preceptx.determinism import run_determinism_check
from preceptx.serving.client import ChatMessage, LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"

_SCHEMA = {"type": "object", "properties": {"dx": {"type": "number"}}}
_MESSAGES = [ChatMessage(role="user", content="nudge")]


def _completion(action: dict[str, float]) -> dict[str, object]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(action)},
                "finish_reason": "stop",
            }
        ],
    }


def _client() -> LLMClient:
    return LLMClient(ServingConfig(model="test-model", base_url=BASE_URL, max_retries=0))


@respx.mock
def test_deterministic_model_reports_full_agreement() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion({"dx": 1.0})))
    report = run_determinism_check(_client(), _MESSAGES, _SCHEMA, k=5)
    assert report.k == 5
    assert report.distinct_outputs == 1
    assert report.agreement_rate == 1.0
    assert report.numeric_variance["dx"] == 0.0


@respx.mock
def test_varying_model_reports_partial_agreement_and_variance() -> None:
    # Three identical answers then two different ones: modal count 3 of 5.
    answers = itertools.cycle([1.0, 1.0, 1.0, 2.0, 3.0])
    respx.post(CHAT).mock(
        side_effect=lambda request: httpx.Response(200, json=_completion({"dx": next(answers)}))
    )
    report = run_determinism_check(_client(), _MESSAGES, _SCHEMA, k=5)
    assert report.distinct_outputs == 3
    assert report.agreement_rate == 0.6
    assert report.numeric_variance["dx"] > 0.0
