from __future__ import annotations

import httpx
import pytest
import respx

from preceptx.serving.client import ChatMessage, LLMClient, ServingConfig, ServingError

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"
MODELS = f"{BASE_URL}/models"


def _config(max_retries: int = 0) -> ServingConfig:
    return ServingConfig(model="test-model", base_url=BASE_URL, max_retries=max_retries)


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@respx.mock
def test_chat_returns_content() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("hello")))
    assert LLMClient(_config()).chat([ChatMessage(role="user", content="hi")]) == "hello"


@respx.mock
def test_structured_parses_json_object() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion('{"action": "N"}'))
    )
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}
    result = LLMClient(_config()).structured([ChatMessage(role="user", content="go")], schema)
    assert result == {"action": "N"}
    # The schema is forwarded to vLLM guided decoding.
    assert b"guided_json" in route.calls.last.request.content


@respx.mock
def test_structured_rejects_non_json() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("not json")))
    with pytest.raises(ServingError, match="not valid JSON"):
        LLMClient(_config()).structured(
            [ChatMessage(role="user", content="go")], {"type": "object"}
        )


@respx.mock
def test_structured_rejects_non_object() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("[1, 2, 3]")))
    with pytest.raises(ServingError, match="not a JSON object"):
        LLMClient(_config()).structured(
            [ChatMessage(role="user", content="go")], {"type": "object"}
        )


@respx.mock
def test_chat_wraps_api_error() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(500))
    with pytest.raises(ServingError, match="chat completion failed"):
        LLMClient(_config()).chat([ChatMessage(role="user", content="hi")])


@respx.mock
def test_chat_sends_thinking_off_template_kwargs_by_default() -> None:
    # P0-3: Qwen3's hybrid thinking must be disabled per request; the default config carries it.
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("ok")))
    LLMClient(_config()).chat([ChatMessage(role="user", content="hi")])
    body = route.calls.last.request.content
    assert b"chat_template_kwargs" in body and b"enable_thinking" in body


@respx.mock
def test_structured_sends_template_kwargs_alongside_guided_decoding() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion('{"action": "N"}'))
    )
    LLMClient(_config()).structured([ChatMessage(role="user", content="go")], {"type": "object"})
    body = route.calls.last.request.content
    assert b"guided_json" in body and b"chat_template_kwargs" in body


@respx.mock
def test_empty_template_kwargs_are_omitted_from_the_request() -> None:
    # Escape hatch for endpoints that reject unknown body keys (e.g. hosted-API fallbacks).
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("ok")))
    cfg = ServingConfig(
        model="test-model", base_url=BASE_URL, max_retries=0, chat_template_kwargs={}
    )
    LLMClient(cfg).chat([ChatMessage(role="user", content="hi")])
    assert b"chat_template_kwargs" not in route.calls.last.request.content


@respx.mock
def test_chat_fails_loud_on_thinking_mode_output() -> None:
    # A CoT dump in the A->B message is a category error, never a degraded mode (P0-3 guard).
    respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion("<think>plan...</think>push east"))
    )
    with pytest.raises(ServingError, match="thinking-mode"):
        LLMClient(_config()).chat([ChatMessage(role="user", content="hi")])


@respx.mock
def test_chat_retries_transient_error() -> None:
    route = respx.post(CHAT).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_completion("recovered")),
        ]
    )
    client = LLMClient(_config(max_retries=2))
    assert client.chat([ChatMessage(role="user", content="hi")]) == "recovered"
    assert route.call_count == 2


@respx.mock
def test_health_check_true() -> None:
    respx.get(MODELS).mock(return_value=httpx.Response(200, json={"object": "list", "data": []}))
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("pong")))
    assert LLMClient(_config()).health_check() is True


@respx.mock
def test_health_check_false_on_error() -> None:
    respx.get(MODELS).mock(return_value=httpx.Response(200, json={"object": "list", "data": []}))
    respx.post(CHAT).mock(return_value=httpx.Response(500))
    assert LLMClient(_config()).health_check() is False


@respx.mock
def test_context_manager_closes() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("hi")))
    with LLMClient(_config()) as client:
        assert client.chat([ChatMessage(role="user", content="hi")]) == "hi"
