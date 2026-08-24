from __future__ import annotations

import json

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

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


def _models(*ids: str) -> dict[str, object]:
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


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
    respx.get(MODELS).mock(return_value=httpx.Response(200, json=_models("test-model")))
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("pong")))
    assert LLMClient(_config()).health_check() is True


@respx.mock
def test_health_check_false_on_error() -> None:
    respx.get(MODELS).mock(return_value=httpx.Response(200, json=_models("test-model")))
    respx.post(CHAT).mock(return_value=httpx.Response(500))
    assert LLMClient(_config()).health_check() is False


@respx.mock
def test_health_check_false_when_a_different_model_is_served() -> None:
    """A leftover job serving another tier answers every call; the manifest would record the wrong
    revision, so the mismatch has to fail here rather than pass silently (DSE-002)."""
    respx.get(MODELS).mock(return_value=httpx.Response(200, json=_models("some-other-tier")))
    chat = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("pong")))
    assert LLMClient(_config()).health_check() is False
    assert chat.call_count == 0  # rejected before the smoke completion is even attempted


@respx.mock
def test_health_check_false_when_nothing_is_served() -> None:
    respx.get(MODELS).mock(return_value=httpx.Response(200, json=_models()))
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("pong")))
    assert LLMClient(_config()).health_check() is False


@respx.mock
def test_context_manager_closes() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("hi")))
    with LLMClient(_config()) as client:
        assert client.chat([ChatMessage(role="user", content="hi")]) == "hi"


# --- DSE-032: the two schema-constraint wire formats -------------------------------------------


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"action": {"enum": ["N", "S"]}},
        "required": ["action"],
        "additionalProperties": False,
    }


def _sent_schema(request: httpx.Request, mode: str) -> object:
    body = json.loads(request.content)
    if mode == "guided_json":
        return body["guided_json"]
    return body["response_format"]["json_schema"]["schema"]


@respx.mock
def test_response_format_mode_sends_openai_json_schema() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion('{"action": "N"}'))
    )
    cfg = _config().model_copy(update={"structured_mode": "response_format"})
    result = LLMClient(cfg).structured([ChatMessage(role="user", content="go")], _schema())
    assert result == {"action": "N"}
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "action"
    assert body["response_format"]["json_schema"]["strict"] is True
    # The vLLM-only keys must not leak to a local runtime that would reject them.
    assert "guided_json" not in body
    assert "guided_decoding_backend" not in body


@respx.mock
def test_both_modes_send_a_byte_identical_schema() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion('{"action": "N"}'))
    )
    schema = _schema()
    sent = []
    for mode in ("guided_json", "response_format"):
        cfg = _config().model_copy(update={"structured_mode": mode})
        LLMClient(cfg).structured([ChatMessage(role="user", content="go")], schema)
        sent.append(_sent_schema(route.calls.last.request, mode))
    assert sent[0] == sent[1] == schema  # same constraint, two wire formats


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.one_of(st.text(max_size=8), st.integers(), st.booleans(), st.lists(st.text(max_size=4))),
        max_size=5,
    )
)
@settings(max_examples=25, deadline=None)
def test_schema_round_trips_through_both_paths_unchanged(schema: dict[str, object]) -> None:
    for mode in ("guided_json", "response_format"):
        with respx.mock:
            route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("{}")))
            cfg = _config().model_copy(update={"structured_mode": mode})
            LLMClient(cfg).structured([ChatMessage(role="user", content="go")], schema)
            assert _sent_schema(route.calls.last.request, mode) == schema


# --- S1 substrate adapters: the thinking switch and the empty-content guard --------------------


@respx.mock
def test_thinking_switch_is_appended_to_the_last_user_turn() -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("ok")))
    cfg = _config().model_copy(update={"thinking_switch": "/no_think"})
    LLMClient(cfg).chat(
        [
            ChatMessage(role="system", content="you are A"),
            ChatMessage(role="user", content="describe the state"),
        ]
    )
    sent = json.loads(route.calls.last.request.content)["messages"]
    assert sent[0]["content"] == "you are A"  # the system turn is untouched
    assert sent[1]["content"] == "describe the state /no_think"


@respx.mock
def test_no_switch_leaves_the_conversation_byte_identical() -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("ok")))
    LLMClient(_config()).chat([ChatMessage(role="user", content="describe the state")])
    assert json.loads(route.calls.last.request.content)["messages"] == [
        {"role": "user", "content": "describe the state"}
    ]


@respx.mock
def test_empty_content_fails_loud() -> None:
    # A runtime that spends the whole budget on reasoning returns "" with a 200; an episode of
    # empty A-messages must crash, not look like a completed run.
    respx.post(CHAT).mock(return_value=httpx.Response(200, json=_completion("   ")))
    with pytest.raises(ServingError, match="no content"):
        LLMClient(_config()).chat([ChatMessage(role="user", content="hi")])
