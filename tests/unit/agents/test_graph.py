from __future__ import annotations

import json

import httpx
import respx

from preceptx.agents.channel import ChannelConfig
from preceptx.agents.graph import EpisodeRunner
from preceptx.config import ExperimentConfig, ModelConfig
from preceptx.data.schema import Condition, Difficulty, Serialisation
from preceptx.serving.client import LLMClient, ServingConfig

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


def _client() -> LLMClient:
    return LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))


def _cell(
    condition: Condition = "C0",
    serialisation: Serialisation = "numeric",
    difficulty: Difficulty = "easy",
) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        serialisation=serialisation,
        difficulty=difficulty,
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        seed=0,
    )


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _script(action: str):  # type: ignore[no-untyped-def]
    """Route A's chat to a fixed instruction and B's structured call to ``action``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if b"guided_json" in request.content:  # B's guided-decoding call carries the schema
            return httpx.Response(200, json=_completion(json.dumps({"action": action})))
        return httpx.Response(200, json=_completion("push the load east"))

    return handler


@respx.mock
def test_episode_runs_to_budget_when_action_never_succeeds() -> None:
    respx.post(CHAT).mock(side_effect=_script("WAIT"))  # WAIT never moves the load
    records = EpisodeRunner(_client(), max_steps=4).run_episode(_cell(), "ep")
    assert len(records) == 4  # loops exactly to the budget
    assert all(r.action == {"action": "WAIT"} for r in records)
    assert not records[-1].success


@respx.mock
def test_episode_terminates_on_success() -> None:
    respx.post(CHAT).mock(side_effect=_script("E"))  # 7 east pushes reach the easy goal
    records = EpisodeRunner(_client(), max_steps=12).run_episode(_cell(), "ep")
    assert records[-1].success  # reached the goal
    assert len(records) < 12  # stopped early, before the budget
    assert records[-1].y_terminal_success  # labelled true after the episode


@respx.mock
def test_invalid_action_falls_back_to_wait() -> None:
    respx.post(CHAT).mock(side_effect=_script("JUMP"))  # not a MacroAction
    records = EpisodeRunner(_client(), max_steps=3).run_episode(_cell(), "ep")
    assert all(r.action == {"action": "WAIT"} for r in records)  # schema-violating -> WAIT


@respx.mock
def test_episode_is_deterministic_under_fixed_responses() -> None:
    respx.post(CHAT).mock(side_effect=_script("E"))
    a = EpisodeRunner(_client(), max_steps=6).run_episode(_cell(), "ep")
    respx.post(CHAT).mock(side_effect=_script("E"))
    b = EpisodeRunner(_client(), max_steps=6).run_episode(_cell(), "ep")
    assert [r.post_state for r in a] == [r.post_state for r in b]  # identical trajectory


@respx.mock
def test_records_capture_channel_delivery_under_c1() -> None:
    respx.post(CHAT).mock(side_effect=_script("WAIT"))
    runner = EpisodeRunner(_client(), max_steps=1, channel_cfg=ChannelConfig(c1_max_tokens=2))
    records = runner.run_episode(_cell(condition="C1"), "ep")
    assert records[0].message_raw == "push the load east"  # A's full message
    assert records[0].message_delivered == "push the"  # capped to 2 tokens by C1
