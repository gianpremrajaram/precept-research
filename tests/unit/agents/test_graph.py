from __future__ import annotations

import json

import httpx
import pytest
import respx

from preceptx.agents.channel import ChannelConfig
from preceptx.agents.graph import EpisodeRunner
from preceptx.config import ExperimentConfig, ModelConfig
from preceptx.data.schema import Condition, Difficulty, Serialisation
from preceptx.serving.client import LLMClient, ServingConfig, ServingError
from preceptx.sim.arena import ScenarioJitter

BASE_URL = "http://localhost:8000/v1"
CHAT = f"{BASE_URL}/chat/completions"


def _client() -> LLMClient:
    return LLMClient(ServingConfig(model="m", base_url=BASE_URL, max_retries=0))


def _cell(
    condition: Condition = "C0",
    serialisation: Serialisation = "numeric",
    difficulty: Difficulty = "easy",
    seed: int = 0,
) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        serialisation=serialisation,
        difficulty=difficulty,
        model=ModelConfig(name="m", revision="rev", tier="8b"),
        seed=seed,
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


def _script(action: str, *, prefix: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    """Route A's chat to a fixed instruction and B's structured call to ``action``.

    ``prefix`` emits a leading action sequence before settling on ``action`` - needed since
    DSE-058, because the load starts broadside and pure east never threads the channel.
    """
    remaining = list(prefix)

    def handler(request: httpx.Request) -> httpx.Response:
        if b"structured_outputs" in request.content:  # B's action call carries the schema
            nxt = remaining.pop(0) if remaining else action
            return httpx.Response(200, json=_completion(json.dumps({"action": nxt})))
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
    # The certified easy path (DSE-059): five rotations onto the passing angle, then push east.
    # Five, not two, because the rotation step is now 12 deg rather than 57.8; the certificate in
    # tests/unit/sim/test_feasibility.py is what guards the count, this test only has to use it.
    respx.post(CHAT).mock(side_effect=_script("E", prefix=("ROT+",) * 5))
    records = EpisodeRunner(_client(), max_steps=20).run_episode(_cell(), "ep")
    assert records[-1].success  # reached the goal
    assert len(records) < 20  # stopped early, before the budget
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


@respx.mock
def test_records_persist_the_receiver_observation() -> None:
    # P0-1: the record carries B's delivered view. Full visibility -> observation == state_str;
    # C3 -> the restricted window (numeric mode hides the goal line), with state_str kept intact.
    respx.post(CHAT).mock(side_effect=_script("WAIT"))
    c0 = EpisodeRunner(_client(), max_steps=1).run_episode(_cell(), "ep")
    assert c0[0].observation == c0[0].state_str

    respx.post(CHAT).mock(side_effect=_script("WAIT"))
    c3 = EpisodeRunner(_client(), max_steps=1).run_episode(_cell(condition="C3"), "ep")
    assert c3[0].observation != c3[0].state_str
    assert "goal=" not in c3[0].observation  # the window is what B actually saw
    assert "goal=" in c3[0].state_str  # the full state stays for the dual-baseline diagnostic


@respx.mock
def test_transport_error_on_action_call_fails_the_episode_loud() -> None:
    # P1-3: a dead endpoint mid-episode must crash the run, not record a passing-looking WAIT.
    def handler(request: httpx.Request) -> httpx.Response:
        if b"structured_outputs" in request.content:  # B's action call: transport failure
            return httpx.Response(500)
        return httpx.Response(200, json=_completion("push the load east"))

    respx.post(CHAT).mock(side_effect=handler)
    with pytest.raises(ServingError):
        EpisodeRunner(_client(), max_steps=3).run_episode(_cell(), "ep")


@respx.mock
def test_jittered_runner_reproduces_within_seed_and_varies_across_seeds() -> None:
    respx.post(CHAT).mock(side_effect=_script("WAIT"))
    runner = EpisodeRunner(_client(), max_steps=1, jitter=ScenarioJitter())
    a = runner.run_episode(_cell(seed=0), "ep-a")
    b = runner.run_episode(_cell(seed=0), "ep-b")
    assert a[0].pre_state == b[0].pre_state  # same seed -> same jittered instance
    c = runner.run_episode(_cell(seed=7), "ep-c")
    assert c[0].pre_state != a[0].pre_state  # different seed -> different instance


# --- DSE-049: per-role clients -----------------------------------------------------------------

BASE_URL_B = "http://localhost:8001/v1"
CHAT_B = f"{BASE_URL_B}/chat/completions"


def _client_b() -> LLMClient:
    return LLMClient(ServingConfig(model="mb", base_url=BASE_URL_B, max_retries=0))


@respx.mock
def test_omitting_client_b_reproduces_the_single_client_path() -> None:
    respx.post(CHAT).mock(side_effect=_script("WAIT"))
    client = _client()
    one = EpisodeRunner(client, max_steps=3).run_episode(_cell(), "ep")
    two = EpisodeRunner(client, None, max_steps=3).run_episode(_cell(), "ep")
    assert [r.model_dump() for r in one] == [r.model_dump() for r in two]


@respx.mock
def test_each_role_calls_only_its_own_client() -> None:
    # A's endpoint only ever serves the message; B's only ever serves the structured action.
    route_a = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_completion("push the load east"))
    )
    route_b = respx.post(CHAT_B).mock(
        return_value=httpx.Response(200, json=_completion(json.dumps({"action": "WAIT"})))
    )
    records = EpisodeRunner(_client(), _client_b(), max_steps=2).run_episode(_cell(), "ep")

    assert len(records) == 2
    assert route_a.call_count == 2 and route_b.call_count == 2
    assert all(b"structured_outputs" not in c.request.content for c in route_a.calls)
    assert all(b"structured_outputs" in c.request.content for c in route_b.calls)


@respx.mock
def test_history_line_reaches_both_agents_and_grows_with_the_episode() -> None:
    """v5: the prompt surface carries what the last actions gained, so a greedy policy facing a
    repeated state is no longer facing a repeated prompt."""
    respx.post(CHAT).mock(side_effect=_script("E"))
    records = EpisodeRunner(_client(), max_steps=3).run_episode(_cell(), "ep")
    assert "recent=()" in records[0].state_str  # step 0: nothing has happened yet
    assert "recent=()" in records[0].observation
    assert "(E, +" in records[1].state_str  # step 1: the first push and its gain
    assert "(E, +" in records[1].observation


@respx.mock
def test_c3_restricts_the_scene_but_never_b_s_own_action_history() -> None:
    """The channel degrades one thing only. C3 windows B's view of the *world*; B's memory of what
    it already did is not the world, so the history line survives the restriction intact."""
    respx.post(CHAT).mock(side_effect=_script("E"))
    records = EpisodeRunner(_client(), max_steps=3).run_episode(_cell(condition="C3"), "ep")
    assert "goal=" not in records[1].observation  # the scene is still restricted
    assert "recent=" in records[1].observation  # the history is not
    assert records[1].observation.splitlines()[-1] == records[1].state_str.splitlines()[-1]
