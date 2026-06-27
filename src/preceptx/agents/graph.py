"""The two-agent negotiation loop as a LangGraph ``StateGraph`` (DSE-010).

Nodes: ``agent_A`` emits a natural-language handoff, ``agent_B`` chooses a structured ``Action`` via
guided decoding, ``apply`` steps the simulator and records the handoff. A conditional edge loops
back to ``agent_A`` until the goal is reached or the step budget is spent. The graph is
framework-thin: LangGraph only sequences the nodes; all task logic lives in plain closures and the
injected ``LLMClient``, so a LangGraph API change touches only this module. The A->B message passes
through one choke point (``apply_channel``) - the seam the runtime gate (DSE-018) later intercepts.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import numpy as np
import pymunk
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from preceptx.agents.channel import ChannelConfig, apply_channel
from preceptx.agents.prompts import prompt_a, prompt_b
from preceptx.config import ExperimentConfig
from preceptx.data.schema import HandoffRecord
from preceptx.serving.client import LLMClient, ServingError
from preceptx.sim.actions import (
    BodyState,
    MacroAction,
    StepConfig,
    apply_macro_action,
    detect_collision,
    detect_stuck,
    read_state,
)
from preceptx.sim.arena import ArenaGeometry, Goal, make_scenario, slit_width_for
from preceptx.sim.outcomes import OutcomeConfig, label_episode, reached_goal, step_progress
from preceptx.sim.serialise import SceneState, serialise

logger = logging.getLogger(__name__)


class Action(BaseModel):
    """B's structured action; the JSON schema is enforced by vLLM guided decoding."""

    model_config = ConfigDict(extra="forbid")

    action: MacroAction


class _GraphState(TypedDict):
    """Dynamic per-step state threaded through the graph; static handles are closure-bound."""

    step: int
    state_str: str  # full serialised state (what A sees, recorded on the handoff)
    observation: str  # B's view: state_str, or a C3-restricted window
    message_raw: str  # A's message before the channel
    message_delivered: str  # what B actually receives
    action: MacroAction  # B's chosen action this step
    buffered: str | None  # C2 one-step delay buffer
    records: list[HandoffRecord]
    done: bool


class EpisodeRunner:
    """Runs one episode of the negotiation loop end to end, returning labelled handoff records.

    Holds the injected ``LLMClient`` and the fixed channel / step / outcome configs; ``run_episode``
    builds the per-episode scenario, compiles a fresh graph over it, runs to termination, and labels
    the trajectory (DSE-009). A stub/mock client makes the whole loop testable with no live model.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_steps: int,
        channel_cfg: ChannelConfig | None = None,
        step_cfg: StepConfig | None = None,
        outcome_cfg: OutcomeConfig | None = None,
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._channel_cfg = channel_cfg or ChannelConfig()
        self._step_cfg = step_cfg or StepConfig()
        self._outcome_cfg = outcome_cfg or OutcomeConfig()

    def run_episode(self, cell: ExperimentConfig, episode_id: str) -> list[HandoffRecord]:
        """Run one episode for ``cell`` and return its records with the four Y labels filled."""
        scenario = make_scenario(cell.difficulty)
        geometry = ArenaGeometry()
        slit = slit_width_for(cell.difficulty)
        graph = self._build(
            cell, episode_id, scenario.space, scenario.load, scenario.goal, geometry, slit
        )
        init: _GraphState = {
            "step": 0,
            "state_str": "",
            "observation": "",
            "message_raw": "",
            "message_delivered": "",
            "action": "WAIT",
            "buffered": None,
            "records": [],
            "done": False,
        }
        final = cast(
            _GraphState,
            graph.invoke(init, config={"recursion_limit": 3 * self._max_steps + 10}),
        )
        return label_episode(final["records"], scenario.goal, geometry, self._outcome_cfg)

    def _build(
        self,
        cell: ExperimentConfig,
        episode_id: str,
        space: pymunk.Space,
        load: pymunk.Body,
        goal: Goal,
        geometry: ArenaGeometry,
        slit: float,
    ) -> Any:  # langgraph's compiled graph is untyped; callers cast invoke()'s result
        client, channel_cfg, step_cfg, max_steps = (
            self._client,
            self._channel_cfg,
            self._step_cfg,
            self._max_steps,
        )
        post_history: list[BodyState] = []

        def agent_a(state: _GraphState) -> dict[str, object]:
            scene = SceneState(
                load=read_state(space, load), geometry=geometry, goal=goal, slit_width=slit
            )
            state_str = serialise(scene, cell.serialisation)
            message_raw = client.chat(prompt_a(state_str))
            result = apply_channel(
                message_raw,
                cell.condition,
                serialisation=cell.serialisation,
                observation=state_str,
                cfg=channel_cfg,
                rng=np.random.default_rng([cell.seed, state["step"]]),
                buffered=state["buffered"],
            )
            return {
                "state_str": state_str,
                "message_raw": message_raw,
                "message_delivered": result.message_delivered,
                "observation": result.observation_override or state_str,
                "buffered": result.new_buffer,
            }

        def agent_b(state: _GraphState) -> dict[str, object]:
            try:
                raw = client.structured(
                    prompt_b(state["observation"], state["message_delivered"]),
                    Action.model_json_schema(),
                )
                action: MacroAction = Action.model_validate(raw).action
            except (ServingError, ValidationError):
                logger.warning("agent_B emitted an invalid action; defaulting to WAIT")
                action = "WAIT"
            return {"action": action}

        def apply_node(state: _GraphState) -> dict[str, object]:
            pre = read_state(space, load)
            action = state["action"]
            apply_macro_action(space, load, action, step_cfg)
            post = read_state(space, load)
            post_history.append(post)
            success = reached_goal(post, goal)
            record = HandoffRecord(
                episode_id=episode_id,
                step=state["step"],
                condition=cell.condition,
                serialisation=cell.serialisation,
                difficulty=cell.difficulty,
                model=cell.model.name,
                seed=cell.seed,
                state=pre.model_dump(),
                state_str=state["state_str"],
                message_raw=state["message_raw"],
                message_delivered=state["message_delivered"],
                action={"action": action},
                pre_state=pre.model_dump(),
                post_state=post.model_dump(),
                progress=step_progress(pre, post, goal, geometry),
                success=success,
                collision=detect_collision(post),
                stuck=detect_stuck(post_history),
            )
            next_step = state["step"] + 1
            return {
                "step": next_step,
                "records": [*state["records"], record],
                "done": success or next_step >= max_steps,
            }

        def route(state: _GraphState) -> str:
            return "stop" if state["done"] else "continue"

        graph = StateGraph(_GraphState)
        graph.add_node("agent_A", agent_a)
        graph.add_node("agent_B", agent_b)
        graph.add_node("apply", apply_node)
        graph.set_entry_point("agent_A")
        graph.add_edge("agent_A", "agent_B")
        graph.add_edge("agent_B", "apply")
        graph.add_conditional_edges("apply", route, {"continue": "agent_A", "stop": END})
        return graph.compile()
