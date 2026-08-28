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
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import numpy as np
import pymunk
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from preceptx.agents.channel import ChannelConfig, ChannelResult, apply_channel
from preceptx.agents.prompts import prompt_a, prompt_b
from preceptx.config import ExperimentConfig
from preceptx.data.schema import Difficulty, HandoffRecord
from preceptx.gate.integration import RuntimeGate
from preceptx.serving.client import LLMClient
from preceptx.sim.actions import (
    BodyState,
    MacroAction,
    StepConfig,
    apply_macro_action,
    detect_collision,
    detect_stuck,
    read_state,
)
from preceptx.sim.arena import ArenaGeometry, Goal, ScenarioJitter, make_scenario, slit_width_for
from preceptx.sim.outcomes import OutcomeConfig, label_episode, reached_goal, step_progress
from preceptx.sim.serialise import HISTORY_WINDOW, SceneState, history_line, serialise

logger = logging.getLogger(__name__)

# Spawn-key salt for the jitter RNG stream: the channel streams key on [seed, step] with
# step < max_steps, so any constant >= 2**16 can never collide with them.
_JITTER_SALT = 2**16


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
    gate_blocked: bool  # the runtime gate fired at this handoff (DSE-018)
    gate_retries: int  # re-prompts the block cost
    message_blocked: str | None  # the first rejected message, kept for audit
    records: list[HandoffRecord]
    done: bool


class EpisodeRunner:
    """Runs one episode of the negotiation loop end to end, returning labelled handoff records.

    Holds the injected per-role ``LLMClient``s and the fixed channel / step / outcome configs;
    ``run_episode``
    builds the per-episode scenario, compiles a fresh graph over it, runs to termination, and labels
    the trajectory (DSE-009). A stub/mock client makes the whole loop testable with no live model.
    """

    def __init__(
        self,
        client_a: LLMClient,
        client_b: LLMClient | None = None,
        *,
        max_steps: int | Mapping[Difficulty, int],
        channel_cfg: ChannelConfig | None = None,
        step_cfg: StepConfig | None = None,
        outcome_cfg: OutcomeConfig | None = None,
        jitter: ScenarioJitter | None = None,
        gate: RuntimeGate | None = None,
    ) -> None:
        self._client_a = client_a
        # Self-play stays the default and the primary cell (DSE-049): omitting client_b points both
        # roles at one client, so the path is identical to the single-client runner. A second client
        # is what unblocks the heterogeneous-pair cell (DSE-021) with no further runner change.
        self._client_b = client_b or client_a
        # Per-difficulty step budget (P1-4); a bare int applies to every difficulty (scripted tests)
        self._max_steps = max_steps
        self._channel_cfg = channel_cfg or ChannelConfig()
        self._step_cfg = step_cfg or StepConfig()
        self._outcome_cfg = outcome_cfg or OutcomeConfig()
        # None = legacy fixed start pose (scripted unit tests); sweeps always pass a jitter so the
        # seed axis is true replication (P0-2). The rng keys on [cell.seed, salt], so the same seed
        # reproduces the same pose and different seeds get different problem instances.
        self._jitter = jitter
        # None = no gate in the loop, and the path is byte-identical to the ungated runner: A's
        # prompt keeps its default `gate_feedback=False` form and the three gate fields keep their
        # HandoffRecord defaults. RQ1's frozen dataset semantics do not move (P0-4).
        self._gate = gate

    def _budget(self, difficulty: Difficulty) -> int:
        """Resolve the per-difficulty step budget (a bare int applies to every difficulty)."""
        return self._max_steps if isinstance(self._max_steps, int) else self._max_steps[difficulty]

    def run_episode(self, cell: ExperimentConfig, episode_id: str) -> list[HandoffRecord]:
        """Run one episode for ``cell`` and return its records with the four Y labels filled."""
        scenario = make_scenario(
            cell.difficulty,
            rng=None if self._jitter is None else np.random.default_rng([cell.seed, _JITTER_SALT]),
            jitter=self._jitter,
        )
        geometry = ArenaGeometry()
        slit = slit_width_for(cell.difficulty)
        budget = self._budget(cell.difficulty)
        graph = self._build(
            cell, episode_id, scenario.space, scenario.load, scenario.goal, geometry, slit, budget
        )
        init: _GraphState = {
            "step": 0,
            "state_str": "",
            "observation": "",
            "message_raw": "",
            "message_delivered": "",
            "action": "WAIT",
            "buffered": None,
            "gate_blocked": False,
            "gate_retries": 0,
            "message_blocked": None,
            "records": [],
            "done": False,
        }
        final = cast(
            _GraphState,
            graph.invoke(init, config={"recursion_limit": 3 * budget + 10}),
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
        max_steps: int,
    ) -> Any:  # langgraph's compiled graph is untyped; callers cast invoke()'s result
        client_a, client_b = self._client_a, self._client_b
        channel_cfg, step_cfg = self._channel_cfg, self._step_cfg
        gate = self._gate
        post_history: list[BodyState] = []

        def agent_a(state: _GraphState) -> dict[str, object]:
            scene = SceneState(
                load=read_state(space, load), geometry=geometry, goal=goal, slit_width=slit
            )
            # The scene is what the channel may restrict; the action history is not part of it.
            # C3 windows B's view of the *world*, and B's own past actions are not the world - they
            # are B's memory of what it already did. Appending the line after `apply_channel` keeps
            # `apply_channel` touching exactly what it touched before (CLAUDE.md: the channel
            # degrades one thing only) and keeps the history identical across all three
            # serialisations, which a whitelist/window per form could not guarantee.
            scene_str = serialise(scene, cell.serialisation)
            history = history_line(
                [(r.action["action"], r.progress) for r in state["records"][-HISTORY_WINDOW:]]
            )
            state_str = f"{scene_str}\n{history}"

            def emit(*, retry: bool) -> tuple[str, ChannelResult]:
                """One A turn through the channel. ``retry`` appends the gate feedback (DSE-045).

                Under greedy decoding a bare re-prompt is a fixed point, so the retry has to differ
                in content or the gate would re-block the identical message for every retry.
                """
                raw = client_a.chat(prompt_a(state_str, gate_feedback=retry))
                return raw, apply_channel(
                    raw,
                    cell.condition,
                    serialisation=cell.serialisation,
                    observation=scene_str,
                    cfg=channel_cfg,
                    rng=np.random.default_rng([cell.seed, state["step"]]),
                    buffered=state["buffered"],
                )

            def observed(res: ChannelResult) -> str:
                return f"{res.observation_override or scene_str}\n{history}"

            message_raw, result = emit(retry=False)
            blocked_text: str | None = None
            retries = 0
            # The gate scores the POST-channel pair - what B will actually see - because that is
            # the pair the featuriser conditions on offline (P0-1). Note the consequence under C2:
            # `message_delivered` is the previous step's buffered message, so a retry cannot change
            # what B reads this step (it changes the buffer for the next one). That is a property
            # of scoring what B sees, not a special case, so the gate does not reach into the
            # channel to correct it - flagged for DSE-025's arm selection instead.
            while (
                gate is not None
                and gate.decide(
                    observed(result), result.message_delivered, seed=cell.seed, step=state["step"]
                ).blocked
            ):
                if blocked_text is None:
                    # The FIRST rejection: `message_blocked` vs `message_delivered` is then exactly
                    # the counterfactual the causal arm is about - what B would have seen with no
                    # gate, against what it saw with one.
                    blocked_text = result.message_delivered
                if retries >= gate.max_retries:
                    break  # bounded: A proceeds with the still-blocked message, and it is recorded
                retries += 1
                message_raw, result = emit(retry=True)
            return {
                "state_str": state_str,
                "message_raw": message_raw,
                "message_delivered": result.message_delivered,
                "observation": observed(result),
                "buffered": result.new_buffer,
                "gate_blocked": blocked_text is not None,
                "gate_retries": retries,
                "message_blocked": blocked_text,
            }

        def agent_b(state: _GraphState) -> dict[str, object]:
            # Only a schema-invalid ACTION degrades to WAIT (the DSE-010-sanctioned fallback). A
            # transport-level ServingError propagates and fails the episode loud - catching it here
            # would let a dead endpoint record a passing-looking run of WAITs (P1-3).
            raw = client_b.structured(
                prompt_b(state["observation"], state["message_delivered"]),
                Action.model_json_schema(),
            )
            try:
                action: MacroAction = Action.model_validate(raw).action
            except ValidationError:
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
                observation=state["observation"],
                message_raw=state["message_raw"],
                message_delivered=state["message_delivered"],
                gate_blocked=state["gate_blocked"],
                gate_retries=state["gate_retries"],
                message_blocked=state["message_blocked"],
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
