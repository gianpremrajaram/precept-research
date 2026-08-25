"""Thin client over a vLLM OpenAI-compatible endpoint.

Greedy, seed-pinned decoding is enforced here (``temperature`` defaults to 0), so determinism is a
client property independent of the served model. The client is model-agnostic: switching ladder
tiers is a config change, not a code change. The serving process itself runs on Myriad GPU nodes via
``scripts/myriad/serve.sh``; this module only talks to it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ServingError(RuntimeError):
    """A serving call failed or returned an unusable response."""


class ChatMessage(BaseModel):
    """One chat message in an OpenAI-style conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ServingConfig(BaseModel):
    """Connection and decoding settings for a vLLM OpenAI-compatible endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 0
    max_tokens: int = Field(default=512, gt=0)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # Wire format for schema-constrained decoding (DSE-032). vLLM takes the schema in extra_body;
    # an OpenAI-compatible local runtime (the free pre-cluster pilot) takes the standard
    # `response_format.json_schema`. Both constrain decoding against the SAME schema object -
    # xgrammar under vLLM, llama.cpp grammars / Outlines locally - so this is a wire-format
    # difference, not a semantic one. Default keeps served-vLLM behaviour unchanged.
    #
    # The value is still spelled `guided_json` after vLLM renamed the field (DSE-052): it names
    # which endpoint dialect the branch speaks, not the field, and it is recorded in every
    # SweepManifest. Renaming it would churn a CLI contract and a manifest label to no effect.
    structured_mode: Literal["guided_json", "response_format"] = "guided_json"
    # Rendered into the served model's chat template per request (P0-3). The default disables
    # Qwen3's hybrid thinking - greedy decoding in thinking mode is explicitly discouraged by Qwen,
    # and a CoT dump in the A->B message would confound every channel condition. Templates that do
    # not use the variable ignore it; set {} for endpoints that reject unknown body keys.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})
    # In-band fallback for runtimes that ignore chat_template_kwargs. LM Studio's MLX runtime does:
    # Qwen3 stays in thinking mode, the reasoning lands in a non-standard `reasoning_content` field
    # and `content` comes back EMPTY. Qwen3's `/no_think` switch selects the same non-thinking
    # branch the cluster selects via the template kwarg, so this is a substrate adapter, not a
    # prompt change - the rendered conversation is what differs between substrates, and it is
    # recorded per run. Empty (the default) leaves the vLLM path untouched.
    thinking_switch: str = ""


class LLMClient:
    """Wraps an OpenAI-compatible client pointed at the local vLLM endpoint."""

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @property
    def config(self) -> ServingConfig:
        return self._config

    def _payload(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        payload = [m.model_dump() for m in messages]
        if self._config.thinking_switch:  # appended to the final user turn, where Qwen3 reads it
            last_user = next(
                (i for i in reversed(range(len(payload))) if payload[i]["role"] == "user"), None
            )
            if last_user is not None:
                payload[last_user]["content"] += f" {self._config.thinking_switch}"
        return cast("list[ChatCompletionMessageParam]", payload)

    def _template_kwargs(self) -> dict[str, Any]:
        """The chat-template extra-body payload, empty when no kwargs are configured."""
        if not self._config.chat_template_kwargs:
            return {}
        return {"chat_template_kwargs": self._config.chat_template_kwargs}

    def _structured_kwargs(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Request kwargs carrying ``schema`` in whichever wire format the endpoint speaks.

        The schema object itself is passed through untouched in both branches, so the constraint
        the two backends enforce is identical and a schema-adherence difference between them is a
        property of the runtime, not of this code (DSE-005 reports that rate).
        """
        if self._config.structured_mode == "guided_json":
            # vLLM removed the `guided_*` request fields in v0.12.0 (DSE-052); `guided_json` and
            # `guided_decoding_backend` are both gone, replaced by the unified `structured_outputs`
            # object. The backend is no longer a per-request choice at all - it is the server's
            # `--structured-outputs-config.backend`, which serve.sh sets and records in serve_env.
            return {
                "extra_body": {
                    "structured_outputs": {"json": schema},
                    **self._template_kwargs(),
                }
            }
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": schema, "strict": True},
            },
            "extra_body": self._template_kwargs() or None,
        }

    def chat(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> str:
        """Return the assistant message content for a chat completion.

        Fails loud on thinking-mode output: a ``<think>`` block in the A->B message is a category
        error (the channel would degrade reasoning, not the instruction), never a degraded mode.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._payload(messages),
                temperature=self._config.temperature,
                seed=self._config.seed,
                max_tokens=max_tokens or self._config.max_tokens,
                extra_body=self._template_kwargs() or None,
            )
        except openai.APIError as exc:
            raise ServingError(
                f"chat completion failed for model {self._config.model!r}: {exc}"
            ) from exc
        content = response.choices[0].message.content
        # Empty is as broken as absent and far more dangerous: a runtime that spends the whole
        # budget in a reasoning channel returns "" with a 200, and an episode of empty A-messages
        # would look like a completed run rather than a failed one.
        if content is None or not content.strip():
            raise ServingError(
                "chat completion returned no content; if the endpoint reports reasoning tokens, "
                "it is ignoring chat_template_kwargs - set ServingConfig.thinking_switch"
            )
        if "<think>" in content:
            raise ServingError(
                "thinking-mode output detected ('<think>' in the completion); disable it via "
                "chat_template_kwargs={'enable_thinking': False} on the serving config"
            )
        return content

    def structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a JSON object constrained to ``schema`` by the endpoint's guided decoding."""
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._payload(messages),
                temperature=self._config.temperature,
                seed=self._config.seed,
                max_tokens=max_tokens or self._config.max_tokens,
                **self._structured_kwargs(schema),
            )
        except openai.APIError as exc:
            raise ServingError(
                f"structured completion failed for model {self._config.model!r}: {exc}"
            ) from exc
        content = response.choices[0].message.content
        if content is None:
            raise ServingError("structured completion returned no content")
        try:
            parsed: object = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ServingError(f"structured output was not valid JSON: {content!r}") from exc
        if not isinstance(parsed, dict):
            raise ServingError(f"structured output was not a JSON object: {parsed!r}")
        return cast("dict[str, Any]", parsed)

    def health_check(self) -> bool:
        """Return True if the endpoint serves the configured model and a smoke completion succeeds.

        Two pre-flight failures are caught here rather than mid-sweep. The ping asks for a few
        tokens rather than one, because a runtime stuck in thinking mode returns an empty
        completion. And the served ids are *compared* against ``config.model`` rather than merely
        fetched: pointed at a leftover job serving a different tier, every call would succeed and
        the manifest would record the tier that was configured instead of the one that answered.
        A wrong recorded revision is worse than a missing one (DSE-002).
        """
        try:
            served = [model.id for model in self._client.models.list().data]
            if self.config.model not in served:
                logger.warning(
                    "serving health check failed: %s serves %s, not the configured %r",
                    self.config.base_url,
                    served or "no model",
                    self.config.model,
                )
                return False
            self.chat(
                [
                    ChatMessage(role="user", content="Reply with the word pong."),
                ],
                max_tokens=16,
            )
        except (openai.APIError, ServingError) as exc:
            logger.warning("serving health check failed: %s", exc)
            return False
        return True

    def close(self) -> None:
        """Close HTTP connections. The served job is torn down separately via ``qdel``."""
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
