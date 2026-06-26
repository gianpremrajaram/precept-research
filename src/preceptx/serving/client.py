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
    guided_decoding_backend: Literal["xgrammar", "outlines"] = "xgrammar"


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
        return cast("list[ChatCompletionMessageParam]", [m.model_dump() for m in messages])

    def chat(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> str:
        """Return the assistant message content for a chat completion."""
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._payload(messages),
                temperature=self._config.temperature,
                seed=self._config.seed,
                max_tokens=max_tokens or self._config.max_tokens,
            )
        except openai.APIError as exc:
            raise ServingError(
                f"chat completion failed for model {self._config.model!r}: {exc}"
            ) from exc
        content = response.choices[0].message.content
        if content is None:
            raise ServingError("chat completion returned no content")
        return content

    def structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a JSON object constrained to ``schema`` via vLLM guided decoding."""
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._payload(messages),
                temperature=self._config.temperature,
                seed=self._config.seed,
                max_tokens=max_tokens or self._config.max_tokens,
                extra_body={
                    "guided_json": schema,
                    "guided_decoding_backend": self._config.guided_decoding_backend,
                },
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
        """Return True if ``/v1/models`` is reachable and a smoke completion succeeds."""
        try:
            self._client.models.list()
            self.chat([ChatMessage(role="user", content="ping")], max_tokens=1)
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
