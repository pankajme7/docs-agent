"""Provider-neutral chat-completion wrapper for OpenAI and Ollama's compatibility API."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from grounded_docs.config import Settings
from grounded_docs.logging_utils import log_payload
from grounded_docs.metrics import count_message_tokens, count_text_tokens
from grounded_docs.schemas import GroundedAnswer

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    provider_usage_available: bool
    tool_calls: list[Any]


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.provider == "openai" and not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env or switch LLM_PROVIDER=ollama.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `pip install -e .` to use an LLM provider.") from exc
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=120.0,
            max_retries=2,
        )

    def _response_format(self, json_mode: bool) -> dict[str, Any] | None:
        if not json_mode:
            return None
        if self.settings.provider == "ollama":
            # Ollama's OpenAI-compatible endpoint supports JSON mode; Pydantic validates the schema.
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "strict": True,
                "schema": GroundedAnswer.model_json_schema(),
            },
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
        }
        response_format = self._response_format(json_mode)
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        log_payload(
            logger,
            "LLM request",
            {
                "provider": self.settings.provider,
                "model": self.settings.llm_model,
                "messages": messages,
                "tools": tools or [],
                "json_mode": json_mode,
            },
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        message = response.choices[0].message
        content = message.content or ""
        usage = response.usage
        provider_usage_available = usage is not None
        prompt_tokens = usage.prompt_tokens if usage else count_message_tokens(messages, self.settings.llm_model)
        completion_tokens = usage.completion_tokens if usage else count_text_tokens(content, self.settings.llm_model)
        tool_calls = list(message.tool_calls or [])
        log_payload(
            logger,
            "LLM response",
            {
                "content": content,
                "tool_calls": [call.model_dump(exclude_none=True) for call in tool_calls],
                "latency_seconds": round(latency, 3),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
            },
        )
        return CompletionResult(
            content=content,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            latency_seconds=latency,
            provider_usage_available=provider_usage_available,
            tool_calls=tool_calls,
        )

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        # Keep streaming independent from JSON mode: this path prints incremental text.
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if isinstance(delta, str) and delta:
                yield delta


def parse_grounded_answer(raw: str) -> GroundedAnswer:
    """Parse and validate JSON; reject malformed structure instead of silently trusting it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(text)
        return GroundedAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Model output did not satisfy GroundedAnswer schema: %s", exc)
        # Preserve the text but make the validation failure visible in the answer state.
        return GroundedAnswer(
            answer=raw.strip() or "The model returned an empty response.",
            sources=[],
            insufficient_context=True,
        )
