"""Claude provider via the Anthropic SDK.

``structured_output=True`` enforces the JSON schema through ``output_config.format``.
System prompt and few-shot block carry ``cache_control`` so only the final user turn
is uncached. Refusals and API errors are recorded per example rather than raised.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from eval_harness.providers.base import CompletionRequest, CompletionResult, Provider
from eval_harness.schema import Usage

log = logging.getLogger(__name__)

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


class AnthropicProvider(Provider):
    def __init__(
        self,
        model: str,
        *,
        structured_output: bool,
        effort: str | None = None,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        pricing: dict[str, float] | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.structured_output = structured_output
        self.effort = effort
        self._pricing = pricing or DEFAULT_PRICING.get(model, {})
        self._client = client or anthropic.AsyncAnthropic(
            max_retries=max_retries, timeout=timeout_s
        )

    @property
    def id(self) -> str:
        return (
            f"anthropic:{self.model}:structured={int(self.structured_output)}"
            f":effort={self.effort or 'default'}"
        )

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": request.prompt.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages: list[dict[str, Any]] = []
        for user_text, assistant_json in request.prompt.few_shot:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_json})
        if messages:
            last = messages[-1]
            last["content"] = [
                {"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}
            ]
        messages.append({"role": "user", "content": request.prompt.user})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if self.structured_output and request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        kwargs = self._build_kwargs(request)
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            return CompletionResult(text="", error=f"rate_limited: {e.message}")
        except anthropic.APIStatusError as e:
            return CompletionResult(text="", error=f"api_status_{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return CompletionResult(text="", error=f"connection: {e}")

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        if response.stop_reason == "refusal":
            return CompletionResult(text="", usage=usage, stop_reason="refusal", error="refusal")

        text = "".join(b.text for b in response.content if b.type == "text")
        return CompletionResult(text=text, usage=usage, stop_reason=response.stop_reason)

    def cost_usd(self, usage: Usage) -> float:
        p = self._pricing
        if not p:
            return 0.0
        return (
            usage.input_tokens * p.get("input", 0.0)
            + usage.output_tokens * p.get("output", 0.0)
            + usage.cache_read_input_tokens * p.get("cache_read", 0.0)
            + usage.cache_creation_input_tokens * p.get("cache_write", 0.0)
        ) / 1_000_000
