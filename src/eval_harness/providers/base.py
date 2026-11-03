"""Provider interface. The runner only talks to this."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from eval_harness.prompts import RenderedPrompt
from eval_harness.schema import Usage


@dataclass(frozen=True)
class CompletionRequest:
    prompt: RenderedPrompt
    json_schema: dict[str, Any] | None
    max_tokens: int = 2048

    def cache_key(self, provider_id: str) -> str:
        """Stable key for the on-disk response cache."""
        payload = {
            "provider": provider_id,
            "system": self.prompt.system,
            "few_shot": self.prompt.few_shot,
            "user": self.prompt.user,
            "schema": self.json_schema,
            "max_tokens": self.max_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass
class CompletionResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = self.usage.model_dump()
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CompletionResult:
        return cls(
            text=d["text"],
            usage=Usage(**d.get("usage", {})),
            stop_reason=d.get("stop_reason"),
            error=d.get("error"),
        )


class Provider(ABC):
    """A thing that turns a prompt into text."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Provider, model and settings. Used in the cache key and the manifest."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult: ...

    def cost_usd(self, usage: Usage) -> float:
        return 0.0
