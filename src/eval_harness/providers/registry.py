from __future__ import annotations

from typing import Any

from eval_harness.providers.base import Provider


def build_provider(cfg: dict[str, Any]) -> Provider:
    """Instantiate a provider from the ``provider`` block of a run config."""
    kind = cfg.get("kind")
    params = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "anthropic":
        from eval_harness.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**params)
    if kind == "mock":
        from eval_harness.providers.mock_provider import MockProvider

        return MockProvider(**params)
    raise ValueError(f"unknown provider kind: {kind!r}")
