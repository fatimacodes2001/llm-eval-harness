"""Exercises request construction and response handling against a stubbed SDK client."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from eval_harness.prompts import build_prompt
from eval_harness.providers import CompletionRequest
from eval_harness.providers.anthropic_provider import AnthropicProvider
from eval_harness.schema import GoldenExample, HazardReport, Usage


class _StubMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _client(messages):
    return SimpleNamespace(messages=messages)


def _response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=0,
        ),
    )


@pytest.fixture
def request_(golden: GoldenExample) -> CompletionRequest:
    prompt = build_prompt(
        golden,
        [golden.model_copy(update={"split": "dev", "id": "x"})],
        1,
        ask_for_json_in_text=False,
    )
    return CompletionRequest(prompt=prompt, json_schema=HazardReport.json_schema_for_api())


def test_structured_request_shape(request_: CompletionRequest):
    stub = _StubMessages(_response("{}"))
    provider = AnthropicProvider(
        "claude-opus-5-5", structured_output=True, effort="low", client=_client(stub)
    )
    kwargs = provider._build_kwargs(request_)
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert kwargs["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "structured=1" in provider.id


def test_unstructured_request_has_no_format(request_: CompletionRequest):
    provider = AnthropicProvider(
        "claude-haiku-5-5", structured_output=False, client=_client(_StubMessages())
    )
    kwargs = provider._build_kwargs(request_)
    assert "format" not in kwargs.get("output_config", {})


async def test_complete_records_usage_and_cost(request_: CompletionRequest):
    stub = _StubMessages(_response('{"ok": 1}'))
    provider = AnthropicProvider("claude-opus-5-5", structured_output=True, client=_client(stub))
    result = await provider.complete(request_)
    assert result.text == '{"ok": 1}'
    assert result.usage.cache_read_input_tokens == 900
    cost = provider.cost_usd(result.usage)
    assert cost == pytest.approx((100 * 4 + 50 * 20 + 900 * 0.20) / 1_000_000)


async def test_refusal_is_recorded_not_raised(request_: CompletionRequest):
    stub = _StubMessages(_response("", stop_reason="refusal"))
    provider = AnthropicProvider("claude-opus-5-5", structured_output=True, client=_client(stub))
    result = await provider.complete(request_)
    assert result.error == "refusal" and result.text == ""


async def test_api_errors_are_recorded(request_: CompletionRequest):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIStatusError("boom", response=httpx.Response(500, request=req), body=None)
    provider = AnthropicProvider(
        "claude-opus-5-5", structured_output=True, client=_client(_StubMessages(error=err))
    )
    result = await provider.complete(request_)
    assert result.error is not None and result.error.startswith("api_status_500")


def test_unknown_model_costs_zero():
    provider = AnthropicProvider("some-model", structured_output=True, client=_client(None))
    assert provider.cost_usd(Usage(input_tokens=10)) == 0.0
