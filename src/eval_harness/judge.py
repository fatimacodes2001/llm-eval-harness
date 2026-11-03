"""LLM-as-judge for the free-text ``summary`` field.

Deterministic checks catch invented ids and hazard types. They cannot tell whether
the prose summary is faithful to the input, so a second model answers a yes/no
question with a fixed rubric. The judge goes through the same ``Provider``
interface, so it can be mocked in CI.
"""

from __future__ import annotations

import json

from eval_harness.prompts import RenderedPrompt
from eval_harness.providers.base import CompletionRequest, Provider
from eval_harness.schema import GoldenExample, Prediction

JUDGE_SYSTEM = """You are a strict factual-consistency checker.
Given SOURCE (structured detections) and CLAIM (a short summary), decide whether every
statement in CLAIM is supported by SOURCE. Numbers, counts and hazard names that do not
appear in or follow from SOURCE make the claim unsupported.
Answer with JSON: {"supported": true|false, "reason": "<one sentence>"}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
    "additionalProperties": False,
}


async def judge_summary(
    judge: Provider, golden: GoldenExample, pred: Prediction
) -> tuple[bool | None, str]:
    if pred.parsed is None:
        return None, "no parsed output"
    user = (
        "SOURCE:\n"
        + json.dumps(golden.input.model_dump(by_alias=True), sort_keys=True)
        + "\n\nCLAIM:\n"
        + pred.parsed.summary
    )
    prompt = RenderedPrompt(system=JUDGE_SYSTEM, few_shot=[], user=user)
    result = await judge.complete(
        CompletionRequest(prompt=prompt, json_schema=JUDGE_SCHEMA, max_tokens=256)
    )
    if result.error:
        return None, result.error
    try:
        data = json.loads(result.text)
        return bool(data["supported"]), str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"judge output unparseable: {e}"
