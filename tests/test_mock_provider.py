import json

import pytest

from eval_harness.prompts import build_prompt
from eval_harness.providers import CompletionRequest
from eval_harness.providers.mock_provider import MockProvider, oracle_hazards, oracle_risk_score
from eval_harness.schema import GoldenExample, HazardReport


def test_oracle_matches_golden(golden: GoldenExample):
    dets = [d.model_dump(by_alias=True) for d in golden.input.detections]
    hazards = oracle_hazards(dets)
    assert {h["type"] for h in hazards} == {"roof_damage", "unfenced_pool"}
    assert oracle_risk_score(hazards) == 5
    assert oracle_risk_score([]) == 1


async def test_clean_mock_is_exact_and_deterministic(golden: GoldenExample):
    prompt = build_prompt(golden, [], 0, ask_for_json_in_text=True)
    req = CompletionRequest(prompt=prompt, json_schema=None)
    p = MockProvider()
    r1, r2 = await p.complete(req), await p.complete(req)
    assert r1.text == r2.text
    report = HazardReport.model_validate_json(r1.text)
    assert {h.type.value for h in report.hazards} == {"roof_damage", "unfenced_pool"}


@pytest.mark.parametrize("structured", [False, True])
async def test_malformed_only_without_schema(golden: GoldenExample, structured: bool):
    prompt = build_prompt(golden, [], 0, ask_for_json_in_text=not structured)
    schema = HazardReport.json_schema_for_api() if structured else None
    p = MockProvider(malformed_rate=1.0)
    text = (await p.complete(CompletionRequest(prompt=prompt, json_schema=schema))).text
    is_valid_json = True
    try:
        json.loads(text)
    except json.JSONDecodeError:
        is_valid_json = False
    assert is_valid_json == structured or not structured


def test_cache_key_changes_with_provider_and_prompt(golden: GoldenExample):
    p1 = build_prompt(golden, [], 0, ask_for_json_in_text=True)
    p2 = build_prompt(golden, [], 0, ask_for_json_in_text=False)
    k1 = CompletionRequest(prompt=p1, json_schema=None).cache_key("a")
    assert k1 != CompletionRequest(prompt=p1, json_schema=None).cache_key("b")
    assert k1 != CompletionRequest(prompt=p2, json_schema=None).cache_key("a")
    assert k1 == CompletionRequest(prompt=p1, json_schema=None).cache_key("a")
