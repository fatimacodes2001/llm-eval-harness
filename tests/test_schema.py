import pytest
from pydantic import ValidationError

from eval_harness.schema import GoldenExample, HazardReport


def test_api_schema_is_strict():
    schema = HazardReport.json_schema_for_api()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"hazards", "risk_score", "summary"}
    flag = schema["$defs"]["HazardFlag"]
    assert flag["additionalProperties"] is False
    assert set(flag["required"]) == {"type", "evidence_ids", "rationale"}


def test_duplicate_hazard_types_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        HazardReport.model_validate(
            {
                "hazards": [
                    {"type": "trampoline", "evidence_ids": ["d1"], "rationale": "a"},
                    {"type": "trampoline", "evidence_ids": ["d1"], "rationale": "b"},
                ],
                "risk_score": 2,
                "summary": "x",
            }
        )


def test_golden_rejects_unknown_evidence(golden: GoldenExample):
    data = golden.model_dump(mode="json", by_alias=True)
    data["expected"]["hazards"][0]["evidence_ids"] = ["d99"]
    with pytest.raises(ValidationError, match="unknown detections"):
        GoldenExample.model_validate(data)


def test_golden_rejects_extra_fields(golden: GoldenExample):
    data = golden.model_dump(mode="json", by_alias=True)
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        GoldenExample.model_validate(data)
