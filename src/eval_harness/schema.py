"""Data contracts: golden examples, the model output contract, and per-example predictions.

The JSON schema sent to the API is derived from ``HazardReport`` so the eval and the
output contract cannot drift apart.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HazardType(StrEnum):
    """Closed vocabulary. Anything outside it counts as a hallucination."""

    OVERHANGING_VEGETATION = "overhanging_vegetation"
    UNFENCED_POOL = "unfenced_pool"
    ROOF_DAMAGE = "roof_damage"
    ROOF_TARP = "roof_tarp"
    YARD_DEBRIS = "yard_debris"
    TRAMPOLINE = "trampoline"
    DEAD_VEGETATION = "dead_vegetation"
    WATER_POOLING = "water_pooling"


# ---------------------------------------------------------------------------
# Input side
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    """One computer-vision detection from the upstream CV service."""

    id: str
    cls: str = Field(alias="class", description="Detector class label")
    confidence: float = Field(ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PropertyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_id: str
    survey_date: str
    detections: list[Detection]

    def detection_ids(self) -> set[str]:
        return {d.id for d in self.detections}


# ---------------------------------------------------------------------------
# Output contract (what the model must return)
# ---------------------------------------------------------------------------


class HazardFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: HazardType
    evidence_ids: list[str] = Field(
        min_length=1,
        description="IDs of detections that justify this flag. Must exist in the input.",
    )
    rationale: str = Field(max_length=300)


class HazardReport(BaseModel):
    """The structured output contract."""

    model_config = ConfigDict(extra="forbid")

    hazards: list[HazardFlag]
    risk_score: int = Field(ge=1, le=5, description="1 = benign, 5 = severe")
    summary: str = Field(max_length=600)

    @field_validator("hazards")
    @classmethod
    def _unique_types(cls, v: list[HazardFlag]) -> list[HazardFlag]:
        types = [h.type for h in v]
        if len(types) != len(set(types)):
            raise ValueError("duplicate hazard types in report")
        return v

    @classmethod
    def json_schema_for_api(cls) -> dict[str, Any]:
        """Strict JSON schema for ``output_config.format`` or a tool ``input_schema``."""
        schema = cls.model_json_schema()
        return _strictify(schema)


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Set ``additionalProperties: false`` and mark every field required on all objects."""

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = sorted(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


class ExpectedHazard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: HazardType
    evidence_ids: list[str] = Field(min_length=1)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hazards: list[ExpectedHazard]
    risk_score: int = Field(ge=1, le=5)


class Annotation(BaseModel):
    """Label provenance, kept per example so slices can be filtered by label quality."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic", "human", "human_adjudicated"]
    annotators: int = Field(ge=1)
    agreement: float = Field(ge=0.0, le=1.0, description="Fraction of annotators agreeing")
    notes: str = ""


class GoldenExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    split: Literal["dev", "test"] = "test"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    input: PropertyInput
    expected: Expected
    annotation: Annotation

    @field_validator("expected")
    @classmethod
    def _evidence_exists(cls, v: Expected, info: Any) -> Expected:
        prop: PropertyInput | None = info.data.get("input")
        if prop is None:
            return v
        ids = prop.detection_ids()
        for h in v.hazards:
            missing = set(h.evidence_ids) - ids
            if missing:
                raise ValueError(f"expected hazard {h.type} cites unknown detections {missing}")
        return v


# ---------------------------------------------------------------------------
# Runner records
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class Prediction(BaseModel):
    example_id: str
    raw_text: str
    parsed: HazardReport | None
    parse_error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False
    provider_error: str | None = None
