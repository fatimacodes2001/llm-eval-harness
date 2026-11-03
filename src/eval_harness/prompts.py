"""Prompt construction. Prompts are versioned and hashed into the run manifest.

Few-shot examples come from the ``dev`` split only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eval_harness.schema import GoldenExample, HazardReport, HazardType, PropertyInput

PROMPT_VERSION = "hazard-extract/v2"

_VOCAB = "\n".join(f"- {h.value}" for h in HazardType)

SYSTEM_PROMPT = f"""You are an underwriting assistant. You receive computer-vision detections for a
single property and must produce a structured hazard report.

Rules:
1. Only flag hazards from this closed vocabulary:
{_VOCAB}
2. Every hazard MUST cite one or more detection ids from the input as evidence.
   Never invent ids. If no detection supports a hazard, do not flag it.
3. Decision rules:
   - overhanging_vegetation: a tree_canopy detection with overhang_roof_pct >= 0.20
   - unfenced_pool: a pool detection with fenced == false
   - roof_damage: a roof detection with condition in ("poor", "severe")
   - roof_tarp: a tarp detection
   - yard_debris: a debris detection with area_m2 >= 10
   - trampoline: a trampoline detection
   - dead_vegetation: a vegetation detection with health == "dead"
   - water_pooling: a standing_water detection
4. risk_score: 1 if no hazards; otherwise 1 + number of hazards, capped at 5,
   and always 5 if roof_damage is present.
5. summary: one or two sentences. Do not mention numbers that are not in the input.

Prompt version: {PROMPT_VERSION}
"""


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    few_shot: list[tuple[str, str]]
    user: str

    def hash(self) -> str:
        payload = json.dumps(
            {"system": self.system, "few_shot": self.few_shot}, sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def render_input(prop: PropertyInput) -> str:
    return "Property detections:\n" + json.dumps(
        prop.model_dump(by_alias=True), indent=2, sort_keys=True
    )


def render_expected_as_report(example: GoldenExample) -> str:
    """Turn a golden ``expected`` into a model-shaped answer for few-shot use."""
    report = HazardReport(
        hazards=[
            {
                "type": h.type,
                "evidence_ids": h.evidence_ids,
                "rationale": (
                    f"{h.type.value.replace('_', ' ')} supported by {', '.join(h.evidence_ids)}"
                ),
            }
            for h in example.expected.hazards
        ],
        risk_score=example.expected.risk_score,
        summary=(
            "No hazards detected."
            if not example.expected.hazards
            else f"{len(example.expected.hazards)} hazard(s) identified from the detections."
        ),
    )
    return report.model_dump_json()


def build_prompt(
    example: GoldenExample,
    few_shot_pool: list[GoldenExample],
    k: int,
    *,
    ask_for_json_in_text: bool,
) -> RenderedPrompt:
    """Build the prompt for one example.

    With ``ask_for_json_in_text`` the schema is pasted into the system prompt and the
    response is parsed from text. Without it the caller enforces the schema via the API.
    """
    system = SYSTEM_PROMPT
    if ask_for_json_in_text:
        system += (
            "\nRespond with ONLY a JSON object matching this schema, no prose:\n"
            + json.dumps(HazardReport.json_schema_for_api(), sort_keys=True)
        )

    shots = [e for e in few_shot_pool if e.split == "dev" and e.id != example.id][:k]
    few_shot = [(render_input(e.input), render_expected_as_report(e)) for e in shots]
    return RenderedPrompt(system=system, few_shot=few_shot, user=render_input(example.input))
