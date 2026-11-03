"""Deterministic mock provider for CI and metric tests.

Applies the task's decision rules, then injects seeded noise: missed hazards,
hallucinated hazards, wrong risk scores and malformed output.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from eval_harness.providers.base import CompletionRequest, CompletionResult, Provider
from eval_harness.schema import HazardType, Usage


def _rng_for(example_user_text: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{example_user_text}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def oracle_hazards(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decision rules from the system prompt. Also used by the data generator."""
    out: dict[str, list[str]] = {}

    def add(t: HazardType, det_id: str) -> None:
        out.setdefault(t.value, []).append(det_id)

    for d in detections:
        cls, a = d["class"], d.get("attributes", {})
        if cls == "tree_canopy" and a.get("overhang_roof_pct", 0) >= 0.20:
            add(HazardType.OVERHANGING_VEGETATION, d["id"])
        elif cls == "pool" and a.get("fenced") is False:
            add(HazardType.UNFENCED_POOL, d["id"])
        elif cls == "roof" and a.get("condition") in ("poor", "severe"):
            add(HazardType.ROOF_DAMAGE, d["id"])
        elif cls == "tarp":
            add(HazardType.ROOF_TARP, d["id"])
        elif cls == "debris" and a.get("area_m2", 0) >= 10:
            add(HazardType.YARD_DEBRIS, d["id"])
        elif cls == "trampoline":
            add(HazardType.TRAMPOLINE, d["id"])
        elif cls == "vegetation" and a.get("health") == "dead":
            add(HazardType.DEAD_VEGETATION, d["id"])
        elif cls == "standing_water":
            add(HazardType.WATER_POOLING, d["id"])
    return [{"type": t, "evidence_ids": ids} for t, ids in out.items()]


def oracle_risk_score(hazards: list[dict[str, Any]]) -> int:
    if not hazards:
        return 1
    if any(h["type"] == HazardType.ROOF_DAMAGE.value for h in hazards):
        return 5
    return min(5, 1 + len(hazards))


class MockProvider(Provider):
    def __init__(
        self,
        *,
        name: str = "mock",
        seed: int = 0,
        miss_rate: float = 0.0,
        hallucination_rate: float = 0.0,
        malformed_rate: float = 0.0,
        score_noise_rate: float = 0.0,
    ) -> None:
        self.name = name
        self.seed = seed
        self.miss_rate = miss_rate
        self.hallucination_rate = hallucination_rate
        self.malformed_rate = malformed_rate
        self.score_noise_rate = score_noise_rate

    @property
    def id(self) -> str:
        return (
            f"mock:{self.name}:seed={self.seed}:miss={self.miss_rate}"
            f":halluc={self.hallucination_rate}:malformed={self.malformed_rate}"
            f":score_noise={self.score_noise_rate}"
        )

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        if request.prompt.user.startswith("SOURCE:"):
            return self._judge(request)
        rng = _rng_for(request.prompt.user, self.seed)
        payload = json.loads(request.prompt.user.split("\n", 1)[1])
        detections: list[dict[str, Any]] = payload["detections"]
        known_ids = [d["id"] for d in detections]

        hazards = oracle_hazards(detections)

        hazards = [h for h in hazards if rng.random() >= self.miss_rate]

        if rng.random() < self.hallucination_rate:
            present = {h["type"] for h in hazards}
            candidates = [t.value for t in HazardType if t.value not in present]
            if candidates:
                fake_type = rng.choice(candidates)
                evidence = (
                    [f"d{rng.randint(900, 999)}"]
                    if rng.random() < 0.5 or not known_ids
                    else [rng.choice(known_ids)]
                )
                hazards.append({"type": fake_type, "evidence_ids": evidence})

        score = oracle_risk_score(hazards)
        if rng.random() < self.score_noise_rate:
            score = max(1, min(5, score + rng.choice([-1, 1])))

        report = {
            "hazards": [
                {**h, "rationale": f"{h['type']} supported by {', '.join(h['evidence_ids'])}"}
                for h in hazards
            ],
            "risk_score": score,
            "summary": "No hazards detected."
            if not hazards
            else f"{len(hazards)} hazard(s) identified from the detections.",
        }
        text = json.dumps(report)

        if request.json_schema is None and rng.random() < self.malformed_rate:
            variant = rng.choice(["prose", "truncated", "wrong_field"])
            if variant == "prose":
                text = "Sure! Here is the report:\n```json\n" + text + "\n```"
            elif variant == "truncated":
                text = text[: len(text) // 2]
            else:
                text = text.replace('"risk_score"', '"risk"')

        usage = Usage(input_tokens=len(request.prompt.user) // 4, output_tokens=len(text) // 4)
        return CompletionResult(text=text, usage=usage, stop_reason="end_turn")

    def _judge(self, request: CompletionRequest) -> CompletionResult:
        """Judge the summary the way the mock would: supported iff the hazard count matches."""
        source, claim = request.prompt.user.split("\n\nCLAIM:\n", 1)
        detections = json.loads(source.removeprefix("SOURCE:\n"))["detections"]
        n_true = len(oracle_hazards(detections))
        claimed = re.match(r"(\d+) hazard", claim)
        n_claimed = int(claimed.group(1)) if claimed else 0
        verdict = {
            "supported": n_claimed == n_true,
            "reason": f"claimed {n_claimed}, source {n_true}",
        }
        return CompletionResult(text=json.dumps(verdict), stop_reason="end_turn")
