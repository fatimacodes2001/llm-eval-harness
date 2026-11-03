from __future__ import annotations

import pytest

from eval_harness.schema import GoldenExample


@pytest.fixture
def golden() -> GoldenExample:
    return GoldenExample.model_validate(
        {
            "id": "prop-0001",
            "split": "test",
            "difficulty": "medium",
            "input": {
                "property_id": "P1",
                "survey_date": "2026-03-01",
                "detections": [
                    {
                        "id": "d1",
                        "class": "roof",
                        "confidence": 0.9,
                        "attributes": {"condition": "poor"},
                    },
                    {
                        "id": "d2",
                        "class": "pool",
                        "confidence": 0.8,
                        "attributes": {"fenced": False},
                    },
                    {
                        "id": "d3",
                        "class": "tree_canopy",
                        "confidence": 0.7,
                        "attributes": {"overhang_roof_pct": 0.1},
                    },
                ],
            },
            "expected": {
                "hazards": [
                    {"type": "roof_damage", "evidence_ids": ["d1"]},
                    {"type": "unfenced_pool", "evidence_ids": ["d2"]},
                ],
                "risk_score": 5,
            },
            "annotation": {"source": "synthetic", "annotators": 1, "agreement": 1.0},
        }
    )
