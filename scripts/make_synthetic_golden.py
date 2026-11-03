"""Generate a synthetic golden set.

Labels are produced by the same decision rules the prompt states, then written with
``annotation.source = "synthetic"`` so nobody mistakes them for human labels.
See data/LABELING.md for how the real set was built.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from eval_harness.data import write_golden
from eval_harness.providers.mock_provider import oracle_hazards, oracle_risk_score
from eval_harness.schema import GoldenExample

CLASSES = [
    "roof",
    "tree_canopy",
    "pool",
    "tarp",
    "debris",
    "trampoline",
    "vegetation",
    "standing_water",
    "driveway",
    "solar_panel",
    "shed",
]


def _detection(rng: random.Random, idx: int, cls: str, hard: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if cls == "roof":
        attrs["condition"] = rng.choice(["good", "fair", "poor", "severe"])
        attrs["material"] = rng.choice(["tile", "metal", "shingle"])
    elif cls == "tree_canopy":
        attrs["overhang_roof_pct"] = round(
            rng.choice([0.18, 0.19, 0.20, 0.21]) if hard else rng.uniform(0.0, 0.6), 2
        )
    elif cls == "pool":
        attrs["fenced"] = rng.random() < 0.6
    elif cls == "debris":
        attrs["area_m2"] = round(
            rng.choice([8.0, 9.5, 10.0, 11.0]) if hard else rng.uniform(2, 40), 1
        )
    elif cls == "vegetation":
        attrs["health"] = rng.choice(["healthy", "stressed", "dead"])
    return {
        "id": f"d{idx}",
        "class": cls,
        "confidence": round(rng.uniform(0.55, 0.99), 2),
        "attributes": attrs,
    }


def make_example(rng: random.Random, i: int) -> GoldenExample:
    difficulty = rng.choices(["easy", "medium", "hard"], weights=[0.35, 0.4, 0.25])[0]
    n = {"easy": rng.randint(1, 3), "medium": rng.randint(3, 6), "hard": rng.randint(5, 9)}[
        difficulty
    ]
    classes = ["roof", *rng.sample(CLASSES, k=n - 1)]
    detections = [
        _detection(rng, j + 1, cls, hard=difficulty == "hard") for j, cls in enumerate(classes)
    ]
    hazards = oracle_hazards(detections)
    return GoldenExample.model_validate(
        {
            "id": f"prop-{i:04d}",
            "split": "dev" if i % 5 == 0 else "test",
            "difficulty": difficulty,
            "input": {
                "property_id": f"P{100000 + i}",
                "survey_date": f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}",
                "detections": detections,
            },
            "expected": {"hazards": hazards, "risk_score": oracle_risk_score(hazards)},
            "annotation": {
                "source": "synthetic",
                "annotators": 1,
                "agreement": 1.0,
                "notes": "rule-derived; replace with adjudicated human labels",
            },
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [make_example(rng, i) for i in range(args.n)]
    write_golden(args.out, rows)
    print(f"wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
