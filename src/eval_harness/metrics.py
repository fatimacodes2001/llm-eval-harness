"""Metrics over a list of (golden, prediction) pairs.

Everything here is pure and deterministic so it can be unit-tested without a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval_harness.schema import GoldenExample, HazardType, Prediction

SUPPORTS: dict[HazardType, set[str]] = {
    HazardType.OVERHANGING_VEGETATION: {"tree_canopy"},
    HazardType.UNFENCED_POOL: {"pool"},
    HazardType.ROOF_DAMAGE: {"roof"},
    HazardType.ROOF_TARP: {"tarp"},
    HazardType.YARD_DEBRIS: {"debris"},
    HazardType.TRAMPOLINE: {"trampoline"},
    HazardType.DEAD_VEGETATION: {"vegetation"},
    HazardType.WATER_POOLING: {"standing_water"},
}


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ExampleScore:
    example_id: str
    schema_ok: bool
    provider_error: bool
    tp: int
    fp: int
    fn: int
    hallucinated_flags: int
    predicted_flags: int
    exact_match: bool
    score_abs_err: float | None
    latency_s: float
    cost_usd: float
    tags: list[str] = field(default_factory=list)

    @property
    def f1(self) -> float:
        return PRF(self.tp, self.fp, self.fn).f1


def hallucinated_flags(golden: GoldenExample, pred: Prediction) -> int:
    """Flags citing unknown detection ids or detections that cannot support that hazard."""
    if pred.parsed is None:
        return 0
    by_id = {d.id: d for d in golden.input.detections}
    n = 0
    for flag in pred.parsed.hazards:
        ok = all(
            (eid in by_id) and (by_id[eid].cls in SUPPORTS[flag.type]) for eid in flag.evidence_ids
        )
        n += 0 if ok else 1
    return n


def score_example(golden: GoldenExample, pred: Prediction, cost_usd: float) -> ExampleScore:
    expected = {h.type for h in golden.expected.hazards}
    if pred.parsed is None:
        return ExampleScore(
            example_id=golden.id,
            schema_ok=False,
            provider_error=pred.provider_error is not None,
            tp=0,
            fp=0,
            fn=len(expected),
            hallucinated_flags=0,
            predicted_flags=0,
            exact_match=False,
            score_abs_err=None,
            latency_s=pred.latency_s,
            cost_usd=cost_usd,
            tags=[golden.difficulty, golden.split],
        )
    predicted = {h.type for h in pred.parsed.hazards}
    return ExampleScore(
        example_id=golden.id,
        schema_ok=True,
        provider_error=False,
        tp=len(expected & predicted),
        fp=len(predicted - expected),
        fn=len(expected - predicted),
        hallucinated_flags=hallucinated_flags(golden, pred),
        predicted_flags=len(predicted),
        exact_match=expected == predicted,
        score_abs_err=abs(pred.parsed.risk_score - golden.expected.risk_score),
        latency_s=pred.latency_s,
        cost_usd=cost_usd,
        tags=[golden.difficulty, golden.split],
    )


def _bootstrap_ci(
    values: np.ndarray, stat: Any, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = np.array([stat(values[i]) for i in idx])
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def _micro_f1(rows: np.ndarray) -> float:
    tp, fp, fn = rows[:, 0].sum(), rows[:, 1].sum(), rows[:, 2].sum()
    return PRF(int(tp), int(fp), int(fn)).f1


def aggregate(scores: list[ExampleScore], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {"n": 0}

    micro = PRF(sum(s.tp for s in scores), sum(s.fp for s in scores), sum(s.fn for s in scores))
    rows = np.array([[s.tp, s.fp, s.fn] for s in scores], dtype=float)
    f1_lo, f1_hi = _bootstrap_ci(rows, _micro_f1, n_boot, seed)

    predicted_total = sum(s.predicted_flags for s in scores)
    halluc_total = sum(s.hallucinated_flags for s in scores)
    halluc_rows = np.array([[s.hallucinated_flags, s.predicted_flags] for s in scores], dtype=float)

    def _halluc_rate(r: np.ndarray) -> float:
        return float(r[:, 0].sum() / r[:, 1].sum()) if r[:, 1].sum() else 0.0

    h_lo, h_hi = _bootstrap_ci(halluc_rows, _halluc_rate, n_boot, seed)

    errs = [s.score_abs_err for s in scores if s.score_abs_err is not None]
    schema_ok = sum(s.schema_ok for s in scores)

    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({t for s in scores for t in s.tags}):
        sub = [s for s in scores if tag in s.tags]
        prf = PRF(sum(s.tp for s in sub), sum(s.fp for s in sub), sum(s.fn for s in sub))
        by_tag[tag] = {"n": len(sub), **prf.as_dict()}

    return {
        "n": n,
        "schema_adherence": round(schema_ok / n, 4),
        "provider_error_rate": round(sum(s.provider_error for s in scores) / n, 4),
        "hazard_micro": micro.as_dict(),
        "hazard_f1_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        "exact_match_rate": round(sum(s.exact_match for s in scores) / n, 4),
        "hallucination_rate": round(halluc_total / predicted_total, 4) if predicted_total else 0.0,
        "hallucination_rate_ci95": [round(h_lo, 4), round(h_hi, 4)],
        "hallucinated_flags": halluc_total,
        "predicted_flags": predicted_total,
        "risk_score_mae": round(float(np.mean(errs)), 4) if errs else None,
        "risk_score_within_1": round(float(np.mean([e <= 1 for e in errs])), 4) if errs else None,
        "latency_p50_s": round(float(np.median([s.latency_s for s in scores])), 3),
        "latency_p95_s": round(float(np.quantile([s.latency_s for s in scores], 0.95)), 3),
        "cost_usd_total": round(sum(s.cost_usd for s in scores), 4),
        "by_tag": by_tag,
    }


def paired_f1_delta(
    a: list[ExampleScore], b: list[ExampleScore], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap of micro-F1(b) - micro-F1(a) over shared example ids."""
    a_by = {s.example_id: s for s in a}
    b_by = {s.example_id: s for s in b}
    ids = sorted(set(a_by) & set(b_by))
    if not ids:
        raise ValueError("no overlapping example ids between runs")
    ra = np.array([[a_by[i].tp, a_by[i].fp, a_by[i].fn] for i in ids], dtype=float)
    rb = np.array([[b_by[i].tp, b_by[i].fp, b_by[i].fn] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        deltas.append(_micro_f1(rb[idx]) - _micro_f1(ra[idx]))
    d = np.array(deltas)
    lo, hi = np.quantile(d, [0.025, 0.975])
    point = _micro_f1(rb) - _micro_f1(ra)
    return {
        "n_paired": len(ids),
        "f1_delta": round(float(point), 4),
        "f1_delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_improved": round(float((d > 0).mean()), 4),
    }


def confusion_by_type(
    pairs: list[tuple[GoldenExample, Prediction]],
) -> dict[str, dict[str, int]]:
    c: dict[str, Counter[str]] = {t.value: Counter() for t in HazardType}
    for g, p in pairs:
        exp = {h.type for h in g.expected.hazards}
        got = {h.type for h in p.parsed.hazards} if p.parsed else set()
        for t in HazardType:
            if t in exp and t in got:
                c[t.value]["tp"] += 1
            elif t in got:
                c[t.value]["fp"] += 1
            elif t in exp:
                c[t.value]["fn"] += 1
    return {k: dict(v) for k, v in c.items()}
