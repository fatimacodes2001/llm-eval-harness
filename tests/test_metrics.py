import pytest

from eval_harness.metrics import (
    PRF,
    aggregate,
    hallucinated_flags,
    paired_f1_delta,
    score_example,
)
from eval_harness.schema import GoldenExample, HazardReport, Prediction


def _pred(golden: GoldenExample, report: dict | None, error: str | None = None) -> Prediction:
    return Prediction(
        example_id=golden.id,
        raw_text="" if report is None else "{}",
        parsed=None if report is None else HazardReport.model_validate(report),
        parse_error=error,
        latency_s=0.5,
    )


def test_prf_edge_cases():
    assert PRF().f1 == 0.0
    assert PRF(tp=2, fp=0, fn=0).f1 == 1.0
    assert PRF(tp=1, fp=1, fn=1).f1 == pytest.approx(0.5)


def test_perfect_prediction(golden: GoldenExample):
    pred = _pred(
        golden,
        {
            "hazards": [
                {"type": "roof_damage", "evidence_ids": ["d1"], "rationale": "r"},
                {"type": "unfenced_pool", "evidence_ids": ["d2"], "rationale": "r"},
            ],
            "risk_score": 5,
            "summary": "s",
        },
    )
    s = score_example(golden, pred, cost_usd=0.01)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)
    assert s.exact_match and s.schema_ok
    assert s.hallucinated_flags == 0
    assert s.score_abs_err == 0


def test_hallucination_unknown_id_and_unsupporting_detection(golden: GoldenExample):
    pred = _pred(
        golden,
        {
            "hazards": [
                {"type": "roof_damage", "evidence_ids": ["d1"], "rationale": "ok"},
                {"type": "trampoline", "evidence_ids": ["d42"], "rationale": "invented id"},
                {
                    "type": "yard_debris",
                    "evidence_ids": ["d3"],
                    "rationale": "canopy is not debris",
                },
            ],
            "risk_score": 4,
            "summary": "s",
        },
    )
    assert hallucinated_flags(golden, pred) == 2
    s = score_example(golden, pred, 0.0)
    assert (s.tp, s.fp, s.fn) == (1, 2, 1)
    assert s.score_abs_err == 1


def test_unparsed_counts_as_all_missed(golden: GoldenExample):
    s = score_example(golden, _pred(golden, None, "json: boom"), 0.0)
    assert not s.schema_ok
    assert (s.tp, s.fp, s.fn) == (0, 0, 2)
    assert s.score_abs_err is None


def test_aggregate_and_paired_delta(golden: GoldenExample):
    good = score_example(
        golden,
        _pred(
            golden,
            {
                "hazards": [
                    {"type": "roof_damage", "evidence_ids": ["d1"], "rationale": "r"},
                    {"type": "unfenced_pool", "evidence_ids": ["d2"], "rationale": "r"},
                ],
                "risk_score": 5,
                "summary": "s",
            },
        ),
        0.0,
    )
    bad = score_example(golden, _pred(golden, None, "json: boom"), 0.0)

    agg_bad = aggregate([bad] * 5, n_boot=50)
    agg_good = aggregate([good] * 5, n_boot=50)
    assert agg_bad["schema_adherence"] == 0.0
    assert agg_good["schema_adherence"] == 1.0
    assert agg_good["hazard_micro"]["f1"] == 1.0
    assert agg_good["hazard_f1_ci95"] == [1.0, 1.0]

    before = [score_example(golden, _pred(golden, None, "e"), 0.0) for _ in range(5)]
    after = [good] * 5
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        a.example_id = b.example_id = f"p{i}"
    delta = paired_f1_delta(before, after, n_boot=100)
    assert delta["f1_delta"] == 1.0
    assert delta["p_improved"] == 1.0


def test_paired_delta_requires_overlap(golden: GoldenExample):
    a = score_example(golden, _pred(golden, None, "e"), 0.0)
    b = score_example(golden, _pred(golden, None, "e"), 0.0)
    b.example_id = "other"
    with pytest.raises(ValueError, match="no overlapping"):
        paired_f1_delta([a], [b])
