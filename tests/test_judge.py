from eval_harness.judge import judge_summary
from eval_harness.providers.mock_provider import MockProvider
from eval_harness.schema import GoldenExample, HazardReport, Prediction


def _pred(golden: GoldenExample, summary: str) -> Prediction:
    report = HazardReport(hazards=[], risk_score=1, summary=summary)
    return Prediction(example_id=golden.id, raw_text=report.model_dump_json(), parsed=report)


async def test_mock_judge_accepts_consistent_summary(golden: GoldenExample):
    ok, _ = await judge_summary(MockProvider(), golden, _pred(golden, "2 hazard(s) identified."))
    assert ok is True


async def test_mock_judge_rejects_invented_count(golden: GoldenExample):
    ok, reason = await judge_summary(MockProvider(), golden, _pred(golden, "7 hazard(s) found."))
    assert ok is False
    assert "claimed 7" in reason


async def test_judge_skips_unparsed(golden: GoldenExample):
    pred = Prediction(example_id=golden.id, raw_text="", parsed=None, parse_error="x")
    assert await judge_summary(MockProvider(), golden, pred) == (None, "no parsed output")
