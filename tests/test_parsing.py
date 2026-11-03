from eval_harness.parsing import parse_report

GOOD = '{"hazards": [], "risk_score": 1, "summary": "No hazards detected."}'


def test_plain_json():
    report, err = parse_report(GOOD)
    assert err is None and report is not None
    assert report.risk_score == 1


def test_fenced_json_with_prose():
    report, err = parse_report("Sure! Here you go:\n```json\n" + GOOD + "\n```\nHope that helps")
    assert err is None and report is not None


def test_truncated_json():
    report, err = parse_report(GOOD[:20])
    assert report is None and err is not None and err.startswith("json:")


def test_schema_violation():
    report, err = parse_report(GOOD.replace("risk_score", "risk"))
    assert report is None and err is not None and err.startswith("schema:")


def test_empty():
    assert parse_report("   ") == (None, "empty response")
