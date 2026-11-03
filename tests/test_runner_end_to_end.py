import importlib.util
import json
import random
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval_harness.cli import app
from eval_harness.config import RunConfig
from eval_harness.data import load_golden, write_golden
from eval_harness.runner import run

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_golden.py"


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location("gen", SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gen)
    rows = [gen.make_example(random.Random(3), i) for i in range(25)]
    out = tmp_path / "golden.jsonl"
    write_golden(out, rows)
    assert len(load_golden(out)) == 25
    return out


def _cfg(tmp_path: Path, golden_file: Path, name: str, **provider) -> RunConfig:
    return RunConfig(
        name=name,
        golden_path=golden_file,
        provider={"kind": "mock", **provider},
        few_shot_k=2,
        structured_output=provider.get("malformed_rate", 0) == 0,
        cache_dir=tmp_path / ".cache",
        reports_dir=tmp_path / "reports",
        bootstrap_samples=50,
    )


async def test_run_writes_artifacts_and_caches(tmp_path: Path, golden_file: Path):
    cfg = _cfg(tmp_path, golden_file, "clean")
    first = await run(cfg)
    out = Path(first["out_dir"])
    for f in ("manifest.json", "metrics.json", "predictions.jsonl", "scores.jsonl", "report.md"):
        assert (out / f).exists()
    assert first["metrics"]["hazard_micro"]["f1"] == 1.0
    assert first["metrics"]["schema_adherence"] == 1.0
    assert first["metrics"]["cache_hit_rate"] == 0.0

    second = await run(cfg)
    assert second["metrics"]["cache_hit_rate"] == 1.0
    assert second["metrics"]["hazard_micro"] == first["metrics"]["hazard_micro"]


async def test_dev_split_never_evaluated_as_test(tmp_path: Path, golden_file: Path):
    cfg = _cfg(tmp_path, golden_file, "clean")
    res = await run(cfg)
    golden = {g.id: g for g in load_golden(golden_file)}
    with (Path(res["out_dir"]) / "predictions.jsonl").open() as f:
        ids = [json.loads(line)["example_id"] for line in f]
    assert ids and all(golden[i].split == "test" for i in ids)


def test_cli_run_and_compare_gate(tmp_path: Path, golden_file: Path):
    runner = CliRunner()
    configs = {}
    for name, noise in {
        "before": {"miss_rate": 0.5, "malformed_rate": 0.3, "hallucination_rate": 0.3},
        "after": {},
    }.items():
        cfg = _cfg(tmp_path, golden_file, name, **noise)
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
        configs[name] = p
        assert runner.invoke(app, ["run", str(p)]).exit_code == 0, name

    reports = tmp_path / "reports"
    ok = runner.invoke(app, ["compare", str(reports / "before"), str(reports / "after")])
    assert ok.exit_code == 0, ok.output
    assert "PASS" in ok.output

    regress = runner.invoke(
        app, ["compare", str(reports / "after"), str(reports / "before"), "--min-f1-delta", "0"]
    )
    assert regress.exit_code == 1
    assert "FAIL" in regress.output


def test_cli_validate_data(golden_file: Path):
    res = CliRunner().invoke(app, ["validate-data", str(golden_file)])
    assert res.exit_code == 0 and "25 examples" in res.output
