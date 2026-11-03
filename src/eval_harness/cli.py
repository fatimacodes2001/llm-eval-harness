from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from eval_harness.config import load_config
from eval_harness.data import load_golden
from eval_harness.metrics import ExampleScore, paired_f1_delta
from eval_harness.report import write_comparison
from eval_harness.runner import run as run_eval

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

LIMIT = typer.Option(None, help="Only evaluate the first N examples")
NO_CACHE = typer.Option(False, help="Bypass the on-disk response cache")
DRY_RUN = typer.Option(False, help="Print the manifest and exit")
MIN_F1_DELTA = typer.Option(0.0, help="Fail (exit 1) if paired F1 delta is below this value")
MAX_HALLUCINATION = typer.Option(None, help="Fail if the after-run hallucination rate exceeds this")
OUT = typer.Option(None, help="Where to write the comparison markdown")
VERBOSE = typer.Option(False, "--verbose", "-v")


@app.callback()
def _setup(verbose: bool = VERBOSE) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def run(
    config: Path,
    limit: int | None = LIMIT,
    no_cache: bool = NO_CACHE,
    dry_run: bool = DRY_RUN,
) -> None:
    """Evaluate one config and write reports/<name>/."""
    cfg = load_config(config)
    if limit is not None:
        cfg.limit = limit
    if no_cache:
        cfg.use_cache = False
    if dry_run:
        console.print_json(json.dumps(asyncio.run(run_eval(cfg, prompt_hash_only=True))))
        return
    result = asyncio.run(run_eval(cfg))
    m = result["metrics"]
    console.print(
        f"[bold]{cfg.name}[/bold]  F1={m['hazard_micro']['f1']}  "
        f"schema={m['schema_adherence']}  halluc={m['hallucination_rate']}  "
        f"cost=${m['cost_usd_total']}  → {result['out_dir']}/report.md"
    )


def _load_scores(run_dir: Path) -> list[ExampleScore]:
    with (run_dir / "scores.jsonl").open() as f:
        return [ExampleScore(**json.loads(line)) for line in f if line.strip()]


@app.command()
def compare(
    before: Path,
    after: Path,
    min_f1_delta: float = MIN_F1_DELTA,
    max_hallucination: float | None = MAX_HALLUCINATION,
    out: Path | None = OUT,
) -> None:
    """Compare two report directories and act as a regression gate."""
    a = json.loads((before / "metrics.json").read_text())
    b = json.loads((after / "metrics.json").read_text())
    delta = paired_f1_delta(_load_scores(before), _load_scores(after))
    out = out or after / f"compare_vs_{before.name}.md"
    text = write_comparison(out, before.name, after.name, a, b, delta)
    console.print(text)

    failed = False
    if delta["f1_delta"] < min_f1_delta:
        console.print(f"[red]FAIL[/red] F1 delta {delta['f1_delta']} < {min_f1_delta}")
        failed = True
    if max_hallucination is not None and b["hallucination_rate"] > max_hallucination:
        console.print(
            f"[red]FAIL[/red] hallucination {b['hallucination_rate']} > {max_hallucination}"
        )
        failed = True
    if failed:
        raise typer.Exit(code=1)
    console.print("[green]PASS[/green]")


@app.command("validate-data")
def validate_data(path: Path) -> None:
    """Validate a golden JSONL file and print split/difficulty counts."""
    rows = load_golden(path)
    splits = {s: sum(r.split == s for r in rows) for s in ("dev", "test")}
    diff = {d: sum(r.difficulty == d for r in rows) for d in ("easy", "medium", "hard")}
    n_pos = sum(bool(r.expected.hazards) for r in rows)
    console.print(f"{len(rows)} examples  splits={splits}  difficulty={diff}  with_hazards={n_pos}")


if __name__ == "__main__":
    app()
