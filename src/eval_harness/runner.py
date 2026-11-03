"""Run a config against the golden set and write predictions, metrics and a manifest."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.progress import Progress

from eval_harness import __version__
from eval_harness.config import RunConfig
from eval_harness.data import load_golden
from eval_harness.metrics import ExampleScore, aggregate, confusion_by_type, score_example
from eval_harness.parsing import parse_report
from eval_harness.prompts import PROMPT_VERSION, build_prompt
from eval_harness.providers import CompletionRequest, CompletionResult, Provider, build_provider
from eval_harness.report import write_markdown_report
from eval_harness.schema import GoldenExample, HazardReport, Prediction

log = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, root: Path, enabled: bool) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> CompletionResult | None:
        if not self.enabled:
            return None
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        return CompletionResult.from_json(json.loads(p.read_text()))

    def put(self, key: str, result: CompletionResult) -> None:
        if self.enabled and result.error is None:
            (self.root / f"{key}.json").write_text(json.dumps(result.to_json(), sort_keys=True))


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def _predict_one(
    provider: Provider,
    cache: ResponseCache,
    sem: asyncio.Semaphore,
    cfg: RunConfig,
    example: GoldenExample,
    pool: list[GoldenExample],
    schema: dict[str, Any] | None,
) -> Prediction:
    prompt = build_prompt(
        example, pool, cfg.few_shot_k, ask_for_json_in_text=not cfg.structured_output
    )
    request = CompletionRequest(prompt=prompt, json_schema=schema, max_tokens=cfg.max_tokens)
    key = request.cache_key(provider.id)

    cached = cache.get(key)
    if cached is not None:
        result, latency, was_cached = cached, 0.0, True
    else:
        async with sem:
            t0 = time.perf_counter()
            result = await provider.complete(request)
            latency = time.perf_counter() - t0
        cache.put(key, result)
        was_cached = False

    parsed, parse_error = (None, result.error) if result.error else parse_report(result.text)
    return Prediction(
        example_id=example.id,
        raw_text=result.text,
        parsed=parsed,
        parse_error=parse_error,
        usage=result.usage,
        latency_s=latency,
        cached=was_cached,
        provider_error=result.error,
    )


async def run(cfg: RunConfig, *, prompt_hash_only: bool = False) -> dict[str, Any]:
    golden = load_golden(cfg.golden_path)
    pool = [g for g in golden if g.split == "dev"]
    targets = [g for g in golden if g.split == cfg.split]
    if cfg.limit:
        targets = targets[: cfg.limit]
    if not targets:
        raise ValueError(f"no examples in split {cfg.split!r}")

    provider = build_provider(cfg.provider)
    cache = ResponseCache(cfg.cache_dir, cfg.use_cache)
    schema = HazardReport.json_schema_for_api() if cfg.structured_output else None
    sem = asyncio.Semaphore(cfg.concurrency)

    sample_prompt = build_prompt(
        targets[0], pool, cfg.few_shot_k, ask_for_json_in_text=not cfg.structured_output
    )
    manifest: dict[str, Any] = {
        "run_name": cfg.name,
        "harness_version": __version__,
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "config_hash": cfg.hash(),
        "config": json.loads(cfg.model_dump_json()),
        "provider_id": provider.id,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": sample_prompt.hash(),
        "golden_path": str(cfg.golden_path),
        "n_examples": len(targets),
        "few_shot_pool_size": len(pool),
    }
    if prompt_hash_only:
        return manifest

    tasks = [_predict_one(provider, cache, sem, cfg, g, pool, schema) for g in targets]
    preds: list[Prediction] = []
    with Progress(transient=True) as progress:
        bar = progress.add_task(f"[cyan]{cfg.name}", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            preds.append(await coro)
            progress.advance(bar)

    by_id = {p.example_id: p for p in preds}
    pairs = [(g, by_id[g.id]) for g in targets]
    scores: list[ExampleScore] = [score_example(g, p, provider.cost_usd(p.usage)) for g, p in pairs]

    judge_summary_stats: dict[str, Any] | None = None
    if cfg.judge:
        from eval_harness.judge import judge_summary

        judge_provider = build_provider(cfg.judge)
        verdicts = await asyncio.gather(*(judge_summary(judge_provider, g, p) for g, p in pairs))
        decided = [v for v, _ in verdicts if v is not None]
        judge_summary_stats = {
            "judge_provider_id": judge_provider.id,
            "n_judged": len(decided),
            "summary_supported_rate": round(sum(decided) / len(decided), 4) if decided else None,
        }

    metrics = aggregate(scores, n_boot=cfg.bootstrap_samples, seed=cfg.seed)
    metrics["by_hazard_type"] = confusion_by_type(pairs)
    metrics["cache_hit_rate"] = round(sum(p.cached for p in preds) / len(preds), 4)
    if judge_summary_stats:
        metrics["judge"] = judge_summary_stats
    manifest["finished_at"] = datetime.now(UTC).isoformat()

    out = cfg.reports_dir / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "predictions.jsonl").open("w") as f:
        for _, p in pairs:
            f.write(json.dumps(p.model_dump(mode="json"), sort_keys=True) + "\n")
    with (out / "scores.jsonl").open("w") as f:
        for s in scores:
            f.write(json.dumps(s.__dict__, sort_keys=True) + "\n")
    failures = [
        {
            "example_id": g.id,
            "parse_error": p.parse_error,
            "expected": [h.type.value for h in g.expected.hazards],
            "predicted": [h.type.value for h in p.parsed.hazards] if p.parsed else None,
            "raw_text": p.raw_text[:500],
        }
        for g, p in pairs
        if p.parsed is None
        or {h.type for h in g.expected.hazards} != {h.type for h in p.parsed.hazards}
    ]
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    write_markdown_report(out / "report.md", manifest, metrics, failures)
    log.info("wrote %s", out)
    return {"manifest": manifest, "metrics": metrics, "out_dir": str(out)}
