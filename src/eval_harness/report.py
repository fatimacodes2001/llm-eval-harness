from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _row(label: str, value: Any) -> str:
    return f"| {label} | {value} |"


def write_markdown_report(
    path: Path, manifest: dict[str, Any], metrics: dict[str, Any], failures: list[dict[str, Any]]
) -> None:
    m = metrics
    micro = m["hazard_micro"]
    lines = [
        f"# Eval report: `{manifest['run_name']}`",
        "",
        f"- provider: `{manifest['provider_id']}`",
        f"- prompt: `{manifest['prompt_version']}` (hash `{manifest['prompt_hash']}`)",
        f"- config hash: `{manifest['config_hash']}`  git: `{manifest.get('git_sha')}`",
        f"- examples: {m['n']}  cache hit rate: {m['cache_hit_rate']}",
        "",
        "| metric | value |",
        "|---|---|",
        _row("schema adherence", m["schema_adherence"]),
        _row("hazard F1 (micro)", f"{micro['f1']}  CI95 {m['hazard_f1_ci95']}"),
        _row("hazard precision / recall", f"{micro['precision']} / {micro['recall']}"),
        _row("exact match rate", m["exact_match_rate"]),
        _row(
            "hallucination rate",
            f"{m['hallucination_rate']}  CI95 {m['hallucination_rate_ci95']}"
            f"  ({m['hallucinated_flags']}/{m['predicted_flags']} flags)",
        ),
        _row("risk score MAE", m["risk_score_mae"]),
        _row("risk score within ±1", m["risk_score_within_1"]),
        _row("latency p50 / p95 (s)", f"{m['latency_p50_s']} / {m['latency_p95_s']}"),
        _row("cost (USD)", m["cost_usd_total"]),
    ]
    if "judge" in m:
        lines.append(_row("summary supported (judge)", m["judge"]["summary_supported_rate"]))

    lines += ["", "## By hazard type", "", "| type | tp | fp | fn |", "|---|---|---|---|"]
    for t, c in m["by_hazard_type"].items():
        lines.append(f"| {t} | {c.get('tp', 0)} | {c.get('fp', 0)} | {c.get('fn', 0)} |")

    lines += [
        "",
        "## By slice",
        "",
        "| slice | n | precision | recall | f1 |",
        "|---|---|---|---|---|",
    ]
    for tag, s in m["by_tag"].items():
        lines.append(f"| {tag} | {s['n']} | {s['precision']} | {s['recall']} | {s['f1']} |")

    lines += ["", f"## Failures ({len(failures)})", ""]
    for f in failures[:20]:
        lines.append(
            f"- `{f['example_id']}` expected {f['expected']} got {f['predicted']}"
            + (f" — {f['parse_error']}" if f["parse_error"] else "")
        )
    if len(failures) > 20:
        lines.append(f"- … {len(failures) - 20} more in failures.json")
    path.write_text("\n".join(lines) + "\n")


def write_comparison(
    path: Path,
    a_name: str,
    b_name: str,
    a: dict[str, Any],
    b: dict[str, Any],
    delta: dict[str, Any],
) -> str:
    keys = [
        ("schema_adherence", "schema adherence"),
        ("exact_match_rate", "exact match rate"),
        ("hallucination_rate", "hallucination rate"),
        ("risk_score_mae", "risk score MAE"),
        ("latency_p50_s", "latency p50 (s)"),
        ("cost_usd_total", "cost (USD)"),
    ]
    lines = [
        f"# {a_name} → {b_name}",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| hazard F1 (micro) | {a['hazard_micro']['f1']} | {b['hazard_micro']['f1']} |",
    ]
    for k, label in keys:
        lines.append(f"| {label} | {a.get(k)} | {b.get(k)} |")
    lines += [
        "",
        f"Paired F1 delta: **{delta['f1_delta']:+.4f}** "
        f"(CI95 {delta['f1_delta_ci95']}, P(improved) = {delta['p_improved']}, "
        f"n = {delta['n_paired']})",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    path.with_suffix(".json").write_text(json.dumps({"a": a, "b": b, "delta": delta}, indent=2))
    return text
