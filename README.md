# llm-eval-harness

A small, reproducible harness for evaluating an LLM on a structured-extraction task.

The task in the repo: take computer-vision detections for a property (roof, pool,
trees, debris, …) and produce a hazard report with typed flags, cited evidence, and a
risk score. The task is a stand-in. The parts worth copying are the ones around it:
a validated golden set with split and label provenance, a provider interface so the
same run works against a real model or a mock, deterministic metrics with bootstrap
confidence intervals, a regression gate, and run manifests that record exactly which
prompt, config and git commit produced a number.

This is a sample written for sharing. It is not the harness I use at work, which is
tied to that system's data and is more involved. The structure and the decisions are
the same.

## Quick start

Needs [pixi](https://pixi.sh). No API key needed for the offline demo.

```bash
pixi install
pixi run demo        # baseline run, candidate run, then compare
```

`demo` runs two mock providers over the synthetic golden set and writes
`reports/mock_baseline/`, `reports/mock_candidate/` and a comparison. Open
`reports/mock_candidate/compare_vs_mock_baseline.md`.

To run against Claude:

```bash
cp .env.example .env     # add ANTHROPIC_API_KEY, or use `ant auth login`
pixi run eval-baseline   # claude-haiku-5-5, zero-shot, JSON parsed from text
pixi run eval-candidate  # claude-opus-5-5, 3-shot, schema enforced by the API
pixi run eval-compare
```

`make help` lists the rest. Every target wraps a pixi task; `pixi.toml` is the source
of truth and `make` is there for people who reach for it first.

## What a run produces

```
reports/<name>/
  manifest.json     run name, config hash, prompt version + hash, git sha, provider id, timestamps
  metrics.json      everything below, plus per-slice and per-hazard-type breakdowns
  predictions.jsonl raw text, parsed output, token usage, latency, cache flag, per example
  scores.jsonl      per-example tp/fp/fn, hallucinated flags, score error
  failures.json     every example that was wrong or unparseable, with the raw text
  report.md         the human-readable version
```

Metrics:

| metric | what it measures |
|---|---|
| schema adherence | fraction of responses that parse and validate against `HazardReport` |
| hazard F1 (micro) | set overlap of predicted vs expected hazard types, with a bootstrap 95% CI |
| exact match rate | predicted hazard set equals the expected set |
| hallucination rate | predicted flags that cite a detection id not in the input, or a detection that cannot support that hazard type (a pool cited as evidence for roof damage). Denominator is predicted flags |
| risk score MAE, within ±1 | numeric field accuracy |
| latency p50/p95, cost | from the SDK's usage object and a pricing table in the provider |
| summary supported (optional) | LLM-as-judge yes/no on whether the prose summary is supported by the input |

`compare` runs a paired bootstrap on the F1 delta between two report directories and
exits non-zero if the delta is below `--min-f1-delta` or the hallucination rate is
above `--max-hallucination`. CI runs it on every push with the mock providers.

## Layout

```
src/eval_harness/
  schema.py        pydantic models: golden rows, the output contract, predictions.
                   The JSON schema sent to the API is generated from HazardReport.
  prompts.py       system prompt, few-shot builder (dev split only), prompt hashing
  providers/       Provider interface; AnthropicProvider; MockProvider with seeded noise
  parsing.py       text -> HazardReport, tolerant of fences and prose
  metrics.py       pure functions: PRF, hallucination check, bootstrap CIs, paired delta
  judge.py         LLM-as-judge for the free-text field
  runner.py        async run with concurrency limit, on-disk response cache, artefacts
  report.py        markdown rendering
  cli.py           run / compare / validate-data
configs/           one yaml per run
data/golden/       hazards_v1.jsonl (synthetic)
data/LABELING.md   the annotation protocol the schema is built for
scripts/           synthetic data generator
tests/             unit tests for metrics/parsing/schema, end-to-end run via the mock
```

## Decisions

**Golden set is validated, versioned and split.** Rows are pydantic-validated on load:
every expected hazard must cite detection ids that exist in the input, unknown fields
are rejected, duplicate ids fail. Each row carries `split`, `difficulty` and an
`annotation` block (source, annotator count, agreement). Few-shot examples only come
from `dev`. The file is treated as immutable once referenced by a run; fixes go in
`hazards_v2.jsonl`. See `data/LABELING.md` for the human protocol.

**The output contract is one pydantic model.** `HazardReport.json_schema_for_api()`
derives a strict JSON schema from it. The same model validates responses. There is no
second copy of the schema to drift.

**Hallucination is defined so it can be checked without a model.** A flag must cite
detection ids, and each id must exist and be of a class that can support that hazard.
That rule is checkable in a few lines and catches the two failure modes that matter
most for this task: invented evidence and misattributed evidence. The LLM judge is
only used for the prose field where a deterministic check is not possible.

**Providers are swappable and runs are cached.** The runner only sees
`Provider.complete(request)`. Responses are cached on disk keyed by a hash of
provider id, prompt, schema and max tokens, so re-running after a metrics change costs
nothing and CI never needs a secret. The mock applies the task rules and then injects
seeded noise (missed flags, hallucinated flags, malformed JSON, wrong scores), which is
what the metric tests run against.

**Structured output is a config flag, not a prompt trick.** `structured_output: true`
uses the API's `output_config.format` with the derived schema. `false` pastes the
schema into the system prompt and parses text. Both paths exist so the gap can be
measured rather than assumed.

**Numbers come with intervals.** Micro-F1 and hallucination rate are reported with
bootstrap 95% CIs. `compare` uses a paired bootstrap over shared example ids, which is
the right test when both runs saw the same examples.

**Manifests make a number reproducible.** `manifest.json` has the config hash, the
prompt version and hash, the provider id (model, effort, structured flag), the golden
file path and the git sha. Two reports can always be compared knowing what changed.

**Cost is tracked, and the stable prefix is cached.** The system prompt and few-shot
block carry `cache_control`; only the per-example user turn is uncached. Token usage
including cache reads and writes is recorded per example and priced.

## Environments and secrets

Two pixi environments share one solve group, so they resolve to the same versions:
`default` has the dev tools, `prod` is runtime only and is what the real-model
workflow installs. `pixi install --locked` is used everywhere in CI and fails if
`pixi.lock` is stale, so dependency changes go through `make lock` and a committed lock.

The only secret is `ANTHROPIC_API_KEY`. Locally it comes from the environment or a
gitignored `.env` (the CLI loads it). In GitHub it lives in the `eval` environment's
secrets and is only exposed to the `eval` workflow, never to PR builds.

## CI/CD

Three workflows under `.github/workflows/`:

| workflow | trigger | what it does |
|---|---|---|
| `ci` | push to main, PRs | lint, mypy, tests on Ubuntu and macOS; offline eval with the mock providers and the regression gate, comparison table in the job summary, reports as artefacts; build sdist and wheel |
| `release` | tag `v*.*.*` | checks the tag matches `__version__`, re-runs checks and the offline eval, builds, creates a GitHub release with the matching CHANGELOG section as notes. PyPI publishing via trusted publishing is wired and off by default |
| `eval` | manual, weekly | real-model baseline and candidate on the `prod` environment, cached responses restored between runs, `compare` as the gate, reports kept 90 days |

Release is `make release VERSION=x.y.z`: it refuses a dirty tree or a non-main branch,
runs the checks, bumps the version, dates the changelog, tags and pushes. The workflow
does the rest. Dependabot keeps the actions current.

## Adapting it

1. Replace `HazardReport` and `GoldenExample.expected` in `schema.py`.
2. Replace the rules in `prompts.py` and the evidence check in `metrics.py`.
3. Build a golden set following `data/LABELING.md`.
4. Write two configs and run `compare`.

## Licence

MIT.
