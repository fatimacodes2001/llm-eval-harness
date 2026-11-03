# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[SemVer](https://semver.org/). `make release VERSION=x.y.z` moves the Unreleased section
under a new heading, tags, and pushes; the release workflow builds and publishes.

## [Unreleased]

## [0.1.0] - 2026-11-03

### Added
- Golden set schema with split, difficulty and label provenance; validation on load.
- `HazardReport` output contract and derived strict JSON schema for the API.
- Provider interface with Anthropic (structured outputs, prompt caching, cost) and a
  seeded mock for CI.
- Metrics: schema adherence, micro-F1 with bootstrap CI, exact match, deterministic
  hallucination rate, risk-score MAE, latency, cost; optional LLM judge for prose.
- `compare` command with paired bootstrap and regression gate.
- Run manifests, markdown reports, on-disk response cache.
- CI (lint, mypy, tests, offline eval gate, build), tag-driven release workflow,
  scheduled real-model eval workflow.
