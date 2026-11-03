# Contributing

## Setup

```bash
make install        # pixi install --locked
make pre-commit     # ruff + hygiene hooks on commit
make check          # what CI runs
```

Use `make lock` after changing dependencies in `pixi.toml`, and commit `pixi.lock`. CI
installs with `--locked` and fails if the lock is stale.

## Environments

- `default` is the dev environment (tests, linters, build tools).
- `prod` is runtime only, resolved in the same solve group so versions match.
  `pixi run -e prod eval-harness ...` is what the eval workflow uses.
- Secrets come from `ANTHROPIC_API_KEY` in the environment or a local `.env`
  (gitignored, see `.env.example`). Nothing in the repo reads any other secret.

## Changing the eval

- A change to `prompts.py`, a config, or the golden file is a change to the number.
  Run `make demo` before and after; run `make smoke` if you have a key.
- Golden files are immutable once a run has referenced them. Add `hazards_v2.jsonl`.
- Add a line under `## [Unreleased]` in `CHANGELOG.md`.

## Releasing

```bash
make release VERSION=0.2.0
```

Checks a clean tree on `main`, runs `make check`, bumps `__version__`, dates the
changelog section, tags `v0.2.0` and pushes. The `release` workflow verifies the tag
matches the package version, re-runs checks and the offline eval, builds the wheel,
and creates a GitHub release with the changelog section as notes. PyPI publishing is
wired through trusted publishing and off until `PUBLISH_TO_PYPI=true` is set.
