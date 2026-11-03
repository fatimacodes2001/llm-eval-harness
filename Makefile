# Thin wrapper over pixi tasks so `make <thing>` works for people who expect it.
# Everything here delegates to pixi; pixi.toml is the source of truth.

PIXI ?= pixi
VERSION ?=

.DEFAULT_GOAL := help
.PHONY: help install lock lint fmt typecheck test check data validate-data demo \
        eval smoke compare build clean release pre-commit

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the dev environment from the lock file
	$(PIXI) install --locked

lock: ## Re-resolve dependencies and update pixi.lock
	$(PIXI) install

lint: ## ruff check + format check
	$(PIXI) run lint

fmt: ## ruff format
	$(PIXI) run fmt

typecheck: ## mypy --strict on src
	$(PIXI) run typecheck

test: ## pytest with coverage
	$(PIXI) run test

check: ## lint + typecheck + test + validate-data (what CI runs)
	$(PIXI) run check

data: ## Regenerate the synthetic golden set
	$(PIXI) run make-data

validate-data: ## Validate the golden set
	$(PIXI) run validate-data

demo: ## Offline baseline/candidate run + regression gate (no API key)
	$(PIXI) run demo

eval: ## Real-model baseline + candidate + compare (needs ANTHROPIC_API_KEY)
	$(PIXI) run eval-baseline && $(PIXI) run eval-candidate && $(PIXI) run eval-compare

smoke: ## 5-example real-model run, cache bypassed
	$(PIXI) run smoke

compare: ## Compare two report dirs: make compare A=reports/x B=reports/y
	$(PIXI) run eval-harness compare $(A) $(B)

build: ## Build sdist + wheel into dist/
	$(PIXI) run build

pre-commit: ## Install git hooks
	$(PIXI) run pre-commit install

clean: ## Remove caches, reports and build output
	rm -rf .cache reports/* dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	touch reports/.gitkeep

release: ## Bump version, update changelog date, tag and push: make release VERSION=0.2.0
	@test -n "$(VERSION)" || { echo "usage: make release VERSION=x.y.z"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree not clean"; exit 1; }
	@test "$$(git branch --show-current)" = "main" || { echo "release from main only"; exit 1; }
	@grep -q "^## \[Unreleased\]" CHANGELOG.md || { echo "CHANGELOG.md needs an Unreleased section"; exit 1; }
	$(PIXI) run check
	sed -i.bak 's/^__version__ = .*/__version__ = "$(VERSION)"/' src/eval_harness/__init__.py && rm src/eval_harness/__init__.py.bak
	sed -i.bak "s/^## \[Unreleased\]/## [Unreleased]\n\n## [$(VERSION)] - $$(date +%Y-%m-%d)/" CHANGELOG.md && rm CHANGELOG.md.bak
	git add src/eval_harness/__init__.py CHANGELOG.md
	git commit -m "Release v$(VERSION)"
	git tag -a "v$(VERSION)" -m "v$(VERSION)"
	git push origin main "v$(VERSION)"
