# Every CI step calls one of these targets, so local and CI runs can't diverge.
# `make check` runs the fast checks every PR gets; `make check-all` adds the heavy ones.

UV_RUN := uv run --locked
FAST := not slow and not acceptance and not e2e
HEAVY := slow or acceptance or e2e
# CI narrows TESTS to the test directories a change can affect (tools/affected.py).
TESTS ?= tests

.DEFAULT_GOAL := help
.PHONY: help check check-all lint format typecheck arch test test-heavy sanitize

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

check: lint typecheck arch test sanitize ## Fast checks, as on every PR

check-all: check test-heavy ## Fast and heavy checks

lint: ## Ruff lint and format check
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

format: ## Apply ruff fixes and formatting
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

typecheck: ## mypy
	$(UV_RUN) mypy

arch: ## Layer contracts (import-linter)
	$(UV_RUN) lint-imports

test: ## Fast tests, in parallel
	$(UV_RUN) pytest -n auto -m "$(FAST)" $(TESTS)

# pytest exits 5 when it selects no tests, which is expected until the first heavy test lands.
test-heavy: ## Slow, acceptance, and e2e tests
	$(UV_RUN) pytest -n auto -m "$(HEAVY)" $(TESTS) || [ $$? -eq 5 ]

sanitize: ## Denylist scan of every tracked file
	$(UV_RUN) python tools/sanitize_scan.py --tree
