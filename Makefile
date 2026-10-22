# Every CI step calls one of these targets, so local and CI runs can't diverge.
# `make check` runs the fast checks every PR gets; `make check-all` adds the heavy ones.

UV_RUN := uv run --locked
FAST := not slow and not acceptance and not e2e
HEAVY := slow or acceptance or e2e
# The official image, pinned by digest. Used when no local gitleaks binary is installed.
GITLEAKS_IMAGE := ghcr.io/gitleaks/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f
REQUIREMENTS := .cache/requirements-all.txt
COVERAGE_XML := .cache/coverage.xml
# CI narrows TESTS to the test directories a change can affect (tools/affected.py).
TESTS ?= tests

.DEFAULT_GOAL := help
.PHONY: help check check-all lint format typecheck arch workflows deps test test-cov diff-coverage memory test-heavy bench-world baselines sanitize sanitize-pr sanitize-main sanitize-text sanitize-artifact secrets affected

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

check: lint typecheck arch workflows test sanitize ## Fast checks, as on every PR

check-all: check test-heavy deps ## Fast and heavy checks, and the dependency audit

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

workflows: ## Lint the GitHub workflows (actionlint, zizmor)
	$(UV_RUN) actionlint
	$(UV_RUN) zizmor .github

# Needs the network: pip-audit queries vulnerability databases, and the license check reads PyPI.
deps: ## Lockfile freshness, known vulnerabilities, and dependency licenses
	uv lock --check
	@mkdir -p .cache
	uv export --frozen --all-extras --all-groups --format requirements-txt --no-emit-project \
		--quiet > $(REQUIREMENTS)
	$(UV_RUN) pip-audit --disable-pip --require-hashes -r $(REQUIREMENTS)
	$(UV_RUN) python tools/licenses.py $(REQUIREMENTS)

test: ## Fast tests, in parallel
	$(UV_RUN) pytest -n auto -m "$(FAST)" $(TESTS)

test-cov: ## Fast tests with coverage, for the diff-coverage gate
	$(UV_RUN) pytest -n auto -m "$(FAST)" --cov --cov-report=xml:$(COVERAGE_XML) \
		--cov-report=term $(TESTS)

# CI: coverage of the lines a PR changes since $BASE. hcsim.financial carries the accounting, so
# its bar is higher. Run after test-cov.
diff-coverage: ## Changed-line coverage: 90%, and 95% in hcsim.financial
	$(UV_RUN) diff-cover $(COVERAGE_XML) --compare-branch="$$BASE" --fail-under=90
	$(UV_RUN) diff-cover $(COVERAGE_XML) --compare-branch="$$BASE" --fail-under=95 \
		--include 'src/hcsim/financial/*'

# Peak-memory limits on tests marked @pytest.mark.limit_memory("…"). Memory is the engine's
# binding constraint, and unlike timing on shared runners it's nearly deterministic.
memory: ## Peak-memory limits (pytest-memray)
	$(UV_RUN) pytest --memray -m limit_memory $(TESTS) || [ $$? -eq 5 ]

# pytest exits 5 when it selects no tests, which is expected until the first heavy test lands.
test-heavy: ## Slow, acceptance, and e2e tests
	$(UV_RUN) pytest -n auto -m "$(HEAVY)" $(TESTS) || [ $$? -eq 5 ]

# Each preset in its own process, timed, with its peak memory. Timing is recorded, never gated.
# The memory limit stops a preset too big for the machine before it swaps.
BENCH_PRESETS ?= tiny small medium
bench-world: ## Time generating, writing, and checking each world preset
	$(UV_RUN) hcsim bench world $(foreach p,$(BENCH_PRESETS),--preset $(p)) --max-rss-gib 16

# The e2e tests compare each scorecard with a committed baseline. After an intentional change,
# regenerate the baselines and commit them in the same PR.
baselines: ## Regenerate the committed metric baselines
	HCSIM_UPDATE_BASELINES=1 $(UV_RUN) pytest -m e2e tests/e2e

sanitize: ## Denylist scan of every tracked file
	$(UV_RUN) python tools/sanitize_scan.py --tree

# CI: what a PR adds since $BASE (lines, file names, commit messages) and its $BRANCH name.
# Both come from the environment, never from make arguments, so they can't reach the shell.
sanitize-pr: ## Denylist scan of a PR's changes and branch name (CI)
	$(UV_RUN) python tools/sanitize_scan.py --range "$$BASE..HEAD"
	printf '%s' "$$BRANCH" | $(UV_RUN) python tools/sanitize_scan.py --stdin "branch name"

# CI, on every push to main: every tracked file, plus what the push added since $BEFORE
# (zeros when the branch is new). Catches anything a PR-diff scan could miss.
sanitize-main: ## Denylist scan of the whole tree and the pushed commits (CI)
	$(UV_RUN) python tools/sanitize_scan.py --tree
	if [ -n "$$BEFORE" ] && ! printf '%s' "$$BEFORE" | grep -Eq '^0+$$'; then \
		$(UV_RUN) python tools/sanitize_scan.py --range "$$BEFORE..HEAD"; fi

# CI: text on stdin, reported as $NAME. Covers issue and PR text, comments, branch and tag names,
# and release notes. The text never touches a command line.
sanitize-text: ## Denylist scan of text on stdin, reported as NAME (CI)
	$(UV_RUN) python tools/sanitize_scan.py --stdin "$$NAME"

# CI, before any upload: every file under $ARTIFACT_PATH, names and text content. A missing path
# has nothing to upload, so it passes.
sanitize-artifact: ## Denylist scan of an artifact before upload (CI)
	if [ -e "$$ARTIFACT_PATH" ]; then \
		find "$$ARTIFACT_PATH" -type f -print0 | xargs -0 $(UV_RUN) python tools/sanitize_scan.py; \
	fi

# Credential patterns across the whole git history. Findings print redacted. Not part of
# `make check`, so local checks don't need gitleaks: install it with `brew install gitleaks`.
secrets: ## Credential scan of the git history (gitleaks)
	@if command -v gitleaks >/dev/null; then \
		gitleaks git --redact --no-banner --verbose .; \
	elif command -v docker >/dev/null; then \
		docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp -v "$(CURDIR):/repo:ro" \
			$(GITLEAKS_IMAGE) git --redact --no-banner --verbose /repo; \
	else \
		echo "gitleaks isn't installed and docker isn't available: brew install gitleaks"; exit 2; \
	fi

# CI passes the PR's base commit. With no BASE, everything runs.
affected: ## Test paths a change since BASE can affect
	@$(UV_RUN) --quiet python tools/affected.py --base "$(BASE)"
