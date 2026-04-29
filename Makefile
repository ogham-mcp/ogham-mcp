# Ogham MCP -- CI/CD Lite Makefile
# Flow: dev repo → public repo → PyPI + GHCR → gateway wheel → Neon migrations
#
# Usage:
#   make test          Run all tests
#   make sync          Sync dev repo → public repo
#   make publish       Build + publish to PyPI (1Password token)
#   make gateway       Build wheel + update gateway + lock + test
#   make migrate       Apply migrations to all 3 Neon DBs
#   make tag           Commit, push, tag, create GitHub release
#   make release       Full pipeline: sync → test → smoke → publish → tag → gateway
#   make release-patch Bump patch version + full release
#
# Prerequisites:
#   - 1Password CLI (op) for PyPI token
#   - uv for Python builds
#   - psycopg installed (for migrations)

.PHONY: test lint build publish tag clean wheel sync gateway migrate release release-patch check-clean version-check smoke preflight

# --- Paths ---
DEV_REPO := /Users/kevinburns/Developer/web-projects/openbrain-sharedmemory
PUB_REPO := /Users/kevinburns/Developer/web-projects/ogham-mcp
GATEWAY_REPO := /Users/kevinburns/Developer/web-projects/ogham-gateway

# --- Core targets ---

# Run all tests
test:
	uv run pytest tests/ -v

# Lint + format check
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run pyright

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info

# Build wheel + sdist (clean first)
build: clean
	uv build

# --- Sync dev → public repo ---

# Files to sync from dev to public
SYNC_SOURCES := \
	src/ogham/hooks.py \
	src/ogham/hooks_cli.py \
	src/ogham/hooks_install.py \
	src/ogham/hooks_config.yaml \
	src/ogham/service.py \
	src/ogham/database.py \
	src/ogham/config.py \
	src/ogham/extraction.py \
	src/ogham/embeddings.py \
	src/ogham/server.py \
	src/ogham/cli.py \
	src/ogham/init_wizard.py \
	src/ogham/backends/protocol.py \
	src/ogham/backends/postgres.py \
	src/ogham/backends/supabase.py \
	src/ogham/backends/gateway.py \
	tests/test_hooks.py \
	tests/test_extraction.py \
	tests/test_backend_wiring.py \
	sql/schema.sql \
	sql/schema_postgres.sql \
	sql/schema_selfhost_supabase.sql \
	sql/upgrade.sh \
	benchmarks/beam_benchmark.py \
	benchmarks/longmemeval_benchmark.py

sync:
	@echo "=== Syncing dev → public repo ==="
	@for f in $(SYNC_SOURCES); do \
		if [ -f "$(DEV_REPO)/$$f" ]; then \
			mkdir -p "$$(dirname $(PUB_REPO)/$$f)"; \
			cp "$(DEV_REPO)/$$f" "$(PUB_REPO)/$$f"; \
		fi; \
	done
	@echo "Synced $$(echo $(SYNC_SOURCES) | wc -w | tr -d ' ') files"
	@echo ""
	@echo "--- Changes in public repo ---"
	@cd $(PUB_REPO) && git diff --stat HEAD

# --- Publish to PyPI ---

publish-check:
	@echo "=== Publish-check: scanning dist/ for secrets ==="
	@if [ ! -d dist ] || [ -z "$$(ls dist/*.tar.gz 2>/dev/null)" ]; then \
		echo "  ✗ No sdist in dist/ -- run 'make build' first"; exit 1; fi
	@TMPDIR=$$(mktemp -d); \
	SDIST=$$(ls -t dist/*.tar.gz 2>/dev/null | head -1); \
	echo "  Scanning: $$SDIST"; \
	tar -xzf "$$SDIST" -C "$$TMPDIR"; \
	HITS=$$(grep -rIE 'sk_(test|live)_[a-zA-Z0-9]{12,}|pk_(test|live)_[a-zA-Z0-9]{12,}|sbp_[a-zA-Z0-9]{20,}|sb_(secret|publishable)_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[a-zA-Z0-9]{30,}|npg_[a-zA-Z0-9]{10,}|postgres(ql)?://[^:@[:space:]]+:[^@[:space:]]+@[^[:space:]/]+' "$$TMPDIR" 2>/dev/null | grep -vE "\.env\.example|user:pass|your_|YOUR_|<password>|REPLACE_|example\.com|@localhost|@127\.0\.0\.1|ABCDEFghij|ogham_test|s3cretP4ss" || true); \
	rm -rf "$$TMPDIR"; \
	if [ -n "$$HITS" ]; then \
		echo ""; \
		echo "❌ Publish blocked: credential-like strings detected in sdist:"; \
		echo "$$HITS" | head -10 | sed 's/^/    /'; \
		echo ""; \
		echo "   History: v0.10.x series accidentally shipped Neon postgres URLs"; \
		echo "   via hardcoded Makefile vars. Check Makefile / source for embedded"; \
		echo "   creds and move them to env vars or 'op read' indirection."; \
		exit 1; \
	fi
	@echo "  ✓ No credential-like strings in dist/"

publish: build publish-check
	@echo "=== Publishing to PyPI ==="
	uv publish --token $$(op read "op://Ogham-Gateway/PyPi - Ogham Dev token/api_key")
	@echo ""
	@echo "Published $$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"

# --- GitHub release ---

tag:
	@VERSION=$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	echo "=== Creating GitHub release v$$VERSION ==="; \
	BEFORE_HEAD=$$(git rev-parse HEAD); \
	git add -A; \
	if ! git commit -m "v$$VERSION" --allow-empty; then \
		echo "ERROR: 'git commit --allow-empty' returned non-zero -- pre-commit hook (prek/ruff/pyright) likely blocked the commit."; \
		echo "       Run 'git commit --allow-empty -m \"v$$VERSION\"' manually to see the hook output, fix it, then re-run 'make tag'."; \
		exit 1; \
	fi; \
	AFTER_HEAD=$$(git rev-parse HEAD); \
	if [ "$$BEFORE_HEAD" = "$$AFTER_HEAD" ]; then \
		echo "ERROR: HEAD did not move after 'git commit --allow-empty' (was $$BEFORE_HEAD, still $$BEFORE_HEAD)."; \
		echo "       A pre-commit hook silently aborted the commit. Tagging this HEAD would publish stale code."; \
		echo "       This was the root cause of the v0.13.0 ship incident -- do not bypass."; \
		exit 1; \
	fi; \
	git push origin main; \
	git tag -a "v$$VERSION" -m "v$$VERSION"; \
	git push origin "v$$VERSION"; \
	PREV_TAG=$$(git tag --sort=-v:refname | grep -v "v$$VERSION" | head -1); \
	if [ -n "$$PREV_TAG" ]; then \
		CHANGES=$$(git log --oneline "$$PREV_TAG"..HEAD -- src/ tests/ | sed 's/^/- /'); \
	else \
		CHANGES=$$(git log --oneline -10 -- src/ tests/ | sed 's/^/- /'); \
	fi; \
	NOTES=$$(printf "## What'\''s changed\n\n$$CHANGES\n\n**Full changelog:** https://github.com/ogham-mcp/ogham-mcp/compare/$$PREV_TAG...v$$VERSION\n\n**Install:** \`uv tool install ogham-mcp\` or \`pip install ogham-mcp==$$VERSION\`"); \
	gh release create "v$$VERSION" \
		--title "v$$VERSION" \
		--notes "$$NOTES" \
		--latest; \
	echo "Released v$$VERSION on GitHub"

# --- Gateway wheel update ---

wheel: build
	@echo "Wheel ready at dist/"
	@ls dist/*.whl

gateway: build
	@echo "=== Updating gateway vendored wheel ==="
	@VERSION=$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	echo "Version: $$VERSION"; \
	rm -f $(GATEWAY_REPO)/vendor/ogham_mcp-*.whl; \
	cp dist/ogham_mcp-$$VERSION-py3-none-any.whl $(GATEWAY_REPO)/vendor/; \
	cd $(GATEWAY_REPO) && \
	sed -i '' "s|ogham-mcp = { path = \"vendor/ogham_mcp-.*\.whl\" }|ogham-mcp = { path = \"vendor/ogham_mcp-$$VERSION-py3-none-any.whl\" }|" pyproject.toml && \
	uv lock --upgrade-package ogham-mcp && \
	echo "" && \
	echo "--- Running gateway tests ---" && \
	uv run pytest tests/ -x -q && \
	echo "" && \
	echo "Gateway updated to $$VERSION. Run 'make gateway-push' to deploy."

gateway-push:
	@echo "=== Pushing gateway to Railway ==="
	@VERSION=$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	cd $(GATEWAY_REPO) && \
	git add vendor/ pyproject.toml uv.lock && \
	git commit -m "chore: upgrade vendored ogham-mcp to $$VERSION" && \
	git push
	@echo "Gateway deployed. Railway will rebuild."

# --- Database migrations ---

# Regional database URLs come from the environment -- never hardcoded.
# Set these before running `make migrate` (e.g. via direnv / 1Password /
# `op run --env-file=...`). This was where we used to embed live
# credentials in earlier releases -- removed in v0.11.0 after discovering
# the leak. See CHANGELOG v0.11.0 release notes for detail.
NEON_US ?= $(shell op read "op://Ogham-Gateway/supabase-regional-databases/us-direct-url" 2>/dev/null)
NEON_EU ?= $(shell op read "op://Ogham-Gateway/supabase-regional-databases/eu-direct-url" 2>/dev/null)
NEON_AP ?= $(shell op read "op://Ogham-Gateway/supabase-regional-databases/ap-pooler-url" 2>/dev/null)

migrate:
	@echo "=== Applying migrations to all Neon databases ==="
	@for region in US EU AP; do \
		case $$region in \
			US) URL=$(NEON_US);; \
			EU) URL=$(NEON_EU);; \
			AP) URL=$(NEON_AP);; \
		esac; \
		echo "--- $$region ---"; \
		for f in sql/migrations/*.sql; do \
			echo "  Applying $$(basename $$f)..."; \
			psql "$$URL" -f "$$f" -v ON_ERROR_STOP=1 > /dev/null 2>&1 || echo "  Warning: $$(basename $$f) had issues"; \
		done; \
		echo "  Done."; \
	done
	@echo ""
	@echo "All regions migrated."

# --- Version management ---

version-check:
	@echo "=== Version check ==="
	@echo "Public repo: $$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
	@echo "PyPI:        $$(curl -s https://pypi.org/pypi/ogham-mcp/json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null || echo 'unknown')"
	@echo "Gateway:     $$(grep 'ogham_mcp-' $(GATEWAY_REPO)/pyproject.toml | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')"

# --- Full release pipeline ---

# --- Sanity checks ---

# Smoke test: install the wheel in a temp venv and verify all modules import
smoke: build
	@echo "=== Smoke test: clean install from wheel ==="
	@TMPDIR=$$(mktemp -d); \
	python3 -m venv "$$TMPDIR/venv"; \
	WHL=$$(ls dist/ogham_mcp-*.whl | head -1); \
	"$$TMPDIR/venv/bin/pip" install -q "$$WHL[postgres]" 2>&1 | tail -1; \
	echo "--- Module imports ---"; \
	"$$TMPDIR/venv/bin/python" -c "\
import ogham.config; print('  ✓ ogham.config'); \
import ogham.database; print('  ✓ ogham.database'); \
import ogham.embeddings; print('  ✓ ogham.embeddings'); \
import ogham.service; print('  ✓ ogham.service'); \
import ogham.extraction; print('  ✓ ogham.extraction'); \
import ogham.hooks; print('  ✓ ogham.hooks'); \
import ogham.hooks_cli; print('  ✓ ogham.hooks_cli'); \
import ogham.hooks_install; print('  ✓ ogham.hooks_install'); \
import ogham.cli; print('  ✓ ogham.cli'); \
import ogham.server; print('  ✓ ogham.server'); \
import ogham.backends.postgres; print('  ✓ ogham.backends.postgres'); \
import ogham.backends.supabase; print('  ✓ ogham.backends.supabase'); \
import ogham.backends.gateway; print('  ✓ ogham.backends.gateway'); \
" || (echo "✗ SMOKE TEST FAILED"; rm -rf "$$TMPDIR"; exit 1); \
	echo "--- CLI entrypoint ---"; \
	"$$TMPDIR/venv/bin/ogham" --help > /dev/null 2>&1 && echo "  ✓ ogham CLI starts" || echo "  ✗ ogham CLI failed"; \
	"$$TMPDIR/venv/bin/ogham" hooks --help > /dev/null 2>&1 && echo "  ✓ ogham hooks sub-command" || (echo "  ✗ ogham hooks failed"; rm -rf "$$TMPDIR"; exit 1); \
	echo "--- Wheel contents ---"; \
	python3 -m zipfile -l dist/ogham_mcp-*.whl | grep -c "\.py$$" | xargs -I{} echo "  {} Python files in wheel"; \
	python3 -m zipfile -l dist/ogham_mcp-*.whl | grep "hooks_config.yaml" > /dev/null && echo "  ✓ hooks_config.yaml included" || (echo "  ✗ hooks_config.yaml MISSING"; rm -rf "$$TMPDIR"; exit 1); \
	rm -rf "$$TMPDIR"; \
	echo ""; \
	echo "✓ Smoke test passed."

# Preflight: all checks before publishing
preflight: version-check
	@echo ""
	@echo "=== Preflight checks ==="
	@echo "--- Git state ---"
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "  ✗ Working directory not clean:"; \
		git status --short | sed 's/^/    /'; \
	else \
		echo "  ✓ Working directory clean"; \
	fi
	@echo "--- Dev repo sync ---"
	@DRIFT=0; \
	for f in $(SYNC_SOURCES); do \
		if [ -f "$(DEV_REPO)/$$f" ] && [ -f "$(PUB_REPO)/$$f" ]; then \
			if ! diff -q "$(DEV_REPO)/$$f" "$(PUB_REPO)/$$f" > /dev/null 2>&1; then \
				echo "  ✗ Out of sync: $$f"; \
				DRIFT=1; \
			fi; \
		fi; \
	done; \
	if [ $$DRIFT -eq 0 ]; then echo "  ✓ All files in sync with dev repo"; fi
	@echo "--- Version consistency ---"
	@PUB_VER=$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	PYPI_VER=$$(curl -s https://pypi.org/pypi/ogham-mcp/json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null || echo 'unknown'); \
	GW_VER=$$(grep 'ogham_mcp-' $(GATEWAY_REPO)/pyproject.toml | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+'); \
	if [ "$$PUB_VER" = "$$PYPI_VER" ]; then echo "  ✓ PyPI matches ($$PYPI_VER)"; else echo "  ✗ PyPI drift: repo=$$PUB_VER pypi=$$PYPI_VER"; fi; \
	if [ "$$PUB_VER" = "$$GW_VER" ]; then echo "  ✓ Gateway matches ($$GW_VER)"; else echo "  ✗ Gateway drift: repo=$$PUB_VER gateway=$$GW_VER"; fi

check-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: working directory not clean. Commit or stash changes first."; \
		git status --short; \
		exit 1; \
	fi

release: check-clean sync test smoke publish tag gateway
	@echo ""
	@echo "============================================"
	@echo "  Release pipeline complete!"
	@echo "============================================"
	@echo ""
	@make version-check
	@echo ""
	@echo "Remaining manual steps:"
	@echo "  1. Review gateway changes: cd $(GATEWAY_REPO) && git diff"
	@echo "  2. Push gateway: make gateway-push"
	@echo "  3. Apply migrations if needed: make migrate"

release-patch:
	@echo "=== Bumping patch version ==="
	@CURRENT=$$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	PATCH=$$(echo $$CURRENT | cut -d. -f3); \
	NEW="$$MAJOR.$$MINOR.$$((PATCH+1))"; \
	echo "$$CURRENT → $$NEW"; \
	sed -i '' "s/version = \"$$CURRENT\"/version = \"$$NEW\"/" pyproject.toml; \
	git add pyproject.toml && git commit -m "chore: bump to $$NEW"
	@make release
