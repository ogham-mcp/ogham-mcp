# Ogham MCP -- CI/CD Lite Makefile
# Flow: dev repo → public repo → PyPI + GHCR → gateway wheel → Neon migrations
#
# Usage:
#   make test          Run all tests
#   make sync          Sync dev repo → public repo
#   make publish       Build + publish to PyPI (1Password token)
#   make gateway       Build wheel + update gateway + lock + test
#   make migrate       Apply migrations to all 3 Neon DBs
#   make release       Full pipeline: sync → test → publish → gateway → migrate
#   make release-patch Bump patch version + full release
#
# Prerequisites:
#   - 1Password CLI (op) for PyPI token
#   - uv for Python builds
#   - psycopg installed (for migrations)

.PHONY: test lint build publish clean wheel sync gateway migrate release release-patch check-clean version-check

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
	sql/upgrade.sh

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

publish: build
	@echo "=== Publishing to PyPI ==="
	uv publish --token $$(op read "op://Ogham-Gateway/PyPi - Ogham Dev token/api_key")
	@echo ""
	@echo "Published $$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"

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

NEON_US := "postgresql://neondb_owner:npg_2cSLWlVye8Dj@ep-holy-snow-ae0hur3m-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
NEON_EU := "postgresql://neondb_owner:npg_kHJsPtF4i3CR@ep-soft-grass-aln9vxgp-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
NEON_AP := "postgresql://neondb_owner:npg_qXLua6z0cRHe@ep-curly-dew-a1uok6f9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

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

check-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: working directory not clean. Commit or stash changes first."; \
		git status --short; \
		exit 1; \
	fi

release: check-clean sync test publish gateway
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
	@echo "  4. Tag release: gh release create v\$$(VERSION)"

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
