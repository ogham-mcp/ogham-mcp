# Contributing

Thanks for contributing to Ogham MCP.

Keep changes small, focused, and easy to review. Bug fixes, tests, docs, and
small usability improvements are all good contributions. For larger changes,
open an issue or discussion first so the approach is aligned before you spend
time implementing it.

## What to contribute

- bug fixes
- tests that protect real regressions
- documentation improvements
- small, well-scoped developer experience improvements

## Local setup

```bash
git clone https://github.com/ogham-mcp/ogham-mcp.git
cd ogham-mcp
uv sync --extra dev --extra postgres
```

## Validation

Run the full non-integration suite before opening a PR:

```bash
uv run ruff check src tests

uv run pytest tests -m 'not integration and not postgres_integration' -q
```

Integration suites stay separate:

```bash
uv run pytest tests/test_integration.py -v

DATABASE_BACKEND=postgres DATABASE_URL="postgres://..." \
  uv run pytest tests/test_postgres_integration.py -v
```

## Regression-proof rules

- Keep module imports side-effect free. Importing a module should not require a
  real database, API key, model download, or cache directory.
- Do not read `settings.*` at import time when the value can be resolved lazily.
  Read config at the point of use.
- Do not create clients, pools, caches, or model sessions at import time.
  Initialize them on first real use.
- Validate external configuration where the external dependency is actually
  used. Example: validate Supabase credentials when creating the Supabase
  client, not when importing a tools module.
- When fixing a bug, add or update the smallest test that proves the regression.
  Prefer import/collection tests for import-time failures.
- If a new test needs fake environment variables just to import application
  code, treat that as a design smell and fix the source first.

## Pull requests

- Keep the branch and PR focused on one change.
- Explain what changed and why.
- Include the validation commands you ran.
- If the change affects behavior, add or update tests with it.
