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
uv sync
```

## Validation

For the local non-integration test suite, use a placeholder `SUPABASE_URL` if
you have not configured a real backend yet. Some modules validate settings at
import time during test collection.

```bash
uv run ruff check src tests

SUPABASE_URL=https://fake.supabase.co \
  uv run pytest tests -m 'not integration and not postgres_integration' -q
```

Integration suites stay separate:

```bash
uv run pytest tests/test_integration.py -v

DATABASE_BACKEND=postgres DATABASE_URL="postgres://..." \
  uv run pytest tests/test_postgres_integration.py -v
```

## Pull requests

- Keep the branch and PR focused on one change.
- Explain what changed and why.
- Include the validation commands you ran.
- If the change affects behavior, add or update tests with it.
