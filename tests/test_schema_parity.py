"""Schema parity tests -- catches the PR #23 bug class.

Uses inspect.unwrap() to get real method signatures past decorators,
and regex for SQL function signatures. No database needed.
"""

import inspect
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_FILES = [
    REPO_ROOT / "sql" / "schema_postgres.sql",
    REPO_ROOT / "sql" / "schema.sql",
]


def _sql_function_param_count(schema_path: Path, function_name: str) -> int:
    """Count parameters in a SQL CREATE FUNCTION signature."""
    sql = schema_path.read_text()
    pattern = rf"CREATE OR REPLACE FUNCTION\s+{function_name}\s*\((.*?)\)\s*RETURNS"
    match = re.search(pattern, sql, re.DOTALL | re.IGNORECASE)
    if not match:
        return -1
    params = [
        p.strip() for p in match.group(1).split(",") if p.strip() and not p.strip().startswith("--")
    ]
    return len(params)


def _python_method_param_count(cls, method_name: str) -> int:
    """Count parameters using inspect.unwrap to see past decorators."""
    method = getattr(cls, method_name, None)
    if method is None:
        return -1
    original = inspect.unwrap(method)
    sig = inspect.signature(original)
    return len([p for p in sig.parameters if p != "self"])


# --- Tests ---


def test_hybrid_search_param_parity():
    """Python params + hardcoded SQL values must equal SQL function params.

    Python method has 9 params. SQL function has 12 (adds full_text_weight,
    semantic_weight, rrf_k which are hardcoded in the SQL call string).
    9 + 3 = 12.
    """
    from ogham.backends.postgres import PostgresBackend

    py_count = _python_method_param_count(PostgresBackend, "hybrid_search_memories")
    assert py_count > 0, "Could not inspect hybrid_search_memories"

    hardcoded = 3  # 0.3::float, 0.7::float, 10::integer in the SQL string
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql_count = _sql_function_param_count(schema, "hybrid_search_memories")
        assert sql_count > 0, f"hybrid_search_memories not found in {schema.name}"
        assert py_count + hardcoded == sql_count, (
            f"{schema.name}: Python {py_count} + {hardcoded} hardcoded = "
            f"{py_count + hardcoded}, SQL expects {sql_count}"
        )


def test_match_memories_param_parity():
    """match_memories: Python params must match SQL function params."""
    from ogham.backends.postgres import PostgresBackend

    py_count = _python_method_param_count(PostgresBackend, "search_memories")
    if py_count <= 0:
        return
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql_count = _sql_function_param_count(schema, "match_memories")
        if sql_count <= 0:
            continue
        assert py_count == sql_count, f"{schema.name}: Python {py_count}, SQL {sql_count}"


def test_sql_functions_present():
    """Key SQL functions must exist in all schema files."""
    required = [
        "hybrid_search_memories",
        "match_memories",
        "record_access",
        "auto_link_memory",
    ]
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql = schema.read_text()
        for func in required:
            assert func in sql, f"{func} missing from {schema.name}"


def test_no_unnumbered_migrations():
    """Migration files must be numbered (prevents Josh's issue #22).

    Allows an optional letter suffix (008a, 008b) for historical mid-sequence
    inserts, but rejects unnumbered files like `update_search_function.sql`.
    """
    migrations_dir = REPO_ROOT / "sql" / "migrations"
    if not migrations_dir.exists():
        return
    for f in migrations_dir.glob("*.sql"):
        assert re.match(r"^\d{3}[a-z]?_", f.name), (
            f"{f.name} is not numbered -- will sort wrong in upgrade.sh"
        )


def test_schema_has_required_tables():
    """All schema files must define the core tables."""
    required_tables = ["memories", "memory_relationships", "profile_settings"]
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql = schema.read_text().lower()
        for table in required_tables:
            assert table in sql, f"{table} missing from {schema.name}"
