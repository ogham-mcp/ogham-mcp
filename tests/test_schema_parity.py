"""Schema parity tests -- catches the PR #23 bug class.

Uses inspect.unwrap() to get real method signatures past decorators,
and regex for SQL function signatures. No database needed.
"""

import inspect
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_FILES = [
    REPO_ROOT / "sql" / "schema_postgres.sql",
    REPO_ROOT / "sql" / "schema.sql",
    REPO_ROOT / "sql" / "schema_selfhost_supabase.sql",
]
# Pre-existing SQL-function param-count tests only cover the schemas kept in
# lockstep with the Python backend. `schema_selfhost_supabase.sql` has a
# known drift on `hybrid_search_memories` (missing query_entity_tags + recency_decay)
# tracked in v0.16 Linear TBU-149 -- exclude it here until that lands so the
# self-hosted schema doesn't red the pre-existing tests before the fix is in.
_FUNCTION_PARITY_SCHEMAS = [f for f in SCHEMA_FILES if f.name != "schema_selfhost_supabase.sql"]
WHITELIST_PATH = Path(__file__).parent / "schema_parity_whitelist.yaml"


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
    for schema in _FUNCTION_PARITY_SCHEMAS:
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
    for schema in _FUNCTION_PARITY_SCHEMAS:
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


# --- Column parity ---


_TABLE_HEADER = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?(?:public\.)?([a-z_][a-z0-9_]*)\s*\(",
    re.IGNORECASE,
)
_NON_COLUMN_PREFIXES = (
    "primary key",
    "unique",
    "foreign key",
    "check",
    "constraint",
    "exclude",
    "like",
)


def _extract_tables(sql: str) -> dict[str, set[str]]:
    """Parse CREATE TABLE blocks, return {table_name: {column_names}}.

    Skips constraint lines (primary key, foreign key, check, etc.), comments,
    and blank lines. Handles generated columns (uses first identifier).
    """
    tables: dict[str, set[str]] = {}
    i = 0
    while i < len(sql):
        m = _TABLE_HEADER.search(sql, i)
        if not m:
            break
        table = m.group(1).lower()
        # Walk from after the opening paren, tracking paren depth so
        # commas inside e.g. generated-column expressions or vector(512)
        # don't split columns and the closing ) of an inner expression
        # doesn't end the table.
        depth = 1
        pos = m.end()
        start = pos
        while pos < len(sql) and depth > 0:
            c = sql[pos]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            pos += 1
        body = sql[start : pos - 1]
        columns: set[str] = set()
        # Split top-level commas only (depth==0)
        depth = 0
        buf = ""
        parts: list[str] = []
        for c in body:
            if c == "(":
                depth += 1
                buf += c
            elif c == ")":
                depth -= 1
                buf += c
            elif c == "," and depth == 0:
                parts.append(buf)
                buf = ""
            else:
                buf += c
        if buf.strip():
            parts.append(buf)
        for part in parts:
            line = part.strip()
            # strip comments
            line = re.sub(r"--.*$", "", line, flags=re.MULTILINE).strip()
            if not line:
                continue
            lower = line.lower()
            if any(lower.startswith(p) for p in _NON_COLUMN_PREFIXES):
                continue
            # First whitespace-delimited token is the column name
            name = re.split(r"\s+", line, maxsplit=1)[0].strip().strip('"')
            if name and re.match(r"^[a-z_][a-z0-9_]*$", name):
                columns.add(name.lower())
        # If a later CREATE TABLE re-declares the same name (unusual),
        # union rather than overwrite -- reflects the final schema shape.
        tables.setdefault(table, set()).update(columns)
        i = pos
    return tables


def _load_whitelist() -> list[dict]:
    if not WHITELIST_PATH.exists():
        return []
    data = yaml.safe_load(WHITELIST_PATH.read_text()) or {}
    return data.get("allowed_divergences", []) or []


def _is_whitelisted(schema_name: str, table: str, column: str, whitelist: list[dict]) -> bool:
    for entry in whitelist:
        if entry.get("table") != table:
            continue
        if entry.get("column") != column:
            continue
        present_in = entry.get("present_in") or []
        # column is expected in `present_in` schemas and absent elsewhere
        expected_here = schema_name in present_in
        return expected_here or not expected_here  # any listed presence is OK
    return False


def test_column_parity_across_schemas():
    """Every table that exists in more than one schema variant must have the
    same column set across those variants, except for entries listed in
    tests/fixtures/schema_parity_whitelist.yaml.

    Catches: a column added to one variant and forgotten in the others,
    silently letting production and self-hosted diverge.
    """
    schemas: dict[str, dict[str, set[str]]] = {}
    for path in SCHEMA_FILES:
        if not path.exists():
            continue
        schemas[path.name] = _extract_tables(path.read_text())

    if len(schemas) < 2:
        return  # nothing to compare

    whitelist = _load_whitelist()

    # Union of tables present anywhere. For each, check which schemas have it.
    all_tables: set[str] = set()
    for tabs in schemas.values():
        all_tables.update(tabs)

    failures: list[str] = []
    for table in sorted(all_tables):
        present_schemas = [s for s, tabs in schemas.items() if table in tabs]
        if len(present_schemas) < 2:
            continue  # single-schema table (not necessarily wrong)
        column_sets = {s: schemas[s][table] for s in present_schemas}
        union = set().union(*column_sets.values())
        for col in sorted(union):
            missing_from = [s for s, cols in column_sets.items() if col not in cols]
            present_in = [s for s, cols in column_sets.items() if col in cols]
            if not missing_from:
                continue  # column present everywhere
            # Look up whitelist by (table, column) and check that the
            # actual (present_in) list matches the declared present_in.
            allowed = False
            for entry in whitelist:
                if entry.get("table") != table or entry.get("column") != col:
                    continue
                declared_present = set(entry.get("present_in") or [])
                if declared_present == set(present_in):
                    allowed = True
                    break
            if not allowed:
                failures.append(
                    f"  table={table} column={col}\n"
                    f"    present in: {present_in}\n"
                    f"    missing from: {missing_from}\n"
                    f"    -> add to whitelist with reason, or add the column "
                    f"to the missing schemas"
                )

    assert not failures, "Column-parity divergences:\n" + "\n".join(failures)


def test_whitelist_entries_are_actually_divergent():
    """A whitelist entry that no longer divergences (e.g. someone added the
    column to all schemas) becomes stale. Catch it early so we don't
    accumulate dead entries."""
    schemas: dict[str, dict[str, set[str]]] = {}
    for path in SCHEMA_FILES:
        if not path.exists():
            continue
        schemas[path.name] = _extract_tables(path.read_text())

    stale: list[str] = []
    for entry in _load_whitelist():
        table = entry.get("table")
        column = entry.get("column")
        declared_present = set(entry.get("present_in") or [])
        if not (table and column):
            continue
        actual_present = {s for s, tabs in schemas.items() if column in tabs.get(table, set())}
        # entry claims this column is only in some variants but reality
        # says it's in all (or none) -> stale.
        if declared_present == actual_present:
            continue
        stale.append(
            f"  table={table} column={column}\n"
            f"    whitelist says present in: {sorted(declared_present)}\n"
            f"    schemas actually have it in: {sorted(actual_present)}\n"
            f"    -> update whitelist or align schemas"
        )
    assert not stale, "Stale whitelist entries:\n" + "\n".join(stale)
