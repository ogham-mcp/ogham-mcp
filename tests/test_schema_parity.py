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
# All three schema variants must stay in lockstep with the Python backend's
# SQL function calls (TBU-149: schema_selfhost_supabase.sql was 2 params
# behind on hybrid_search_memories -- missing query_entity_tags +
# recency_decay -- which broke self-hosted Supabase deploys with PGRST202
# param-count mismatch. Fixed; all three schemas now checked).
_FUNCTION_PARITY_SCHEMAS = SCHEMA_FILES
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
    required_tables = [
        "memories",
        "memory_relationships",
        "profile_settings",
        "entity_edges",
        "entity_edge_predicates",
        "entity_aliases",
    ]
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql = schema.read_text().lower()
        for table in required_tables:
            assert table in sql, f"{table} missing from {schema.name}"


def test_predicate_vocab_seed_present():
    """All schema files must seed the 16 v1 predicate names (6 inverse pairs
    + 4 standalone), with 'SUPERSEDES' absent per TBU-109 regression guard."""
    from ogham.entity_graph import V1_PREDICATES

    required_predicates = [
        "DEPENDS_ON",
        "DEPENDED_ON_BY",
        "OWNS",
        "OWNED_BY",
        "ASSIGNED_TO",
        "HAS_ASSIGNEE",
        "DECIDED",
        "MENTIONS",
        "BLOCKS",
        "BLOCKED_BY",
        "PART_OF",
        "CONTAINS",
        "SUPPORTS",
        "CONTRADICTS",
        "EVOLVED_INTO",
        "RELATED_TO",
    ]
    # Regression guard: keep V1_PREDICATES aligned with the schema-parity list
    # (TBU-114 amendment -- prevents silent drift between the Python constant
    # and the SQL seed). Migration 042 remains the ultimate source of truth
    # (checked by the loop below); this assertion just chains V1_PREDICATES
    # to it so an edit to either list alone fails here.
    assert set(required_predicates) == V1_PREDICATES, (
        f"V1_PREDICATES drift: schema parity list = {sorted(required_predicates)!r}, "
        f"V1_PREDICATES = {sorted(V1_PREDICATES)!r}. Update both in lockstep."
    )
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        sql = schema.read_text()
        for pred in required_predicates:
            assert f"'{pred}'" in sql, f"predicate '{pred}' missing from {schema.name}"
        assert "'SUPERSEDES'" not in sql, (
            f"SUPERSEDES must not be seeded (dropped per TBU-109) in {schema.name}"
        )


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


def test_no_hardcoded_vector_dim_in_shipping_schemas():
    """Chronic-drift prevention (TBU-159): task #188 and v0.6 #149 both landed
    partial dim-hardcode fixes, leaving pin sites behind. This test refuses any
    commit that re-introduces a literal vector(N) or halfvec(N) in the shipping
    schema files. New code must use vector(:embedding_dim) / halfvec(:embedding_dim)
    -- the psql -v variable placeholder (Design Council Option A, 2026-07-02).

    Migration files under sql/migrations/ are separately scoped -- restricted
    to the top-level shipping schema files (SCHEMA_FILES, same 3-schema list
    the column/function parity tests above already use).
    """
    pattern = re.compile(r"\b(vector|halfvec)\(\d+\)")
    offenders = []
    for schema in SCHEMA_FILES:
        if not schema.exists():
            continue
        for i, line in enumerate(schema.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{schema}:{i}: {line.strip()}")
    assert not offenders, (
        "Re-introduced hardcoded vector/halfvec dim in shipping schema:\n"
        + "\n".join(offenders)
        + "\n\nUse vector(:embedding_dim) / halfvec(:embedding_dim) instead. See TBU-159."
    )


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


_PREDICATE_ROW_URI_RE = (
    r"\('{pred}',.*?'(https://ogham-mcp\.dev/vocab#[^']*)',"
    r"\s*(?:'(https://schema\.org/[^']*)'|NULL),\s*NULL\)"
)


def test_predicate_uri_seed_present():
    """Every schema file must seed ogham_uri for all 16 predicates and the
    five verified schema.org alignments (TBU-129). Guards SQL-side drift.

    Uses a per-row regex (not a file-wide substring check) so a
    transposition -- e.g. OWNS and OWNED_BY swapping their schema.org URIs,
    or any ogham_uri landing on the wrong predicate's row -- fails the test
    instead of passing because the string merely appears somewhere in the
    file.
    """
    schema_dir = Path(__file__).resolve().parents[1] / "sql"
    schemas = [
        schema_dir / "schema.sql",
        schema_dir / "schema_selfhost_supabase.sql",
        schema_dir / "schema_postgres.sql",
    ]
    predicates = [
        "DEPENDS_ON",
        "DEPENDED_ON_BY",
        "OWNS",
        "OWNED_BY",
        "ASSIGNED_TO",
        "HAS_ASSIGNEE",
        "DECIDED",
        "MENTIONS",
        "BLOCKS",
        "BLOCKED_BY",
        "PART_OF",
        "CONTAINS",
        "SUPPORTS",
        "CONTRADICTS",
        "EVOLVED_INTO",
        "RELATED_TO",
    ]
    schema_org = {
        "OWNS": "https://schema.org/owns",
        "OWNED_BY": "https://schema.org/owner",
        "MENTIONS": "https://schema.org/mentions",
        "PART_OF": "https://schema.org/isPartOf",
        "CONTAINS": "https://schema.org/hasPart",
    }
    for schema in schemas:
        sql = schema.read_text()
        # the columns must exist on the table
        assert "ogham_uri" in sql and "schema_org_uri" in sql and "iirds_uri" in sql, (
            f"URI columns missing from entity_edge_predicates in {schema.name}"
        )
        for pred in predicates:
            pattern = _PREDICATE_ROW_URI_RE.format(pred=re.escape(pred))
            match = re.search(pattern, sql)
            assert match, f"seed row for '{pred}' not found (or malformed) in {schema.name}"
            ogham_uri, row_schema_org = match.group(1), match.group(2)
            assert ogham_uri == f"https://ogham-mcp.dev/vocab#{pred}", (
                f"'{pred}' row has ogham_uri={ogham_uri!r} in {schema.name}"
            )
            expected_schema_org = schema_org.get(pred)
            assert row_schema_org == expected_schema_org, (
                f"'{pred}' row has schema_org_uri={row_schema_org!r}, "
                f"expected {expected_schema_org!r} in {schema.name}"
            )


def test_derived_from_column_and_index_present():
    """entity_edges must have derived_from jsonb + GIN index in all 3 schemas (TBU-124)."""
    from pathlib import Path

    schema_dir = Path(__file__).resolve().parents[1] / "sql"
    for name in ("schema.sql", "schema_selfhost_supabase.sql", "schema_postgres.sql"):
        sql = (schema_dir / name).read_text()
        assert "derived_from jsonb" in sql, f"derived_from column missing from {name}"
        assert "entity_edges_derived_from_gin" in sql, f"GIN index missing from {name}"
        assert "USING gin (derived_from)" in sql, f"GIN index def missing from {name}"


# Migrations whose functions must also exist in every fresh-install schema.
# A fresh install applies a schema file and NOTHING else -- init_wizard.py has
# no migration pass -- so a function that lives only in a migration is absent
# forever on new installs.
_BACKPORTED_FUNCTION_MIGRATIONS = ["036_entities_backfill.sql"]

_CREATE_FUNCTION_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:public\.)?([a-z_][a-z0-9_]*)\s*\(",
    re.IGNORECASE,
)


def test_migration_functions_are_backported_into_every_schema():
    """Every function a backported migration defines must exist in all three
    fresh-install schemas.

    This caught a real, shipped gap. Migration 036 defines three functions;
    `refresh_entity_temporal_span` and `spread_entity_activation_memories` were
    copied into all three schema files, and `link_memory_entities` was not. On a
    fresh install the entity TABLES existed but the function that populates them
    did not, so `service.store_memory` raised UndefinedFunction on every write,
    the exception was swallowed at debug level, `memory_entities` never
    populated -- and the OKF export's MENTIONS bridge was silently always empty.

    Found by building a scratch DB from schema_postgres.sql and running an
    export, not by any test. Hence this one.
    """
    for migration_name in _BACKPORTED_FUNCTION_MIGRATIONS:
        migration = REPO_ROOT / "sql" / "migrations" / migration_name
        expected = set(_CREATE_FUNCTION_RE.findall(migration.read_text()))
        assert expected, f"{migration_name}: no CREATE FUNCTION found -- regex drift?"

        for schema in SCHEMA_FILES:
            present = set(_CREATE_FUNCTION_RE.findall(schema.read_text()))
            missing = expected - present
            assert not missing, (
                f"{schema.name} is missing {sorted(missing)} from {migration_name}. "
                "A fresh install applies the schema file only, so these never arrive."
            )
