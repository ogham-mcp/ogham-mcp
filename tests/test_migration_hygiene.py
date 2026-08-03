"""Static hygiene checks on sql/migrations (TBU-196 items A and D).

Written as a pytest test rather than a shell gate on purpose: the Quality
workflow already runs the non-integration suite, so this reaches CI and every
contributor's PR with no new YAML, and -- unlike shell embedded in a Makefile
recipe -- it is itself testable.

Each rule is a real incident, not a hypothetical:

* Rollback naming -- v0.12.0: `028_topic_summaries_rollback.sql` matched the
  upgrade path's `*.sql` glob, sorted straight after its own forward migration,
  and DROPped the table mid-upgrade. Caught between tag and publish.
* Replay-safe seeds -- v0.17.0: migration 042's 5-column seed violated the
  `ogham_uri NOT NULL` that 045 added when replayed against a current schema,
  because NOT NULL is enforced before ON CONFLICT resolves. Caught by CI
  pre-publish, by luck of ordering.
* Destructive DDL in a forward migration -- same class: a forward migration
  runs against live databases.

The rules are pure functions so the NEGATIVE tests at the bottom can prove each
one actually rejects bad input. On 2026-07-27 seven guards in the release
Makefile turned out to have been silently inert; a guard never seen to fail is
indistinguishable from a comment.
"""

import re
from pathlib import Path

import pytest

MIGRATIONS = Path(__file__).resolve().parent.parent / "sql" / "migrations"

# upgrade.sh applies `find "$MIGRATIONS_DIR" -maxdepth 1 -name "*.sql" | sort`,
# so anything at the top level of this directory WILL be applied, in sort order.
FORWARD = sorted(p for p in MIGRATIONS.glob("*.sql") if p.is_file())

DESTRUCTIVE = re.compile(
    r"^\s*(drop\s+(table|column|schema|type|index)|truncate|delete\s+from)\b",
    re.I | re.M,
)


# --- rules (pure, so they can be tested both ways) ------------------------


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"--[^\n]*", "", text)


def strip_function_bodies(text: str) -> str:
    """Drop dollar-quoted blocks before scanning for migration-time statements.

    `DELETE FROM t WHERE id = x` inside a PL/pgSQL function is runtime logic
    against one row; the same line in the migration script runs now, against
    everything. Only the second is a hazard -- conflating them made this guard
    reject 031 and 033, which are both fine.
    """
    return re.sub(r"\$([A-Za-z_]*)\$.*?\$\1\$", " ", text, flags=re.S)


def migration_level_sql(text: str) -> str:
    return strip_function_bodies(strip_comments(text))


def naming_violations(filename: str) -> list[str]:
    """NNN_ or NNN<letter>_ -- the letter suffix is a deliberate interstitial.

    Ordering is what you would hope: upgrade.sh sorts lexically, the comparison
    against a different number resolves on the digits, and '_' (0x5F) sorts
    before 'a' (0x61) -- so 008_x < 008a_y < 009_z. An interstitial does run
    directly after its base number.
    """
    out = []
    if not re.match(r"^\d{3}[a-z]?_", filename):
        out.append(f"{filename} is in the applied path but is not NNN_-prefixed")
    if "rollback" in filename.lower() or filename.upper().startswith("DANGER"):
        out.append(f"{filename} looks like a rollback but sits in the applied path")
    return out


def destructive_violations(sql: str) -> list[str]:
    body = migration_level_sql(sql)
    hits = [m.group(0).strip() for m in DESTRUCTIVE.finditer(body)]
    return [
        h
        for h in hits
        if "if exists" not in body[body.find(h) : body.find(h) + 200].lower()
        and "confirm_rollback" not in body.lower()
    ]


# DDL that errors on a second run unless guarded. Postgres offers
# IF NOT EXISTS for some of these and not for others, so the accepted guard
# differs: a preceding DROP ... IF EXISTS is the idiom for triggers and for
# indexes being redefined, and a DO $$ ... $$ block covers types and
# constraints, which have no IF NOT EXISTS form at all.
NON_IDEMPOTENT = [
    ("CREATE TABLE", re.compile(r"^\s*create\s+table\s+(?!if\s+not\s+exists)", re.I | re.M)),
    ("ADD COLUMN", re.compile(r"\badd\s+column\s+(?!if\s+not\s+exists)", re.I)),
    (
        "CREATE INDEX",
        re.compile(
            r"^\s*create\s+(?:unique\s+)?index\s+(?:concurrently\s+)?(?!if\s+not\s+exists)",
            re.I | re.M,
        ),
    ),
    ("CREATE TRIGGER", re.compile(r"^\s*create\s+trigger\s+(\w+)", re.I | re.M)),
    ("CREATE TYPE", re.compile(r"^\s*create\s+type\s+(\w+)", re.I | re.M)),
    ("ADD CONSTRAINT", re.compile(r"\badd\s+constraint\s+(\w+)", re.I)),
]


def idempotency_violations(sql: str) -> list[str]:
    """Flag DDL that would error if the migration were applied twice.

    Every migration in the tree today is already safe -- five use the
    drop-then-create idiom and the rest use IF NOT EXISTS. This rule exists to
    catch the first one that is not, since a migration that cannot be replayed
    turns a partial failure into a manual repair job.
    """
    body = migration_level_sql(sql)
    out = []
    for label, pattern in NON_IDEMPOTENT:
        for match in pattern.finditer(body):
            name = match.group(1) if match.groups() else None
            preceding = body[: match.start()].lower()
            # A DROP ... IF EXISTS for this object earlier in the file makes the
            # create replay-safe. Fall back to any DROP IF EXISTS when the
            # statement has no captured name (CREATE TABLE / ADD COLUMN forms).
            if name and re.search(
                rf"drop\s+\w+\s+if\s+exists\s+{re.escape(name.lower())}\b", preceding
            ):
                continue
            if not name and "if exists" in preceding:
                continue
            out.append(f"{label} without IF NOT EXISTS or a preceding DROP ... IF EXISTS")
    return out


def seed_violations(sql: str) -> list[str]:
    body = migration_level_sql(sql)
    if not re.search(r"^\s*insert\s+into\b", body, re.I | re.M):
        return []
    guarded = (
        re.search(r"on\s+conflict", body, re.I)
        or re.search(r"where\s+not\s+exists", body, re.I)
        or re.search(r"insert\s+.*\bselect\b.*\bwhere\b", body, re.I | re.S)
    )
    return [] if guarded else ["unguarded INSERT -- not safe to replay"]


# --- applied to the real migrations ---------------------------------------


def test_migrations_directory_is_discoverable():
    """Guard the guard: if the glob breaks, every check below silently passes."""
    assert MIGRATIONS.is_dir(), f"{MIGRATIONS} missing"
    assert len(FORWARD) >= 30, f"only {len(FORWARD)} forward migrations found -- glob wrong?"


@pytest.mark.parametrize("path", FORWARD, ids=lambda p: p.name)
def test_real_migration_naming(path: Path):
    assert not naming_violations(path.name), naming_violations(path.name)


@pytest.mark.parametrize("path", FORWARD, ids=lambda p: p.name)
def test_real_migration_has_no_destructive_ddl(path: Path):
    v = destructive_violations(path.read_text())
    assert not v, f"{path.name}: {v}. Use IF EXISTS, or move it to a DANGER_ rollback."


@pytest.mark.parametrize("path", FORWARD, ids=lambda p: p.name)
def test_real_migration_ddl_is_replay_safe(path: Path):
    v = idempotency_violations(path.read_text())
    assert not v, (
        f"{path.name}: {v}. Use IF NOT EXISTS where Postgres offers it, a "
        "preceding DROP ... IF EXISTS for triggers and redefined indexes, or a "
        "DO $$ ... $$ guard for types and constraints."
    )


@pytest.mark.parametrize("path", FORWARD, ids=lambda p: p.name)
def test_real_migration_seeds_are_replay_safe(path: Path):
    v = seed_violations(path.read_text())
    assert not v, f"{path.name}: {v}. Add ON CONFLICT DO NOTHING or WHERE NOT EXISTS."


def test_migration_numbers_are_unique():
    seen: dict[str, str] = {}
    dupes = []
    for path in FORWARD:
        num = path.name[:3]
        if num in seen:
            dupes.append(f"{num}: {seen[num]} and {path.name}")
        seen[num] = path.name
    assert not dupes, f"duplicate migration numbers (apply order undefined): {dupes}"


# --- NEGATIVE tests: prove each rule can actually fail ---------------------


def test_rule_rejects_rollback_in_applied_path():
    """The literal v0.12.0 incident."""
    assert naming_violations("028_topic_summaries_rollback.sql")
    assert naming_violations("DANGER_028_topic_summaries.sql")


def test_rule_rejects_unnumbered_file():
    assert naming_violations("fix_the_thing.sql")
    assert naming_violations("v2_reindex.sql")


def test_rule_accepts_lettered_interstitial():
    """008a_ccf_search.sql is a real, deliberately interstitial migration."""
    assert not naming_violations("008a_ccf_search.sql")


def test_lettered_interstitial_sorts_where_you_expect():
    """upgrade.sh applies `find | sort`, so pin the order an interstitial gets."""
    assert sorted(["009_c.sql", "008a_b.sql", "008_a.sql"]) == [
        "008_a.sql",
        "008a_b.sql",
        "009_c.sql",
    ]


def test_rule_accepts_a_well_formed_name():
    assert not naming_violations("046_edge_provenance.sql")


def test_rule_rejects_destructive_ddl():
    assert destructive_violations("DROP TABLE memories;")
    assert destructive_violations("TRUNCATE memories;")
    assert destructive_violations("DELETE FROM memories;")


def test_rule_allows_if_exists_and_rollback_guards():
    assert not destructive_violations("DROP TABLE IF EXISTS scratch_tmp;")
    assert not destructive_violations(
        "SELECT current_setting('ogham.confirm_rollback');\nDROP TABLE topic_summaries;"
    )


def test_rule_ignores_destructive_sql_inside_a_function_body():
    """The false positive that 031 and 033 exposed."""
    sql = """
    CREATE FUNCTION upsert_summary() RETURNS void AS $$
    BEGIN
        DELETE FROM topic_summary_sources WHERE summary_id = upserted.id;
    END;
    $$ LANGUAGE plpgsql;
    """
    assert not destructive_violations(sql)


def test_rule_rejects_unguarded_seed():
    """The shape of the 042 replay failure."""
    assert seed_violations("INSERT INTO entity_edge_predicates(predicate) VALUES ('OWNS');")


def test_rule_accepts_guarded_seeds():
    assert not seed_violations("INSERT INTO t(a) VALUES ('x') ON CONFLICT (a) DO NOTHING;")
    assert not seed_violations("INSERT INTO t(a) SELECT 'x' WHERE NOT EXISTS (SELECT 1 FROM t);")


def test_rule_ignores_inserts_inside_function_bodies():
    sql = """
    CREATE FUNCTION f() RETURNS void AS $$
    BEGIN
        INSERT INTO audit_log(op) VALUES ('x');
    END;
    $$ LANGUAGE plpgsql;
    """
    assert not seed_violations(sql)


def test_rule_rejects_unguarded_idempotency_hazards():
    assert idempotency_violations("CREATE TABLE memories (id uuid);")
    assert idempotency_violations("ALTER TABLE memories ADD COLUMN foo text;")
    assert idempotency_violations("CREATE INDEX memories_foo_idx ON memories(foo);")
    assert idempotency_violations("CREATE TRIGGER t AFTER INSERT ON memories EXECUTE f();")
    assert idempotency_violations("CREATE TYPE mood AS ENUM ('a');")
    assert idempotency_violations("ALTER TABLE t ADD CONSTRAINT c CHECK (x > 0);")


def test_rule_accepts_if_not_exists_forms():
    assert not idempotency_violations("CREATE TABLE IF NOT EXISTS memories (id uuid);")
    assert not idempotency_violations("ALTER TABLE m ADD COLUMN IF NOT EXISTS foo text;")
    assert not idempotency_violations("CREATE INDEX IF NOT EXISTS i ON m(foo);")


def test_rule_accepts_the_drop_then_create_idiom():
    """What 013, 015, 023, 026 and 028 actually do -- all safe."""
    assert not idempotency_violations(
        "DROP TRIGGER IF EXISTS memories_init_lifecycle ON memories;\n"
        "CREATE TRIGGER memories_init_lifecycle AFTER INSERT ON memories EXECUTE f();"
    )
    assert not idempotency_violations(
        "DROP INDEX IF EXISTS memories_embedding_idx;\n"
        "CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding);"
    )


def test_rule_does_not_credit_an_unrelated_drop():
    """A DROP for a DIFFERENT object must not excuse this create."""
    assert idempotency_violations(
        "DROP TRIGGER IF EXISTS some_other_trigger ON memories;\n"
        "CREATE TRIGGER memories_init_lifecycle AFTER INSERT ON memories EXECUTE f();"
    )


def test_rule_ignores_ddl_inside_function_bodies():
    sql = """
    CREATE FUNCTION f() RETURNS void AS $$
    BEGIN
        CREATE TABLE scratch (id int);
    END;
    $$ LANGUAGE plpgsql;
    """
    assert not idempotency_violations(sql)
