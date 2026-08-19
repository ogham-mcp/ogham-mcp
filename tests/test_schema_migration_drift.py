"""Does a fresh install actually get every feature the migrations shipped?

`ogham init` applies ONE schema file and nothing else -- there is no migration
pass afterwards (see init_wizard.py). So anything a migration adds that was
never backported into the schema files is absent forever on new installs, while
working fine on any database that grew through the migration history. That
asymmetry is invisible to every other test in this suite, because they all run
against a database that has had migrations applied.

It is not hypothetical. `link_memory_entities` (TBU-221) went missing exactly
this way: the entity TABLES were in the schema, the function that populates them
was not, so `store_memory` raised UndefinedFunction on every write, the
exception was swallowed at debug level, and the OKF export's MENTIONS bridge was
silently always empty on fresh installs.

The check: build a database from the schema file, dump it, apply the entire
migration history on top, dump it again, and compare the two dumps object by
object. Migrations are guarded (IF NOT EXISTS / WHERE NOT EXISTS), so a complete
schema file yields an empty diff.

Two things can go wrong, and this test names them separately:

  MISSING -- an object the migrations create that the schema file does not.
             A fresh install lacks the feature entirely.
  CHANGED -- an object both create, with a DIFFERENT definition. A fresh install
             gets an older or divergent version of a function, index, or policy.

CHANGED is the category this test was rebuilt to catch (TBU-228). The previous
version compared catalog *names* -- `proname(identity_args)`, `indexname`,
`conname`, `table.column` with no type -- and reported a clean run while eight
functions, including `hybrid_search_memories` and `match_memories`, differed only
in their bodies. Names were never the problem.

`pg_dump --schema-only` is the comparator rather than a hand-written set of
catalog queries for one reason: it emits whatever is there. The hand-written
version missed RLS policies and comments because nobody thought to query for
them, which is the same failure mode as the drift it was hunting.

This is a RATCHET, not a clean bill of health. `schema_drift_baseline.yaml`
records the drift that already exists so the test fails on NEW loss rather than
on the backlog. Shrinking that file is the goal; growing it needs a very good
reason.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
BASELINE_PATH = Path(__file__).parent / "schema_drift_baseline.yaml"

# All three shipping schemas are checked (TBU-230). Until 2026-08-17 only the
# vanilla one was, which meant a backport could land in one file and silently
# miss two -- and two of the three are the install targets most users actually
# hit.
#
# The Supabase variants qualify their types as `extensions.vector` and run with
# `search_path = public, extensions`, so they need that schema to exist with the
# extensions installed INTO it. The vanilla file expects them in public. That is
# the whole difference; the roles were already handled.
SUPABASE_EXTENSION_SCHEMA = "extensions"

SCHEMAS = {
    "postgres": REPO_ROOT / "sql" / "schema_postgres.sql",
    "supabase_cloud": REPO_ROOT / "sql" / "schema.sql",
    "supabase_selfhost": REPO_ROOT / "sql" / "schema_selfhost_supabase.sql",
}

# Roles the migrations' guarded blocks key off. Vanilla Postgres has none of
# them; Supabase ships all three. See the CREATE ROLE block in the test.
_SUPABASE_ROLES = ("anon", "authenticated", "service_role")

# pg_dump precedes every object with a three-line comment block:
#     --
#     -- Name: <name>; Type: <TYPE>; Schema: <schema>; Owner: <owner>
#     --
# That header is the object boundary, and (name, type) is a stable key across
# two dumps of the same database. Anything before the first header is pg_dump's
# SET preamble, which is identical on both sides and carries no schema content.
_OBJECT_HEADER = re.compile(
    r"^--\s*Name:\s*(?P<name>.+?);\s*Type:\s*(?P<type>.+?);\s*Schema:\s*(?P<schema>.+?);"
    r"\s*Owner:",
    re.MULTILINE,
)

# Lines that differ between two dumps of identical schemas and say nothing about
# the schema. `\restrict`/`\unrestrict` carry a per-invocation random token; the
# header comments carry the pg_dump and server version. The completion trailer
# has no header of its own, so without stripping it the LAST object in the dump
# absorbs it -- and since the last object differs between the two dumps, that
# alone reported a spurious change.
_NOISE_PREFIXES = (
    "\\restrict",
    "\\unrestrict",
    "-- Dumped from",
    "-- Dumped by",
    "-- PostgreSQL database dump complete",
)

# A single-quoted SQL literal, doubled quotes included. Case is preserved inside
# these and folded everywhere else -- see _normalize.
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _pg_dump(url: str) -> str:
    """Schema-only dump. Grants and ownership are deliberately NOT suppressed --
    both dumps come from the same database and role, so they are stable, and
    dropping them would reintroduce the blind spot this test exists to close."""
    result = subprocess.run(
        ["pg_dump", "--schema-only", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # The most likely failure is the version rule: pg_dump refuses to read a
        # server newer than itself. Surface it rather than reporting empty drift.
        raise RuntimeError(f"pg_dump failed: {result.stderr.strip()[:500]}")
    return result.stdout


def _normalize(ddl: str) -> str:
    """Fold the differences that are not schema differences.

    A migration that re-creates a function verbatim but re-indented, or in a
    different letter case, is not drift -- and treating it as drift is worse than
    useless, because a gate that cries wolf on formatting gets baselined into
    silence. So whitespace runs collapse and identifiers fold to lower case.

    Case folding stops at single-quoted literals: `'fresh'` and `'Fresh'` are
    genuinely different values and must still compare unequal. It does NOT stop
    at double-quoted identifiers -- Postgres treats those as case-sensitive, so
    a rename of "myTable" to "MyTable" would slip through. Nothing in this schema
    uses quoted mixed-case identifiers; if that changes, this needs revisiting.
    """
    parts = []
    last = 0
    for literal in _SQL_LITERAL.finditer(ddl):
        parts.append(ddl[last : literal.start()].lower())
        parts.append(literal.group(0))
        last = literal.end()
    parts.append(ddl[last:].lower())
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _parse_objects(dump: str) -> dict[str, str]:
    """Split a dump into {"TYPE: name": normalized DDL}.

    Duplicate keys are joined rather than overwritten: a type/name pair is not
    unique in pg_dump output (two `CREATE POLICY "Tenant scoped access"` blocks
    on different tables share a Type, and function overloads share a Name), and
    silently keeping the last one would hide exactly the drift being hunted.
    """
    headers = list(_OBJECT_HEADER.finditer(dump))
    objects: dict[str, list[str]] = {}
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(dump)
        body = dump[match.end() : end]
        cleaned = _normalize(
            "\n".join(
                line
                for line in body.splitlines()
                # A bare `--` is the opening rule of the NEXT object's header
                # block, so every object but the last one ends with one. Keeping
                # them makes "is this the final object?" part of an object's
                # identity, and the final object changes as soon as migrations
                # add anything after it.
                if line.strip() not in ("", "--") and not line.startswith(_NOISE_PREFIXES)
            )
        )
        key = f"{match.group('type').strip()}: {match.group('name').strip()}"
        objects.setdefault(key, []).append(cleaned)
    return {key: "\n".join(sorted(bodies)) for key, bodies in objects.items()}


def _load_baseline(schema_key: str) -> tuple[set[str], set[str]]:
    """Per-schema debt register.

    Each shipping schema drifts differently -- the vanilla one has no RLS at all,
    the Supabase ones do -- so a single shared list would either mask real drift
    in one file or report phantom drift in another.
    """
    raw = yaml.safe_load(BASELINE_PATH.read_text()) or {}
    known = ((raw.get("known_drift") or {}).get(schema_key)) or {}
    return set(known.get("missing") or []), set(known.get("changed") or [])


@pytest.mark.postgres_integration
@pytest.mark.parametrize("schema_key", sorted(SCHEMAS))
def test_schema_file_is_not_behind_its_migrations(pg_url, schema_key):
    """Applying every migration to a schema-built database must change nothing.

    Creates and drops its own throwaway database, so the shared scratch DB is
    untouched and this is safe to run alongside everything else.
    """
    import psycopg

    from ogham.schema_apply import render_schema_sql

    if shutil.which("pg_dump") is None:
        pytest.skip("pg_dump not on PATH -- install libpq/postgresql-client to run the gate")

    schema_path = SCHEMAS[schema_key]
    is_supabase = schema_key != "postgres"
    scratch_db = f"ogham_drift_{uuid.uuid4().hex[:8]}"
    base_url = pg_url.rsplit("/", 1)[0]
    admin_url = f"{base_url}/postgres"
    drift_url = f"{base_url}/{scratch_db}"

    created_roles: list[str] = []
    with psycopg.connect(admin_url, autocommit=True) as admin:
        # Several migrations wrap their RLS and GRANT blocks in
        # `IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated')`.
        # Roles are cluster-wide, so whether those blocks fire depends on the
        # machine, not the repo -- and when they do not fire, this test reports
        # LESS drift than exists and calls it a pass. Create them so the
        # measurement is the same everywhere, and drop them again below.
        for role in _SUPABASE_ROLES:
            row = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
            if row is None:
                admin.execute(f'CREATE ROLE "{role}" NOLOGIN')  # type: ignore[arg-type]
                created_roles.append(role)
        # psycopg types `execute` for LiteralString; every statement here is
        # built at runtime from repo-controlled files, never user input.
        admin.execute(f'CREATE DATABASE "{scratch_db}"')  # type: ignore[arg-type]
    try:
        schema_sql = render_schema_sql(schema_path.read_text(), 512)
        with psycopg.connect(drift_url, autocommit=True) as conn:
            if is_supabase:
                # The Supabase files qualify their types as `extensions.vector`
                # and set `search_path = public, extensions`. Reproduce that
                # layout: the extensions must be installed INTO that schema, not
                # merely reachable from it. The vanilla file expects them in
                # public, which is the pgvector image's default.
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{SUPABASE_EXTENSION_SCHEMA}"')  # type: ignore[arg-type]
                ext_schema = SUPABASE_EXTENSION_SCHEMA
                for ext in ("vector", "pg_trgm"):
                    conn.execute(  # type: ignore[arg-type]
                        f'CREATE EXTENSION IF NOT EXISTS {ext} WITH SCHEMA "{ext_schema}"'
                    )
                # Both scopes are needed. ALTER DATABASE only takes effect for
                # NEW connections, which is what pg_dump opens; this connection
                # -- the one that applies the schema and every migration -- keeps
                # the search_path it already had unless told otherwise.
                path = f"public, {ext_schema}"
                alter = f'ALTER DATABASE "{scratch_db}" SET search_path = {path}'
                conn.execute(alter)  # type: ignore[arg-type]
                conn.execute(f"SET search_path = {path}")  # type: ignore[arg-type]
            conn.execute(schema_sql)  # type: ignore[arg-type]
            before = _parse_objects(_pg_dump(drift_url))

            migrations = sorted(
                p
                for p in (REPO_ROOT / "sql" / "migrations").glob("*.sql")
                if not p.name.startswith("DANGER_")
            )
            assert migrations, "no migrations found -- path drift?"
            conflicts: list[str] = []
            for migration in migrations:
                # A migration that cannot replay onto a CURRENT schema is not
                # necessarily a fault. Once a function's final form is
                # backported, the migration that first created it tries to
                # CREATE OR REPLACE an older signature, and Postgres refuses
                # ("cannot change return type of existing function") --
                # wiki_topic_search is replaced by 034, so 031 can no longer
                # apply on top of the backported version.
                #
                # That is the schema being AHEAD, the opposite of the drift this
                # test hunts, and it is not a real user path: fresh installs
                # apply the schema and stop; existing installs replay migrations
                # against an OLD schema, in order, which works. So collect these
                # rather than aborting -- the assertion that matters is whether
                # any object appears or changes that the schema did not produce.
                try:
                    conn.execute(migration.read_text())  # type: ignore[arg-type]
                except psycopg.Error as exc:
                    conflicts.append(f"{migration.name}: {str(exc).splitlines()[0]}")
                    # Migration files carry their own BEGIN/COMMIT, so a failure
                    # inside one leaves the session in an aborted transaction and
                    # every later statement dies with InFailedSqlTransaction.
                    # Clear it before moving on, or one conflict masks the rest
                    # of the run.
                    conn.rollback()

            after = _parse_objects(_pg_dump(drift_url))
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            drop = f'DROP DATABASE IF EXISTS "{scratch_db}" WITH (FORCE)'
            admin.execute(drop)  # type: ignore[arg-type]
            # Only the roles this test created, and only after the database
            # holding their grants is gone -- DROP ROLE fails while any grant
            # anywhere still references them.
            for role in created_roles:
                admin.execute(f'DROP ROLE IF EXISTS "{role}"')  # type: ignore[arg-type]

    assert before, "parsed no objects from the schema-built dump -- parser drift?"

    known_missing, known_changed = _load_baseline(schema_key)
    missing = {key for key in after if key not in before}
    changed = {key for key in after if key in before and after[key] != before[key]}

    new_missing = sorted(missing - known_missing)
    new_changed = sorted(changed - known_changed)
    resolved = sorted((known_missing - missing) | (known_changed - changed))

    if resolved:
        print(  # noqa: T201 - surfaced deliberately on an integration run
            "\nschema drift RESOLVED since the baseline was taken -- "
            f"remove these from {BASELINE_PATH.name}:\n"
            + "\n".join(f"  {item}" for item in resolved)
        )
    if conflicts:
        print(  # noqa: T201
            "\nmigrations that could not replay onto the current schema "
            "(schema is AHEAD -- informational, not drift):\n"
            + "\n".join(f"  {c}" for c in conflicts)
        )

    report = []
    if new_missing:
        report.append(
            f"MISSING -- the migrations create these, {schema_path.name} does not.\n"
            "A fresh install lacks the feature entirely:\n"
            + "\n".join(f"    {item}" for item in new_missing)
        )
    if new_changed:
        report.append(
            "CHANGED -- both create these, with DIFFERENT definitions.\n"
            "A fresh install gets a divergent version:\n"
            + "\n".join(f"    {item}" for item in new_changed)
        )
    assert not report, (
        "\n\n".join(report)
        + f"\n\n(schema under test: {schema_path.name})"
        + "\n\nBackport them into all three schema files, or -- if a fresh install is "
        f"genuinely not meant to have them -- add them under known_drift.{schema_key} "
        f"in {BASELINE_PATH.name} with a reason."
    )


def test_baseline_file_is_well_formed():
    """Runs without a database so the baseline cannot rot unnoticed."""
    raw = yaml.safe_load(BASELINE_PATH.read_text()) or {}
    assert "known_drift" in raw, "baseline must have a known_drift mapping"
    assert raw.get("issue"), "baseline must name the issue tracking the backlog"

    # Every shipping schema needs a section, or a file could be added to SCHEMAS
    # and silently run with an empty baseline -- which looks like a clean bill of
    # health rather than an unmeasured file.
    assert set(raw["known_drift"]) == set(SCHEMAS), (
        f"baseline sections {sorted(raw['known_drift'])} do not match the schemas "
        f"under test {sorted(SCHEMAS)}"
    )
    for schema_key, known in raw["known_drift"].items():
        for kind in known or {}:
            assert kind in {"missing", "changed"}, (
                f"unknown drift kind {kind!r} under known_drift.{schema_key}"
            )


def test_dump_parser_splits_on_object_headers():
    """The parser is the instrument. A silent parse failure would report zero
    drift on a database full of it, so exercise it without a database."""
    dump = """SET statement_timeout = 0;

--
-- Name: thing(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.thing() RETURNS void
    LANGUAGE sql
    AS $$ select 1 $$;

--
-- Name: t_pkey; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX t_pkey ON public.t USING btree (id);

--
-- PostgreSQL database dump complete
--
"""
    objects = _parse_objects(dump)
    assert set(objects) == {"FUNCTION: thing()", "INDEX: t_pkey"}
    assert "language sql" in objects["FUNCTION: thing()"]
    # The SET preamble precedes any header and must not be attributed to an object.
    assert "statement_timeout" not in "".join(objects.values())
    # The completion trailer has no header, so it would otherwise be swept into
    # the last object and report it as changed.
    assert "dump complete" not in "".join(objects.values())


def test_dump_parser_folds_formatting_but_not_literals():
    """Re-indenting or re-casing a function is not drift. Changing a string
    value in it is."""
    template = """--
-- Name: thing(); Type: FUNCTION; Schema: public; Owner: -
--

{body}
"""
    tidy = _parse_objects(
        template.format(body="CREATE FUNCTION public.thing() AS $$ SELECT 'fresh' $$;")
    )
    scruffy = _parse_objects(
        template.format(body="create   function public.thing()\n    as $$\n  select 'fresh'\n$$;")
    )
    assert tidy == scruffy

    relabelled = _parse_objects(
        template.format(body="CREATE FUNCTION public.thing() AS $$ SELECT 'Fresh' $$;")
    )
    assert relabelled != tidy, "case inside a literal is a real value change"


def test_dump_parser_keeps_both_bodies_for_a_duplicated_key():
    """Two policies can share a name across tables, and functions overload. If
    the parser overwrote on collision, one of the pair could change unnoticed."""
    dump = """--
-- Name: a Scoped; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Scoped" ON public.a USING (true);

--
-- Name: a Scoped; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Scoped" ON public.b USING (false);
"""
    objects = _parse_objects(dump)
    assert len(objects) == 1
    body = objects["POLICY: a Scoped"]
    assert "public.a" in body and "public.b" in body


def test_dump_parser_is_order_insensitive_for_a_duplicated_key():
    """Two dumps can emit a duplicated key's blocks in either order; that is not
    a change. Without the sort in _parse_objects this test fails."""
    block = """--
-- Name: dup; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Scoped" ON public.{table} USING (true);
"""
    forward = _parse_objects(block.format(table="a") + block.format(table="b"))
    backward = _parse_objects(block.format(table="b") + block.format(table="a"))
    assert forward == backward


def test_dump_parser_ignores_per_invocation_noise():
    """`\\restrict` tokens are random per pg_dump run -- if they survived
    normalization every object would read as changed on every run."""
    template = """--
-- Name: thing(); Type: FUNCTION; Schema: public; Owner: -
--

\\restrict {token}
CREATE FUNCTION public.thing() RETURNS void LANGUAGE sql AS $$ select 1 $$;
\\unrestrict {token}
"""
    first = _parse_objects(template.format(token="AAAAAA"))
    second = _parse_objects(template.format(token="ZZZZZZ"))
    assert first == second
