"""Regression test for TBU-163 -- FORCE ROW LEVEL SECURITY locking out
non-superuser owners on non-Supabase (vanilla Postgres) installs.

Migrations 041 (entity_edges), 042 (entity_edge_predicates), and 043
(entity_aliases) unconditionally run ``ALTER TABLE <t> FORCE ROW LEVEL
SECURITY`` and only guard the ``CREATE POLICY`` behind an ``anon``-role
existence check. On a vanilla Postgres install (no ``anon`` role -- e.g.
self-hosted Postgres/Neon, not Supabase), the policy is skipped but FORCE
still applies. FORCE ROW LEVEL SECURITY subjects even the table OWNER to
RLS, so "FORCE + zero policies" means deny-all for everyone, including a
non-superuser app role that owns the tables. The 16-row predicate seed in
042 still inserts fine (the migration itself runs as the privileged role
applying it), but a subsequent ``SELECT count(*) FROM
entity_edge_predicates`` from that same owning role returns 0.

The fix moves ``FORCE ROW LEVEL SECURITY`` inside the anon-guarded DO
block, so FORCE is only applied when the ``anon`` role exists (i.e. only
on Supabase-shaped installs that also get the Deny-anon policy).

This test builds a throwaway database + non-superuser, non-BYPASSRLS
role, applies the real migration files 041->042->043 while connected AS
that role (so it owns the resulting tables -- ownership is what FORCE
targets), then reconnects as that role and asserts the seeded predicates
and a round-tripped edge are visible.

Also covers migration 044 (``044_unforce_rls_non_supabase.sql``), the
dynamic self-heal for tables already locked by a pre-fix migration run
(e.g. migration 036, shipped in v0.14, which cannot simply be edited in
place). ``test_migration_044_self_heals_forced_table`` reproduces the
post-036-style locked state directly (ENABLE + FORCE with no policy, on
a table that already has a row -- mirroring 036's seed-then-force
order), confirms the owner is locked out (RED), applies 044, and
confirms the owner regains read + write access (GREEN).

Run against postgres-scratch (see Makefile's ``test-postgres`` target):

    DATABASE_URL="postgresql://ogham:ogham@localhost:5433/ogham_scratch" \
        uv run pytest tests/test_migration_rls_non_superuser.py -v \
        -m postgres_integration

Requires the connecting user for ``DATABASE_URL`` to be a superuser (able
to CREATE DATABASE / CREATE ROLE) -- true for the ``ogham`` bootstrap user
in the postgres-scratch Docker container.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

pytestmark = pytest.mark.postgres_integration

MIGRATIONS_DIR = Path(__file__).parent.parent / "sql" / "migrations"
MIGRATION_041 = MIGRATIONS_DIR / "041_entity_edges.sql"
MIGRATION_042 = MIGRATIONS_DIR / "042_entity_edge_predicates.sql"
MIGRATION_043 = MIGRATIONS_DIR / "043_entity_aliases.sql"
MIGRATION_044 = MIGRATIONS_DIR / "044_unforce_rls_non_supabase.sql"

EXPECTED_PREDICATE_COUNT = 16


def _url_with_db(url: str, dbname: str) -> str:
    """Swap the path component of a Postgres URL to point at ``dbname``."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{dbname}", "", ""))


def _url_with_role(url: str, role: str, password: str) -> str:
    """Swap the userinfo component of a Postgres URL to ``role``/``password``."""
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{role}:{password}@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


@pytest.fixture
def tbu163_fixture(pg_url):
    """Ephemeral database + non-superuser owner role for the TBU-163 repro.

    Yields a connection URL (as the non-superuser app role, pointed at the
    ephemeral database) after applying migrations 041-043 + a minimal
    ``entities`` prerequisite table while connected AS that role, so the
    role owns every resulting table -- ownership is what ``FORCE ROW LEVEL
    SECURITY`` targets.

    Skips (rather than failing) if the cluster unexpectedly has ``anon``
    or ``service_role`` roles -- this repro is specifically about the
    non-Supabase (no such roles) path; a Supabase-shaped cluster would
    exercise a different code path entirely.
    """
    uid = uuid.uuid4().hex[:10]
    db_name = f"tbu163_scratch_{uid}"
    role_name = f"tbu163_app_{uid}"
    role_password = uid  # ephemeral, scoped to this fixture's lifetime only

    admin_conn = psycopg.connect(pg_url, autocommit=True)
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'service_role')")
            existing = {row[0] for row in cur.fetchall()}
            if existing:
                pytest.skip(
                    f"Supabase roles {existing} present in this cluster -- "
                    "TBU-163 repro targets the non-Supabase (no anon/"
                    "service_role) path only"
                )

            # .encode() -- psycopg3's Query type requires a LiteralString for
            # the no-params execute() overload; these are f-strings built
            # from a uuid slug, not literals, so bytes sidesteps that check
            # (same pattern as PostgresBackend._execute). DDL also doesn't
            # accept bind params for identifiers, so string interpolation is
            # how these have to be built; role_password/db_name/role_name
            # are all hex uuid slugs (alphanumeric only) so this is safe.
            cur.execute(f'CREATE DATABASE "{db_name}"'.encode())
            cur.execute(
                (
                    f"CREATE ROLE \"{role_name}\" LOGIN PASSWORD '{role_password}' "
                    "NOSUPERUSER NOBYPASSRLS"
                ).encode()
            )

        # Hand ownership of the new database's public schema to the app
        # role so everything it creates there (entities + 041/042/043's
        # tables) is owned by it, not by the superuser bootstrap user.
        db_admin_conn = psycopg.connect(_url_with_db(pg_url, db_name), autocommit=True)
        try:
            with db_admin_conn.cursor() as cur:
                cur.execute(f'ALTER SCHEMA public OWNER TO "{role_name}"'.encode())
        finally:
            db_admin_conn.close()

        app_url = _url_with_role(_url_with_db(pg_url, db_name), role_name, role_password)

        app_conn = psycopg.connect(app_url)
        try:
            with app_conn.cursor() as cur:
                # Minimal prerequisite -- migrations 041/043 FK to entities(id).
                cur.execute(
                    "CREATE TABLE entities ("
                    "    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
                    "    canonical_name text NOT NULL,"
                    "    entity_type text NOT NULL"
                    ")"
                )
                for migration_path in (MIGRATION_041, MIGRATION_042, MIGRATION_043):
                    cur.execute(migration_path.read_text().encode())
            app_conn.commit()
        finally:
            app_conn.close()

        yield app_url
    finally:
        # Terminate any lingering backends before dropping the database --
        # a failed assertion above could leave app_conn's transaction open.
        with admin_conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"'.encode())
            cur.execute(f'DROP ROLE IF EXISTS "{role_name}"'.encode())
        admin_conn.close()


def test_non_superuser_owner_can_read_seeded_predicates(tbu163_fixture):
    """Core TBU-163 assertion: owner reads all 16 seeded predicates.

    Fails with a count of 0 against the unfixed migrations (FORCE RLS +
    no anon-role policy = deny-all, including the owner). Passes once
    FORCE is moved inside the anon-guarded block, since a cluster without
    an ``anon`` role then never forces RLS at all.
    """
    app_url = tbu163_fixture
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entity_edge_predicates")
        row = cur.fetchone()
        assert row is not None
        (count,) = row
        assert count == EXPECTED_PREDICATE_COUNT, (
            f"expected {EXPECTED_PREDICATE_COUNT} seeded predicates visible to the "
            f"owning non-superuser role, got {count} -- FORCE ROW LEVEL SECURITY is "
            "likely still applied unconditionally (TBU-163 regression)"
        )


def test_non_superuser_owner_can_round_trip_an_edge(tbu163_fixture):
    """Owner can INSERT into entity_edges and read the row back.

    Exercises the same lockout on entity_edges itself (not just the
    predicate seed table), and confirms writes aren't silently accepted
    while reads are denied (FORCE RLS with no policy denies both).
    """
    app_url = tbu163_fixture
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO entities(canonical_name, entity_type) VALUES "
            "('AuthService', 'service'), ('LoginModule', 'module') RETURNING id"
        )
        subject_id, object_id = (row[0] for row in cur.fetchall())

        cur.execute(
            "INSERT INTO entity_edges(subject_id, predicate, object_id, profile) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (subject_id, "DEPENDS_ON", object_id, "tbu163-test"),
        )
        inserted = cur.fetchone()
        assert inserted is not None
        (edge_id,) = inserted
        conn.commit()

        cur.execute(
            "SELECT subject_id, predicate, object_id FROM entity_edges WHERE id = %s",
            (edge_id,),
        )
        row = cur.fetchone()
        assert row == (subject_id, "DEPENDS_ON", object_id), (
            "owning non-superuser role could not read back its own inserted edge -- "
            "FORCE ROW LEVEL SECURITY is likely still applied unconditionally "
            "(TBU-163 regression)"
        )


def test_migration_044_self_heals_forced_table(tbu163_fixture):
    """Migration 044 rescues a table already locked by a pre-fix run.

    Reproduces the post-036-style state directly: seed a row, THEN
    ENABLE + FORCE RLS with no policy (mirroring 036's seed-then-force
    order, and the pre-fix 041-043 behaviour before this same TBU-163 fix
    landed). Confirms the owner is locked out (RED, count 0) even though
    it owns the table and a row already exists -- then applies 044 and
    confirms the owner regains both read and write access (GREEN).
    """
    app_url = tbu163_fixture

    # Seed a row, then force RLS with no policy -- the locked state 044
    # exists to heal.
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO entities(canonical_name, entity_type) VALUES ('LockedEntity', 'demo')"
        )
        cur.execute("ALTER TABLE entities ENABLE ROW LEVEL SECURITY")
        cur.execute("ALTER TABLE entities FORCE ROW LEVEL SECURITY")
        conn.commit()

    # RED: owner is locked out despite owning the table and a row existing.
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities")
        row = cur.fetchone()
        assert row is not None
        (count,) = row
        assert count == 0, (
            f"expected the FORCE-RLS-with-no-policy setup to lock out the owner "
            f"(count 0), got {count} -- test setup did not reproduce the pre-044 "
            "lockout state"
        )

    # Apply the self-heal migration as the (locked-out) owning role --
    # ALTER TABLE ... NO FORCE only requires table ownership, not bypassing
    # RLS, so the owner can run this even while locked out of its data.
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute(MIGRATION_044.read_text().encode())
        conn.commit()

    # GREEN: owner reads the pre-existing row and can insert a new one.
    with psycopg.connect(app_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities")
        row = cur.fetchone()
        assert row is not None
        (count,) = row
        assert count == 1, (
            f"expected the pre-existing row visible after migration 044's "
            f"self-heal, got count {count} -- TBU-163 lockout not healed"
        )

        cur.execute(
            "INSERT INTO entities(canonical_name, entity_type) VALUES "
            "('PostHealEntity', 'demo') RETURNING id"
        )
        inserted = cur.fetchone()
        assert inserted is not None, (
            "owner could not INSERT after migration 044's self-heal -- TBU-163 lockout not healed"
        )
        conn.commit()
