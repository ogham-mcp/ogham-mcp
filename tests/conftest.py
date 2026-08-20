import os
from pathlib import Path
from typing import Any, cast

import pytest


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def pytest_configure(config):
    """Default local collection to hermetic unit-test config.

    Several legacy integration modules perform reachability checks at import
    time, before autouse fixtures can isolate environment variables. If a
    developer's local Ogham config points at Postgres, ordinary ``pytest`` can
    spend minutes trying to connect to that database during collection. Keep
    default local runs hermetic, while preserving explicit scratch Postgres and
    external Supabase/Ollama opt-in paths.
    """
    if _truthy_env("OGHAM_RUN_EXTERNAL_INTEGRATION") or _truthy_env("OGHAM_TEST_ALLOW_DESTRUCTIVE"):
        return

    url = os.environ.get("DATABASE_URL", "")
    if "scratch" in url.lower():
        return

    os.environ.setdefault("DATABASE_BACKEND", "supabase")
    os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "fake-key")
    os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
    os.environ.setdefault("DEFAULT_PROFILE", "default")


def _destructive_db_safe() -> tuple[bool, str]:
    """Return (allowed, reason). Guard for fixtures that DROP / DELETE.

    Default-deny: only allow destructive fixtures when either
    ``OGHAM_TEST_ALLOW_DESTRUCTIVE=1`` is set explicitly, or
    ``DATABASE_URL`` clearly points at a scratch DB (contains ``scratch``).

    Protects against accidentally running the lifecycle test fixtures
    against a prod / demo DB and wiping triggers, columns, or rows.
    """
    if _truthy_env("OGHAM_TEST_ALLOW_DESTRUCTIVE"):
        return True, "OGHAM_TEST_ALLOW_DESTRUCTIVE set"
    url = os.environ.get("DATABASE_URL", "")
    if "scratch" in url.lower():
        return True, "DATABASE_URL contains 'scratch'"
    return (
        False,
        f"refusing destructive fixture: DATABASE_URL={url!r} is not a scratch DB "
        "and OGHAM_TEST_ALLOW_DESTRUCTIVE is not set",
    )


def _postgres_integration_db_safe() -> tuple[bool, str]:
    """Return whether live Postgres integration tests may run.

    Unlike unit tests, postgres integration tests use the configured
    ``Settings`` object so a developer's ~/.ogham/config.env is visible.
    We still require a scratch database name/URL, or explicit opt-in, so
    `pytest` never exercises a personal/prod Ogham database by accident.
    """
    if _truthy_env("OGHAM_TEST_ALLOW_DESTRUCTIVE"):
        return True, "OGHAM_TEST_ALLOW_DESTRUCTIVE set"

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        try:
            from ogham.config import settings

            url = settings.database_url or ""
        except Exception:
            url = ""

    if "scratch" in url.lower():
        return True, "database URL contains 'scratch'"
    return (
        False,
        f"Postgres integration tests require a scratch DATABASE_URL; got {url!r}",
    )


@pytest.fixture(autouse=True)
def _isolated_unit_environment(monkeypatch, request):
    """Keep unit tests independent from a developer's local Ogham env."""
    is_external_integration = request.node.get_closest_marker(
        "integration"
    ) or request.node.get_closest_marker("postgres_integration")

    if request.node.get_closest_marker("postgres_integration"):
        allowed, reason = _postgres_integration_db_safe()
        if not allowed:
            pytest.skip(reason)

    if not is_external_integration:
        monkeypatch.setenv("DATABASE_BACKEND", "supabase")
        monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "fake-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.config import settings
    from ogham.database import _reset_backend, _reset_entity_graph

    settings._reset()
    _reset_backend()
    _reset_entity_graph()
    yield
    _reset_entity_graph()
    _reset_backend()
    settings._reset()


@pytest.fixture(scope="session", autouse=True)
def _ensure_standard_postgres_test_schema():
    """Ensure scratch Postgres has the standard pgvector test schema.

    Local Postgres integration tests should run against a predictable
    scratch database, not whatever schema happens to live in a developer's
    personal Ogham DB. For an empty scratch DB, apply schema_postgres.sql.
    For an older scratch DB, apply the small idempotent baseline migrations
    needed by current tests.

    The ``pg_fresh_db`` fixture drops everything on teardown; tests that
    use it re-apply the migrations explicitly -- so even though this
    session fixture runs first, the tear-down/re-apply dance still
    works.
    """
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return
        allowed, reason = _postgres_integration_db_safe()
        if not allowed:
            # Session-scope fixture can't skip individual tests; just no-op
            # and let per-test guards handle the skip with a clear reason.
            return
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()
        repo_root = Path(__file__).parent.parent

        tables = backend._execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
            fetch="all",
        )
        table_names = {str(r["table_name"]) for r in tables}
        if "memories" not in table_names:
            from ogham.schema_apply import render_schema_sql

            schema = repo_root / "sql/schema_postgres.sql"
            schema_sql = render_schema_sql(schema.read_text(), settings.embedding_dim)
            backend._execute(schema_sql, fetch="none")
            return

        # Current profile stats tests require migration 022's additive
        # relationship/tagging/decay counters.
        mig_022 = repo_root / "sql/migrations/022_profile_health_stats.sql"
        backend._execute(mig_022.read_text(), fetch="none")

        # Has 026 been applied? (i.e. memory_lifecycle exists)
        tables = backend._execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'memory_lifecycle'",
            fetch="all",
        )
        if tables:
            return

        # Does memories.stage exist? If not, apply 025 first.
        cols = backend._execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'memories'",
            fetch="all",
        )
        col_names = {str(r["column_name"]) for r in cols}
        if "stage" not in col_names:
            mig_025 = repo_root / "sql/migrations/025_memory_lifecycle.sql"
            backend._execute(mig_025.read_text(), fetch="none")

        # Apply 026.
        mig_026 = repo_root / "sql/migrations/026_memory_lifecycle_split.sql"
        backend._execute(mig_026.read_text(), fetch="none")
    except Exception:
        # Tests that need the columns will still skip via _can_connect
        # guards; tests that don't touch Postgres are unaffected.
        pass


@pytest.fixture
def pg_url() -> str:
    """Raw Postgres URL for the shared scratch DB.

    Deliberately not named ``pg_fresh_db`` -- that fixture has a different
    shape (a migration harness object, not a URL) and destructive teardown
    (drops entities/memory_entities). Shared by the entity-graph
    ``postgres_integration`` test modules (store_triple, query_join), which
    each open their own ``ConnectionPool`` against this URL and never
    drop/truncate -- they use uuid-prefixed entity names so runs never
    collide, regardless of order.
    """
    url = os.environ.get("DATABASE_URL", "")
    if "scratch" not in url.lower():
        pytest.skip("DATABASE_URL must point at a scratch Postgres database")
    return url


@pytest.fixture
def pg_client():
    """Raw-SQL helper for integration tests.

    Thin wrapper over ``PostgresBackend._execute`` that offers
    ``.execute(sql, params)`` (no fetch) and ``.fetchone(sql, params)``.
    Params must be dicts -- backend uses psycopg named placeholders
    (``%(name)s``), not positional ``%s``.
    """
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()

    class _Client:
        def execute(self, sql, params=None):
            backend._execute(sql, params, fetch="none")

        def fetchone(self, sql, params=None):
            return backend._execute(sql, params, fetch="one")

    return _Client()


@pytest.fixture
def pg_test_profile():
    """Dedicated profile for lifecycle tests; cleaned before and after.

    Idempotently ensures migrations 025 + 026 have been applied -- if a
    prior test ran ``pg_fresh_db`` and dropped everything on teardown,
    this fixture reapplies them in order so downstream tests can rely on
    the ``memory_lifecycle`` table + trigger existing.

    Refuses to run against a non-scratch DB (see ``_destructive_db_safe``).
    """
    allowed, reason = _destructive_db_safe()
    if not allowed:
        pytest.skip(reason)

    from ogham.database import get_backend

    profile = "test-lifecycle-parity"
    backend = cast(Any, get_backend())

    def _table_exists(name):
        rows = backend._execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %(t)s",
            {"t": name},
            fetch="all",
        )
        return bool(rows)

    def _col_names(table):
        rows = backend._execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %(t)s",
            {"t": table},
            fetch="all",
        )
        return {str(r["column_name"]) for r in rows}

    # If memory_lifecycle isn't there yet, apply 025 (if needed) then 026.
    if not _table_exists("memory_lifecycle"):
        if "stage" not in _col_names("memories"):
            mig_025 = Path(__file__).parent.parent / "sql/migrations/025_memory_lifecycle.sql"
            backend._execute(mig_025.read_text(), fetch="none")
        mig_026 = Path(__file__).parent.parent / "sql/migrations/026_memory_lifecycle_split.sql"
        backend._execute(mig_026.read_text(), fetch="none")

    backend._execute(
        "DELETE FROM memories WHERE profile = %(p)s",
        {"p": profile},
        fetch="none",
    )
    yield profile
    backend._execute(
        "DELETE FROM memories WHERE profile = %(p)s",
        {"p": profile},
        fetch="none",
    )


#: Postgres never reclaims a dropped column's attnum slot, and the hard ceiling
#: is 1600 per table. ``pg_fresh_db`` drops the migration-025 lifecycle columns
#: on every setup AND teardown, so a long-lived scratch database burns slots on
#: `memories` steadily -- 1,580 of 1,600 were consumed on this machine before
#: anyone noticed (2026-08-20).
#:
#: The failure mode is the problem: at the ceiling EVERY Postgres test dies with
#: ``tables can have at most 1600 columns``, an error that names no fixture, no
#: migration and no column, and that no amount of reading the diff explains. The
#: database has to be recreated; nothing in the repo is wrong.
_ATTNUM_CEILING = 1600
_ATTNUM_WARN_AT = 1400


def _check_attnum_headroom(backend) -> None:
    """Fail loudly, and with the remedy, before the ceiling makes it cryptic."""
    try:
        row = backend._execute(
            "SELECT count(*) AS used, count(*) FILTER (WHERE a.attisdropped) AS dropped "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = 'memories' AND a.attnum > 0",
            {},
            fetch="one",
        )
    except Exception:
        return  # never let the guard itself break a run
    if not row:
        return
    used = int(row.get("used") or 0)
    dropped = int(row.get("dropped") or 0)
    if used < _ATTNUM_WARN_AT:
        return
    pytest.fail(
        f"scratch database is out of column headroom: `memories` has used "
        f"{used}/{_ATTNUM_CEILING} attribute slots, {dropped} of them from dropped "
        f"columns that Postgres never reclaims.\n\n"
        f"Nothing in the repo is wrong. This fixture drops the migration-025 "
        f"lifecycle columns on every setup and teardown, and the slots accumulate "
        f"across runs.\n\n"
        f"Recreate the scratch database:\n"
        f"  docker exec -e PGPASSWORD=ogham_dev ogham-postgres psql -U ogham -d postgres \\\n"
        f"    -c 'DROP DATABASE IF EXISTS ogham_scratch WITH (FORCE)' \\\n"
        f"    -c 'CREATE DATABASE ogham_scratch'\n"
        f"  then re-create the vector, pg_trgm and uuid-ossp extensions in it."
    )


@pytest.fixture
def pg_fresh_db():
    """Migration harness fixture.

    Yields a helper object exposing ``count``, ``apply_sql``, and
    ``column_names`` against the shared Postgres backend. Scoped to the
    ``test-025`` profile for any row-level cleanup. On setup and teardown
    this fixture deletes ``test-025`` memories and drops the lifecycle
    columns (IF EXISTS), so repeated runs start clean.

    Refuses to run against a non-scratch DB (see ``_destructive_db_safe``).
    DROP TABLE / DROP COLUMN against a prod DB would wipe live state.
    """
    allowed, reason = _destructive_db_safe()
    if not allowed:
        pytest.skip(reason)

    from ogham.database import get_backend

    backend = cast(Any, get_backend())
    _check_attnum_headroom(backend)
    profile = "test-025"

    class _Harness:
        def __init__(self, be):
            self.be = be

        def count(self, table):
            return self.be._execute(
                f"SELECT count(*) FROM {table} WHERE profile = %(p)s",
                {"p": profile},
                fetch="scalar",
            )

        def apply_sql(self, path):
            sql = Path(path).read_text()
            self.be._execute(sql, fetch="none")

        def apply_rollback(self, path):
            sql = Path(path).read_text()
            self.be._execute(
                "SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';\n" + sql,
                fetch="none",
            )

        def column_names(self, table):
            rows = self.be._execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %(t)s",
                {"t": table},
                fetch="all",
            )
            return [str(r["column_name"]) for r in rows]

        def tables(self):
            rows = self.be._execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                fetch="all",
            )
            return [str(r["table_name"]) for r in rows]

    def _cleanup():
        # Row-level cleanup for test-025 profile (shared by 025 + 026 tests).
        backend._execute(
            "DELETE FROM memories WHERE profile = %(p)s",
            {"p": profile},
            fetch="none",
        )

        # Cleanup migration 026 artifacts (triggers, function, table).
        backend._execute(
            "DROP TRIGGER IF EXISTS memories_init_lifecycle ON memories",
            fetch="none",
        )
        backend._execute(
            "DROP TRIGGER IF EXISTS memories_sync_lifecycle_profile ON memories",
            fetch="none",
        )
        backend._execute("DROP FUNCTION IF EXISTS init_memory_lifecycle()", fetch="none")
        backend._execute("DROP FUNCTION IF EXISTS sync_memory_lifecycle_profile()", fetch="none")
        backend._execute("DROP TABLE IF EXISTS memory_lifecycle", fetch="none")

        # Cleanup 025 artifacts on memories.
        backend._execute("DROP INDEX IF EXISTS memories_stage_idx", fetch="none")
        backend._execute(
            "ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_stage_valid",
            fetch="none",
        )
        backend._execute(
            "ALTER TABLE memories "
            "DROP COLUMN IF EXISTS stage, "
            "DROP COLUMN IF EXISTS stage_entered_at",
            fetch="none",
        )

        # Cleanup migration 036 artifacts (v0.14 entities backfill).
        # Drop in dependency order so FK constraints clear cleanly.
        backend._execute(
            "DROP FUNCTION IF EXISTS link_memory_entities(uuid, text, text[]) CASCADE",
            fetch="none",
        )
        backend._execute(
            "DROP FUNCTION IF EXISTS spread_entity_activation_memories"
            "(text[], text, int, float, float, int) CASCADE",
            fetch="none",
        )
        backend._execute(
            "DROP FUNCTION IF EXISTS refresh_entity_temporal_span(bigint) CASCADE",
            fetch="none",
        )
        # entity_edges/entity_aliases go too, and not merely for tidiness:
        # DROP TABLE entities CASCADE drops the FK CONSTRAINT on entity_edges
        # but leaves its ROWS, so re-applying 041 (CREATE TABLE IF NOT EXISTS)
        # no-ops over a table full of orphans pointing at deleted entities.
        # That state is unreachable in a healthy database -- ON DELETE CASCADE
        # prevents it -- and it made list_edges return edges whose subject had
        # no entity row.
        backend._execute("DROP TABLE IF EXISTS entity_aliases CASCADE", fetch="none")
        backend._execute("DROP TABLE IF EXISTS entity_edges CASCADE", fetch="none")
        backend._execute("DROP TABLE IF EXISTS memory_entities CASCADE", fetch="none")
        backend._execute("DROP TABLE IF EXISTS entities CASCADE", fetch="none")

    def _restore():
        """Put back everything _cleanup() removed (TBU-222).

        _cleanup() runs on setup too -- the migration tests need these artifacts
        ABSENT so they can watch a migration create them -- so restoration must
        happen only on teardown.

        Without this, the fixture left the shared scratch database permanently
        stripped: `_ensure_standard_postgres_test_schema` is session-scoped and
        cannot re-run mid-suite, and pytest collects alphabetically, so
        test_migration_036.py ran before every entity-dependent test and left
        19 of them failing with `relation "entities" does not exist`. The damage
        also persisted across runs, so a test that passed in isolation failed on
        the next invocation.

        Applied in migration order; all are guarded (IF NOT EXISTS /
        WHERE NOT EXISTS) so re-application is a no-op when nothing was dropped.

        025/026 are deliberately NOT restored, even though _cleanup() removes
        them. Every one of the 19 failures this fixture caused was
        `relation "entities" does not exist` -- none were lifecycle -- so
        restoring them buys nothing, and it costs something real: re-adding
        `memories.stage` and `stage_entered_at` after each drop permanently
        consumes two attnums, because dropped columns count against Postgres's
        1600-column limit forever and VACUUM FULL does NOT reclaim them (only
        recreating the table does). The scratch database used for this work hit
        1580 dropped columns on `memories` and started failing with
        TooManyColumns -- which then aborted this very restore loop before it
        reached 036, silently bringing the original bug back. Restoring only the
        entity layer halves that burn and still fixes what was broken.
        """
        repo_root = Path(__file__).parent.parent
        for name in (
            "036_entities_backfill",
            "041_entity_edges",
            "042_entity_edge_predicates",
            "043_entity_aliases",
            "045_predicate_uris",
            "046_edge_provenance",
        ):
            path = repo_root / "sql" / "migrations" / f"{name}.sql"
            backend._execute(path.read_text(), fetch="none")

    _cleanup()
    yield _Harness(backend)
    _cleanup()
    _restore()
