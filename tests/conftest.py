import os
from pathlib import Path

import pytest


def _destructive_db_safe() -> tuple[bool, str]:
    """Return (allowed, reason). Guard for fixtures that DROP / DELETE.

    Default-deny: only allow destructive fixtures when either
    ``OGHAM_TEST_ALLOW_DESTRUCTIVE=1`` is set explicitly, or
    ``DATABASE_URL`` clearly points at a scratch DB (contains ``scratch``).

    Protects against accidentally running the lifecycle test fixtures
    against a prod / demo DB and wiping triggers, columns, or rows.
    """
    if os.environ.get("OGHAM_TEST_ALLOW_DESTRUCTIVE", "").strip().lower() in ("1", "true", "yes"):
        return True, "OGHAM_TEST_ALLOW_DESTRUCTIVE set"
    url = os.environ.get("DATABASE_URL", "")
    if "scratch" in url.lower():
        return True, "DATABASE_URL contains 'scratch'"
    return (
        False,
        f"refusing destructive fixture: DATABASE_URL={url!r} is not a scratch DB "
        "and OGHAM_TEST_ALLOW_DESTRUCTIVE is not set",
    )


@pytest.fixture(autouse=True)
def _reset_db_backend():
    """Reset the database backend singleton between tests."""
    from ogham.database import _reset_backend

    _reset_backend()
    yield
    _reset_backend()


@pytest.fixture(scope="session", autouse=True)
def _apply_lifecycle_migrations():
    """Ensure lifecycle schema exists on the scratch DB for the run.

    Migrations 025 + 026 are both idempotent. We apply them once per
    test session in order: 025 adds memories.stage columns, then 026
    moves lifecycle state into ``memory_lifecycle`` (dropping the
    memories columns). After this fixture runs, the scratch DB is in the
    post-026 state.

    The ``pg_fresh_db`` fixture drops everything on teardown; tests that
    use it re-apply the migrations explicitly -- so even though this
    session fixture runs first, the tear-down/re-apply dance still
    works.
    """
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return
        allowed, reason = _destructive_db_safe()
        if not allowed:
            # Session-scope fixture can't skip individual tests; just no-op
            # and let per-test guards handle the skip with a clear reason.
            return
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()

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
        col_names = {r[0] if isinstance(r, tuple) else r["column_name"] for r in cols}
        if "stage" not in col_names:
            mig_025 = (
                Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
            )
            backend._execute(mig_025.read_text(), fetch="none")

        # Apply 026.
        mig_026 = (
            Path(__file__).parent.parent / "src/ogham/sql/migrations/026_memory_lifecycle_split.sql"
        )
        backend._execute(mig_026.read_text(), fetch="none")
    except Exception:
        # Tests that need the columns will still skip via _can_connect
        # guards; tests that don't touch Postgres are unaffected.
        pass


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
    backend = get_backend()

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
        return {r[0] if isinstance(r, tuple) else r["column_name"] for r in rows}

    # If memory_lifecycle isn't there yet, apply 025 (if needed) then 026.
    if not _table_exists("memory_lifecycle"):
        if "stage" not in _col_names("memories"):
            mig_025 = (
                Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
            )
            backend._execute(mig_025.read_text(), fetch="none")
        mig_026 = (
            Path(__file__).parent.parent / "src/ogham/sql/migrations/026_memory_lifecycle_split.sql"
        )
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

    backend = get_backend()
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
            return [r[0] if isinstance(r, tuple) else r["column_name"] for r in rows]

        def tables(self):
            rows = self.be._execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                fetch="all",
            )
            return [r[0] if isinstance(r, tuple) else r["table_name"] for r in rows]

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

    _cleanup()
    yield _Harness(backend)
    _cleanup()
