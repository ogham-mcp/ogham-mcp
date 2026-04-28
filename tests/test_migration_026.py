"""Migration 026 dry-run + rollback test.

Verifies (a) 026 forward creates memory_lifecycle, backfills from memories,
drops memories.stage columns; (b) 026 rollback re-adds memories.stage
columns, backfills from memory_lifecycle, drops memory_lifecycle; (c) the
trigger auto-inits a lifecycle row on new memory INSERT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_025 = Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
MIG_026 = Path(__file__).parent.parent / "src/ogham/sql/migrations/026_memory_lifecycle_split.sql"
ROLLBACK_026 = (
    Path(__file__).parent.parent / "src/ogham/sql/migrations/DANGER_026_memory_lifecycle_split.sql"
)


def _can_connect() -> bool:
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()
        backend._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(
        not _can_connect(),
        reason="Postgres backend not configured or unreachable",
    ),
]


def test_migration_026_forward_creates_lifecycle_table_and_backfills(pg_fresh_db):
    """Seed 3 memories with stage='fresh' (025 state), apply 026, verify
    memory_lifecycle has 3 rows, memories.stage column is gone."""
    from ogham.service import store_memory_enriched

    # Ensure we start in the post-025 state: stage columns present.
    pg_fresh_db.apply_sql(MIG_025)

    for content in (
        "row a split test content body",
        "row b split test content body",
        "row c split test content body",
    ):
        store_memory_enriched(content=content, profile="test-025", source="t", tags=[])

    assert pg_fresh_db.count("memories") == 3

    pg_fresh_db.apply_sql(MIG_026)

    # Lifecycle table exists + backfilled.
    lifecycle_cols = pg_fresh_db.column_names("memory_lifecycle")
    assert {"memory_id", "profile", "stage", "stage_entered_at", "updated_at"} <= set(
        lifecycle_cols
    )

    # memories.stage / stage_entered_at are gone.
    memories_cols = pg_fresh_db.column_names("memories")
    assert "stage" not in memories_cols
    assert "stage_entered_at" not in memories_cols

    # Row counts match.
    assert pg_fresh_db.count("memories") == 3
    assert pg_fresh_db.count("memory_lifecycle") == 3


def test_migration_026_trigger_inits_lifecycle_on_new_memory(pg_fresh_db, pg_client):
    """After migration 026, a new memory auto-gets a lifecycle row at stage='fresh'."""
    from ogham.service import store_memory_enriched

    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)

    result = store_memory_enriched(
        content="trigger test memory content body",
        profile="test-025",
        source="t",
        tags=[],
    )
    mid = str(result["id"])

    row = pg_client.fetchone(
        "SELECT stage FROM memory_lifecycle WHERE memory_id = %(id)s::uuid",
        {"id": mid},
    )
    assert row is not None
    assert row["stage"] == "fresh"


def test_migration_026_rollback_restores_columns(pg_fresh_db):
    """Apply 026, write 2 memories, rollback, verify stage columns back + data preserved."""
    from ogham.service import store_memory_enriched

    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)

    # Write 2 memories (via trigger they get lifecycle rows at stage='fresh').
    for content in ("rollback a memory body content", "rollback b memory body content"):
        store_memory_enriched(content=content, profile="test-025", source="t", tags=[])

    # Rollback 026.
    pg_fresh_db.apply_rollback(ROLLBACK_026)

    # memories.stage is back.
    memories_cols = pg_fresh_db.column_names("memories")
    assert "stage" in memories_cols
    assert "stage_entered_at" in memories_cols

    # memory_lifecycle is gone.
    tables = pg_fresh_db.tables()
    assert "memory_lifecycle" not in tables

    # Row count preserved.
    assert pg_fresh_db.count("memories") == 2
