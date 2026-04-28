"""Migration 025 dry-run + rollback test.

Proves we can apply the forward migration, then the rollback, and end
up at the same schema state we started with. No data loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG = Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
ROLLBACK = Path(__file__).parent.parent / "src/ogham/sql/migrations/DANGER_025_memory_lifecycle.sql"


def _can_connect() -> bool:
    """Check if Postgres backend is configured and reachable."""
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


def test_migration_025_forward_then_rollback_no_data_loss(pg_fresh_db):
    """Seed 3 memories, apply 025 forward, verify columns + counts,
    apply rollback, verify columns gone + counts preserved."""
    from ogham.service import store_memory_enriched

    for content in ("row a migration test", "row b migration test", "row c migration test"):
        store_memory_enriched(
            content=content,
            profile="test-025",
            source="t",
            tags=[],
        )

    before = pg_fresh_db.count("memories")
    assert before == 3

    pg_fresh_db.apply_sql(MIG)
    cols = pg_fresh_db.column_names("memories")
    assert "stage" in cols
    assert "stage_entered_at" in cols
    assert pg_fresh_db.count("memories") == 3

    pg_fresh_db.apply_rollback(ROLLBACK)
    cols_after = pg_fresh_db.column_names("memories")
    assert "stage" not in cols_after
    assert "stage_entered_at" not in cols_after
    assert pg_fresh_db.count("memories") == 3
