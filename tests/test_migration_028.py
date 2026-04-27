"""Migration 028 dry-run + rollback test.

Verifies (a) 028 forward creates topic_summaries + topic_summary_sources with
the expected columns and indexes; (b) 028 rollback DROPs both tables; (c) the
reverse-cascade index on topic_summary_sources.memory_id is present (essential
for the T1.4 invalidation path).

Phase 0: RED placeholder -- migration SQL not yet written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_025 = Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
MIG_026 = Path(__file__).parent.parent / "src/ogham/sql/migrations/026_memory_lifecycle_split.sql"
MIG_028 = Path(__file__).parent.parent / "src/ogham/sql/migrations/028_topic_summaries.sql"
ROLLBACK_028 = (
    Path(__file__).parent.parent / "src/ogham/sql/migrations/028_topic_summaries_rollback.sql"
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


def _apply_baseline(pg_fresh_db):
    """Bring DB to the post-026 state that 028 builds on."""
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)


def test_migration_028_forward_creates_summary_tables(pg_fresh_db):
    """After 028, topic_summaries + topic_summary_sources tables exist with the spec columns."""
    _apply_baseline(pg_fresh_db)

    pg_fresh_db.apply_sql(MIG_028)

    tables = pg_fresh_db.tables()
    assert "topic_summaries" in tables
    assert "topic_summary_sources" in tables

    summary_cols = set(pg_fresh_db.column_names("topic_summaries"))
    required = {
        "id",
        "topic_key",
        "profile_id",
        "content",
        "embedding",
        "source_count",
        "source_cursor",
        "source_hash",
        "token_count",
        "importance",
        "model_used",
        "version",
        "status",
        "created_at",
        "updated_at",
        "stale_reason",
    }
    missing = required - summary_cols
    assert not missing, f"topic_summaries missing columns: {missing}"

    junction_cols = set(pg_fresh_db.column_names("topic_summary_sources"))
    assert {"summary_id", "memory_id"} <= junction_cols


def test_migration_028_creates_reverse_cascade_index(pg_fresh_db, pg_client):
    """The memory_id index on topic_summary_sources is the reverse-cascade path.

    Without it, invalidation on memory write/update/delete is O(N) over all
    summaries; with it, O(log N). This is the single most load-bearing index
    for the T1.4 hot path.
    """
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    row = pg_client.fetchone(
        """
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'topic_summary_sources'
          AND indexdef ILIKE '%(memory_id)%'
        """
    )
    assert row is not None, "reverse-cascade index on topic_summary_sources.memory_id missing"


def test_migration_028_unique_constraint_on_profile_topic(pg_fresh_db, pg_client):
    """UNIQUE(profile_id, topic_key) enforces one live summary per topic per profile."""
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    row = pg_client.fetchone(
        """
        SELECT 1 FROM pg_constraint
        WHERE conname LIKE '%topic_summaries%'
          AND contype = 'u'
        """
    )
    assert row is not None, "UNIQUE(profile_id, topic_key) constraint missing"


def test_migration_028_hnsw_index_is_dim_agnostic_and_partial_on_fresh(pg_fresh_db, pg_client):
    """Perf contract for the HNSW index.

    (1) Uses plain vector_cosine_ops, not a hardcoded halfvec(512) expression
        cast. Matches migration 023's "drop hardcoded halfvec casts" rewrite:
        fresh installs are dim-agnostic, auto-sizing to whatever EMBEDDING_DIM
        the column was created at. Operators who want the halfvec memory
        optimization opt in via migration 021's rebuild_hnsw=on path.

    (2) Partial on status='fresh'. This is the load-bearing guard against
        the HNSW-HOT-update bloat that migration 026 learned on memories.stage.
        A nightly sweep flipping status fresh->stale must not rewrite the
        tuple into the index; the partial predicate makes stale flips
        tombstone-out cheaply.
    """
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    row = pg_client.fetchone(
        """
        SELECT indexdef FROM pg_indexes
        WHERE tablename = 'topic_summaries'
          AND indexname = 'topic_summaries_embedding_hnsw_idx'
        """
    )
    assert row is not None, "HNSW index missing on topic_summaries.embedding"
    indexdef = row["indexdef"]

    # Dim-agnostic default per migration 023. The halfvec opt-in belongs
    # in a future 028-equivalent-of-021, not baked into the forward migration.
    assert "halfvec" not in indexdef, (
        f"HNSW must be dim-agnostic (plain vector_cosine_ops), not halfvec; got: {indexdef}"
    )
    assert "vector_cosine_ops" in indexdef, (
        f"HNSW must use vector_cosine_ops for dim-agnostic default; got: {indexdef}"
    )
    # Partial-on-fresh is the real perf guard.
    assert "WHERE" in indexdef.upper() and "fresh" in indexdef.lower(), (
        f"HNSW must be partial on status='fresh'; got: {indexdef}"
    )


def test_migration_028_updated_at_trigger_bumps_on_update(pg_fresh_db, pg_client):
    """Correctness contract: UPDATE on any column bumps updated_at.

    Without this, the nightly stale-sweep predicate
    `updated_at < NOW() - INTERVAL '30 days'` runs against a timestamp
    that never moves -- every row stays stale forever after the first
    sweep, or never goes stale at all depending on the direction of
    the bug.
    """
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    pg_client.execute(
        """
        INSERT INTO topic_summaries
            (topic_key, profile_id, content, source_count, source_hash, model_used)
        VALUES ('trigger_test', 'test-028', 'body', 1,
                '\\x00000000000000000000000000000000000000000000000000000000deadbeef', 'test')
        """
    )
    before = pg_client.fetchone(
        "SELECT updated_at FROM topic_summaries WHERE topic_key = 'trigger_test'"
    )["updated_at"]

    pg_client.execute(
        "UPDATE topic_summaries SET content = 'changed' WHERE topic_key = 'trigger_test'"
    )
    after = pg_client.fetchone(
        "SELECT updated_at FROM topic_summaries WHERE topic_key = 'trigger_test'"
    )["updated_at"]

    assert after > before, f"updated_at must advance on UPDATE; before={before} after={after}"


def test_migration_028_importance_is_double_precision(pg_fresh_db, pg_client):
    """Parity with memories.importance which is double precision (float8).

    If we ship this as real (float4), cross-table comparisons and
    ranking math silently lose precision at the boundary.
    """
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    row = pg_client.fetchone(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'topic_summaries' AND column_name = 'importance'
        """
    )
    assert row is not None
    assert row["data_type"] == "double precision", (
        f"importance must be double precision (float8) to match memories; got {row['data_type']}"
    )


def test_migration_028_rollback_drops_tables(pg_fresh_db):
    """Rollback restores the pre-028 state: both new tables are gone."""
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_028)

    assert "topic_summaries" in pg_fresh_db.tables()

    pg_fresh_db.apply_sql(ROLLBACK_028)

    tables = pg_fresh_db.tables()
    assert "topic_summaries" not in tables
    assert "topic_summary_sources" not in tables
