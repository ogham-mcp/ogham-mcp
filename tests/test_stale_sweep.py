"""Phase 7 tests — nightly stale sweep for topic_summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_025 = Path(__file__).parent.parent / "src/ogham/sql/migrations/025_memory_lifecycle.sql"
MIG_026 = Path(__file__).parent.parent / "src/ogham/sql/migrations/026_memory_lifecycle_split.sql"
MIG_028 = Path(__file__).parent.parent / "src/ogham/sql/migrations/028_topic_summaries.sql"
MIG_030 = (
    Path(__file__).parent.parent / "src/ogham/sql/migrations/030_topic_summaries_dim_agnostic.sql"
)
MIG_031 = Path(__file__).parent.parent / "src/ogham/sql/migrations/031_wiki_rpc_functions.sql"
ROLLBACK_028 = (
    Path(__file__).parent.parent / "src/ogham/sql/migrations/DANGER_028_topic_summaries.sql"
)


def _can_connect() -> bool:
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        PostgresBackend()._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(not _can_connect(), reason="Postgres backend not configured"),
]


def _apply_028(pg_fresh_db):
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)
    pg_fresh_db.apply_rollback(ROLLBACK_028)
    pg_fresh_db.apply_sql(MIG_028)
    pg_fresh_db.apply_sql(MIG_030)
    pg_fresh_db.apply_sql(MIG_031)


def _seed_summary_with_age(
    *,
    profile: str = "test-025",
    topic: str = "old-topic",
    age_days: int,
) -> str:
    """Insert a summary row then backdate its updated_at via direct SQL.

    ALTER TRIGGER ... DISABLE -> direct update -> ALTER ... ENABLE so
    the BEFORE UPDATE trigger doesn't clobber our backdated timestamp
    with now().
    """
    from ogham.database import get_backend

    backend = get_backend()
    row = backend._execute(
        """INSERT INTO topic_summaries
             (topic_key, profile_id, content, source_count, source_hash, model_used)
           VALUES (%(topic)s, %(profile)s, 'body', 1,
                   '\\x00000000000000000000000000000000000000000000000000000000deadbeef',
                   'test')
           RETURNING id::text AS id""",
        {"topic": topic, "profile": profile},
        fetch="one",
    )
    sid = row["id"]

    backend._execute(
        "ALTER TABLE topic_summaries DISABLE TRIGGER topic_summaries_bump_updated_at",
        {},
        fetch="none",
    )
    backend._execute(
        """UPDATE topic_summaries
              SET updated_at = now() - make_interval(days => %(d)s)
            WHERE id = %(id)s::uuid""",
        {"d": age_days, "id": sid},
        fetch="none",
    )
    backend._execute(
        "ALTER TABLE topic_summaries ENABLE TRIGGER topic_summaries_bump_updated_at",
        {},
        fetch="none",
    )
    return sid


def test_sweep_flips_old_fresh_to_stale(pg_fresh_db):
    _apply_028(pg_fresh_db)
    sid = _seed_summary_with_age(age_days=45)

    from ogham.topic_summaries import sweep_stale_summaries

    flipped = sweep_stale_summaries("test-025", older_than_days=30)
    assert flipped == 1

    from ogham.database import get_backend

    row = get_backend()._execute(
        "SELECT status, stale_reason FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": sid},
        fetch="one",
    )
    assert row["status"] == "stale"
    assert "nightly sweep" in (row["stale_reason"] or "")


def test_sweep_leaves_young_rows_alone(pg_fresh_db):
    _apply_028(pg_fresh_db)
    sid = _seed_summary_with_age(age_days=10)

    from ogham.topic_summaries import sweep_stale_summaries

    flipped = sweep_stale_summaries("test-025", older_than_days=30)
    assert flipped == 0

    from ogham.database import get_backend

    row = get_backend()._execute(
        "SELECT status FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": sid},
        fetch="one",
    )
    assert row["status"] == "fresh"


def test_sweep_is_profile_scoped(pg_fresh_db):
    """Sweeping profile A must not touch profile B's summaries."""
    _apply_028(pg_fresh_db)
    sa = _seed_summary_with_age(profile="test-025", topic="t-a", age_days=45)
    sb = _seed_summary_with_age(profile="test-025b", topic="t-b", age_days=45)

    from ogham.database import get_backend
    from ogham.topic_summaries import sweep_stale_summaries

    assert sweep_stale_summaries("test-025", older_than_days=30) == 1

    status_a = get_backend()._execute(
        "SELECT status FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": sa},
        fetch="one",
    )
    status_b = get_backend()._execute(
        "SELECT status FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": sb},
        fetch="one",
    )
    assert status_a["status"] == "stale"
    assert status_b["status"] == "fresh", "other profile must be untouched"

    # Cleanup -- test-025b leaks outside pg_fresh_db scope.
    get_backend()._execute(
        "DELETE FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": sb},
        fetch="none",
    )


def test_sweep_skips_already_stale_rows(pg_fresh_db):
    """Idempotent: repeated sweeps don't thrash updated_at on stale rows."""
    _apply_028(pg_fresh_db)
    _seed_summary_with_age(age_days=45)

    from ogham.topic_summaries import sweep_stale_summaries

    first = sweep_stale_summaries("test-025", older_than_days=30)
    second = sweep_stale_summaries("test-025", older_than_days=30)
    assert first == 1
    assert second == 0, "second sweep must find no fresh+old rows"


def test_session_start_hook_schedules_sweep(pg_fresh_db):
    """session_start hook must submit the stale sweep to the lifecycle executor."""
    import ogham.hooks as hooks

    _apply_028(pg_fresh_db)
    _seed_summary_with_age(age_days=45)

    # Patch the search side so we don't touch embeddings / real queries.
    # hooks.py imports these inside the function body, so patch at source.
    from unittest.mock import patch

    with (
        patch("ogham.database.hybrid_search_memories", return_value=[]),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        hooks.session_start(cwd="/tmp/test", profile="test-025", limit=1)

    # Flush the lifecycle executor so the scheduled sweep actually runs.
    from ogham.lifecycle_executor import flush

    flush(timeout=5.0)

    from ogham.database import get_backend

    row = get_backend()._execute(
        "SELECT status FROM topic_summaries "
        "WHERE topic_key = 'old-topic' AND profile_id = 'test-025'",
        {},
        fetch="one",
    )
    assert row is not None
    assert row["status"] == "stale"
