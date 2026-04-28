"""Phase 2 tests for src/ogham/topic_summaries.py.

Covers the pure schema / CRUD / cascade layer. No LLM calls.
"""

from __future__ import annotations

import hashlib
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


def _apply_028(pg_fresh_db):
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)
    # Rollback-then-apply guarantees a clean topic_summaries regardless
    # of what the previous test in this file left behind. pg_fresh_db
    # only scrubs memories + 025/026 state; 028 tables aren't in its
    # cleanup path.
    pg_fresh_db.apply_rollback(ROLLBACK_028)
    pg_fresh_db.apply_sql(MIG_028)
    pg_fresh_db.apply_sql(MIG_030)
    pg_fresh_db.apply_sql(MIG_031)


def _seed_memories(n: int = 3, profile: str = "test-025") -> list[str]:
    """Seed N memories via direct SQL.

    Bypasses store_memory_enriched so the tests don't need the embedding
    provider chain (Gemini SDK, OpenAI, etc.) to be installed. We only
    need rows to exist in memories so the topic_summary_sources FK
    CASCADE works -- actual embeddings are irrelevant to Phase 2.
    """
    from ogham.database import get_backend

    backend = get_backend()
    rows = backend._execute(
        """INSERT INTO memories (content, profile, source, tags)
           SELECT 'seed content row ' || i::text, %(profile)s, 't', ARRAY['ogham-memory']
             FROM generate_series(1, %(n)s) AS i
           RETURNING id::text AS id""",
        {"n": n, "profile": profile},
        fetch="all",
    )
    return [r["id"] for r in rows]


# ---------- compute_source_hash ----------


def test_compute_source_hash_is_sorted_and_deterministic():
    """Hash must be stable under permutation of input ids."""
    from ogham.topic_summaries import compute_source_hash

    ids = ["3f5a1b2c-...", "aa11bb22-...", "00112233-..."]
    h1 = compute_source_hash(ids)
    h2 = compute_source_hash(list(reversed(ids)))
    assert h1 == h2, "hash must be permutation-invariant (sorted before hashing)"


def test_compute_source_hash_matches_manual_sha256():
    """Hash is sha256 over sorted memory ids joined by a delimiter we control."""
    from ogham.topic_summaries import compute_source_hash

    ids = ["zzz", "aaa", "mmm"]
    expected = hashlib.sha256(b"aaa\nmmm\nzzz").digest()
    assert compute_source_hash(ids) == expected


def test_compute_source_hash_empty_is_well_defined():
    """Empty source set returns a stable, non-error hash (hash of empty bytes)."""
    from ogham.topic_summaries import compute_source_hash

    h = compute_source_hash([])
    assert h == hashlib.sha256(b"").digest()


# ---------- upsert_summary ----------


def test_upsert_summary_inserts_new_row(pg_fresh_db, pg_client):
    """First write creates a row with fresh status + version=1 + source rows."""
    _apply_028(pg_fresh_db)
    mem_ids = _seed_memories(3)

    from ogham.topic_summaries import upsert_summary

    summary = upsert_summary(
        profile="test-025",
        topic_key="quantum",
        content="Quantum computing summary body",
        embedding=[0.1] * 512,
        source_memory_ids=mem_ids,
        model_used="test-model",
        token_count=42,
        importance=0.8,
    )
    assert summary["topic_key"] == "quantum"
    assert summary["status"] == "fresh"
    assert summary["version"] == 1
    assert summary["source_count"] == 3
    assert summary["token_count"] == 42

    row = pg_client.fetchone(
        "SELECT count(*) AS n FROM topic_summary_sources WHERE summary_id = %(id)s::uuid",
        {"id": str(summary["id"])},
    )
    assert row["n"] == 3


def test_upsert_summary_updates_existing_row_and_replaces_sources(pg_fresh_db, pg_client):
    """Second write for same (profile, topic_key) updates in place, replaces junction rows."""
    _apply_028(pg_fresh_db)
    mem_ids_v1 = _seed_memories(3)
    mem_ids_v2 = _seed_memories(2)

    from ogham.topic_summaries import upsert_summary

    s1 = upsert_summary(
        profile="test-025",
        topic_key="quantum",
        content="v1 body",
        embedding=[0.1] * 512,
        source_memory_ids=mem_ids_v1,
        model_used="test-model",
    )
    s2 = upsert_summary(
        profile="test-025",
        topic_key="quantum",
        content="v2 body",
        embedding=[0.2] * 512,
        source_memory_ids=mem_ids_v2,
        model_used="test-model",
    )
    assert s1["id"] == s2["id"], "same (profile, topic) collapses to same row"
    assert s2["version"] == 2
    assert s2["source_count"] == 2
    assert s2["content"] == "v2 body"

    # Junction rows are replaced, not appended -- only v2 sources survive.
    from ogham.database import get_backend

    rows = get_backend()._execute(
        "SELECT memory_id::text AS mid FROM topic_summary_sources WHERE summary_id = %(id)s::uuid",
        {"id": str(s2["id"])},
        fetch="all",
    )
    stored_ids = {r["mid"] for r in rows}
    assert stored_ids == set(mem_ids_v2)


def test_upsert_summary_sets_source_hash_and_cursor(pg_fresh_db):
    """source_hash + source_cursor are derived from the passed memory ids."""
    _apply_028(pg_fresh_db)
    mem_ids = _seed_memories(4)

    from ogham.topic_summaries import compute_source_hash, upsert_summary

    summary = upsert_summary(
        profile="test-025",
        topic_key="q",
        content="body",
        embedding=[0.1] * 512,
        source_memory_ids=mem_ids,
        model_used="test-model",
    )
    assert summary["source_hash"] == compute_source_hash(mem_ids)
    assert str(summary["source_cursor"]) == max(mem_ids)


# ---------- get_summary_by_topic ----------


def test_get_summary_by_topic_returns_row_or_none(pg_fresh_db):
    _apply_028(pg_fresh_db)
    mem_ids = _seed_memories(1)

    from ogham.topic_summaries import get_summary_by_topic, upsert_summary

    assert get_summary_by_topic("test-025", "nope") is None

    upsert_summary(
        profile="test-025",
        topic_key="yes",
        content="body",
        embedding=[0.1] * 512,
        source_memory_ids=mem_ids,
        model_used="test-model",
    )
    row = get_summary_by_topic("test-025", "yes")
    assert row is not None
    assert row["topic_key"] == "yes"


# ---------- reverse cascade ----------


def test_get_affected_summaries_by_memory_id(pg_fresh_db):
    """Given a memory id, return every (summary_id, profile, topic_key) that cites it."""
    _apply_028(pg_fresh_db)
    mem_ids = _seed_memories(3)
    shared = mem_ids[0]

    from ogham.topic_summaries import get_affected_summaries_by_memory_id, upsert_summary

    upsert_summary(
        profile="test-025",
        topic_key="alpha",
        content="a",
        embedding=[0.1] * 512,
        source_memory_ids=[shared, mem_ids[1]],
        model_used="t",
    )
    upsert_summary(
        profile="test-025",
        topic_key="beta",
        content="b",
        embedding=[0.2] * 512,
        source_memory_ids=[shared, mem_ids[2]],
        model_used="t",
    )
    upsert_summary(
        profile="test-025",
        topic_key="gamma",
        content="g",
        embedding=[0.3] * 512,
        source_memory_ids=[mem_ids[1], mem_ids[2]],
        model_used="t",
    )

    affected = get_affected_summaries_by_memory_id(shared)
    topics = {a["topic_key"] for a in affected}
    assert topics == {"alpha", "beta"}


def test_get_affected_summaries_unknown_memory_returns_empty(pg_fresh_db):
    _apply_028(pg_fresh_db)
    from ogham.topic_summaries import get_affected_summaries_by_memory_id

    assert get_affected_summaries_by_memory_id("00000000-0000-0000-0000-000000000000") == []


# ---------- mark_stale / list_stale ----------


def test_mark_stale_sets_status_and_reason(pg_fresh_db, pg_client):
    _apply_028(pg_fresh_db)
    mem_ids = _seed_memories(1)

    from ogham.topic_summaries import mark_stale, upsert_summary

    s = upsert_summary(
        profile="test-025",
        topic_key="q",
        content="body",
        embedding=[0.1] * 512,
        source_memory_ids=mem_ids,
        model_used="t",
    )
    mark_stale(s["id"], reason="source memory edited")

    row = pg_client.fetchone(
        "SELECT status, stale_reason FROM topic_summaries WHERE id = %(id)s::uuid",
        {"id": str(s["id"])},
    )
    assert row["status"] == "stale"
    assert row["stale_reason"] == "source memory edited"


def test_list_stale_filters_profile(pg_fresh_db):
    _apply_028(pg_fresh_db)
    # Two scratch profiles. test-025 is pg_fresh_db's cleanup scope, so
    # its memory rows get scrubbed. test-025b leaks but is harmless --
    # no other test queries it, and topic_summaries is rebuilt fresh
    # via ROLLBACK+MIG_028 at the top of _apply_028.
    mem_a = _seed_memories(1, profile="test-025")
    mem_b = _seed_memories(1, profile="test-025b")

    from ogham.topic_summaries import list_stale, mark_stale, upsert_summary

    sa = upsert_summary(
        profile="test-025",
        topic_key="x",
        content="a",
        embedding=[0.1] * 512,
        source_memory_ids=mem_a,
        model_used="t",
    )
    sb = upsert_summary(
        profile="test-025b",
        topic_key="x",
        content="b",
        embedding=[0.1] * 512,
        source_memory_ids=mem_b,
        model_used="t",
    )
    mark_stale(sa["id"])
    mark_stale(sb["id"])

    out = list_stale(profile="test-025")
    ids = {str(r["id"]) for r in out}
    assert str(sa["id"]) in ids
    assert str(sb["id"]) not in ids
