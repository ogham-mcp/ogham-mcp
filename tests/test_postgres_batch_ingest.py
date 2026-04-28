"""Tests for PostgresBackend.store_memories_batch true-batching.

Regression fix: the previous implementation called cur.execute() in a loop
(N round-trips per batch). This test asserts ONE execute per batch via a
multi-row VALUES INSERT. Benchmarks and any bulk ingest caller get ~10x-50x
speedup as a result.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


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


def _rows(n: int, profile: str = "test-025") -> list[dict]:
    """N minimal valid rows for memories insert."""
    return [
        {
            "content": f"batch row {i} content body",
            "embedding": str([0.1 + i * 0.001] * 512),
            "profile": profile,
            "metadata": {"batch_idx": i},
            "source": "test",
            "tags": ["batch-test"],
        }
        for i in range(n)
    ]


def test_store_memories_batch_inserts_all_rows(pg_fresh_db):
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = _rows(5)
    result = backend.store_memories_batch(rows)

    assert len(result) == 5
    assert all("id" in r for r in result)
    assert all("content" in r for r in result)

    count = pg_fresh_db.count("memories")
    assert count >= 5


def test_store_memories_batch_returns_rows_in_order(pg_fresh_db):
    """RETURNING order must match input order so callers can zip(results, inputs)."""
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = _rows(8)
    result = backend.store_memories_batch(rows)

    assert len(result) == 8
    for i, r in enumerate(result):
        assert r["content"] == f"batch row {i} content body", (
            f"RETURNING reordered: expected row {i} at position {i}, got {r['content']!r}"
        )


def test_store_memories_batch_does_one_execute_not_n(pg_fresh_db):
    """Performance contract: a single multi-row INSERT, not N+1.

    This is the trap the prior implementation fell into -- the method was
    named 'batch' but internally looped cur.execute(). At 100 rows per
    harness-batch * 500 benchmark questions, the N+1 pattern was making
    clean LME runs take hours instead of minutes.
    """
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = _rows(10)

    insert_calls: list[str] = []

    real_execute = PostgresBackend._execute

    def wrapped_execute(self, sql, *a, **kw):
        sql_str = str(sql).strip().upper()
        if sql_str.startswith("INSERT INTO MEMORIES"):
            insert_calls.append(str(sql))
        return real_execute(self, sql, *a, **kw)

    with patch.object(PostgresBackend, "_execute", wrapped_execute):
        backend.store_memories_batch(rows)

    assert len(insert_calls) == 1, (
        f"store_memories_batch must do 1 INSERT for 10 rows, got {len(insert_calls)}: "
        f"{[c[:80] for c in insert_calls]}"
    )
    # Sanity: the one INSERT we did must be a multi-row VALUES, not a 1-row one.
    assert insert_calls[0].count("(") >= 10, (
        "INSERT statement doesn't look like multi-row VALUES -- need 10 parenthesized tuples"
    )


def test_store_memories_batch_empty_is_noop(pg_fresh_db):
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    assert backend.store_memories_batch([]) == []


def test_store_memories_batch_handles_metadata_json(pg_fresh_db, pg_client):
    """Metadata dict must be stored as JSON, not as Python repr."""
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = [
        {
            "content": "with metadata",
            "embedding": str([0.1] * 512),
            "profile": "test-025",
            "metadata": {"nested": {"key": "value"}, "n": 42},
            "source": "test",
            "tags": [],
        }
    ]
    backend.store_memories_batch(rows)
    row = pg_client.fetchone(
        "SELECT metadata FROM memories WHERE profile='test-025' AND content='with metadata'"
    )
    assert row["metadata"] == {"nested": {"key": "value"}, "n": 42}
