"""Phase 6 tests — all 5 memory-mutation MCP tools enqueue topic recomputes.

Real Postgres scratch DB for the mutation surface; _run_recompute is
patched so we can assert enqueue calls without running LLM / embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

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


@pytest.fixture
def _hooks_env(monkeypatch, pg_fresh_db):
    """Apply migrations + set scratch profile + reload executor state."""
    monkeypatch.setenv("OGHAM_PROFILE", "test-025")
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)
    pg_fresh_db.apply_rollback(ROLLBACK_028)
    pg_fresh_db.apply_sql(MIG_028)
    pg_fresh_db.apply_sql(MIG_030)
    pg_fresh_db.apply_sql(MIG_031)

    import ogham.recompute_executor as exe

    exe.shutdown()
    yield exe
    exe.shutdown()


def _enqueued_topics(mock_runner) -> set[str]:
    """Extract unique topic_keys from captured _run_recompute calls."""
    return {
        call.args[1] if len(call.args) > 1 else call.kwargs.get("topic_key")
        for call in mock_runner.call_args_list
    }


def test_store_memory_enqueues_per_tag(_hooks_env):
    """store_memory() must enqueue one recompute per tag on the new row.

    Mock the underlying enriched write so we don't need a real embedding
    provider -- the hook logic (enqueue_for_tags) sits above that call.
    """
    from ogham.tools.memory import store_memory

    with (
        patch(
            "ogham.service.store_memory_enriched",
            return_value={"id": "00000000-0000-0000-0000-000000000001", "status": "stored"},
        ),
        patch("ogham.recompute_executor._run_recompute") as runner,
    ):
        store_memory(
            content="seed memory for store-hook test",
            tags=["project:ogham", "type:decision"],
        )
        _hooks_env.flush(timeout=5.0)

    topics = _enqueued_topics(runner)
    assert topics == {"project:ogham", "type:decision"}


def test_delete_memory_enqueues_pre_delete_tags(_hooks_env):
    """Tags must be fetched BEFORE the delete or they're unrecoverable."""
    from ogham.database import get_backend
    from ogham.tools.memory import delete_memory

    # Seed directly so we don't chain through store_memory (which would
    # also enqueue and confuse the test).
    backend = cast(Any, get_backend())
    row = backend._execute(
        """INSERT INTO memories (content, profile, source, tags)
           VALUES ('body to delete', 'test-025', 't', ARRAY['alpha', 'beta'])
           RETURNING id::text AS id""",
        {},
        fetch="one",
    )
    mid = row["id"]

    with patch("ogham.recompute_executor._run_recompute") as runner:
        out = delete_memory(mid)
        _hooks_env.flush(timeout=5.0)

    assert out["status"] == "deleted"
    assert _enqueued_topics(runner) == {"alpha", "beta"}


def test_update_memory_enqueues_union_of_old_and_new_tags(_hooks_env):
    """Tag-replace from [a,b] to [b,c] must enqueue a (dropped), b, and c."""
    from ogham.database import get_backend
    from ogham.tools.memory import update_memory

    backend = cast(Any, get_backend())
    row = backend._execute(
        """INSERT INTO memories (content, profile, source, tags)
           VALUES ('body to edit', 'test-025', 't', ARRAY['alpha', 'beta'])
           RETURNING id::text AS id""",
        {},
        fetch="one",
    )
    mid = row["id"]

    with patch("ogham.recompute_executor._run_recompute") as runner:
        # Mock embedding so content update doesn't hit the real provider chain.
        with patch("ogham.tools.memory.generate_embedding", return_value=[0.1] * 512):
            update_memory(mid, content="new body", tags=["beta", "gamma"])
        _hooks_env.flush(timeout=5.0)

    topics = _enqueued_topics(runner)
    assert topics == {"alpha", "beta", "gamma"}, (
        "dropped tag alpha must be enqueued so its summary drops this memory"
    )


def test_reinforce_memory_enqueues_current_tags(_hooks_env):
    from ogham.database import get_backend
    from ogham.tools.memory import reinforce_memory

    backend = cast(Any, get_backend())
    row = backend._execute(
        """INSERT INTO memories (content, profile, source, tags, confidence)
           VALUES ('reinforce me', 'test-025', 't', ARRAY['x', 'y'], 0.5)
           RETURNING id::text AS id""",
        {},
        fetch="one",
    )
    mid = row["id"]

    with patch("ogham.recompute_executor._run_recompute") as runner:
        reinforce_memory(mid, strength=0.9)
        _hooks_env.flush(timeout=5.0)

    assert _enqueued_topics(runner) == {"x", "y"}


def test_contradict_memory_enqueues_current_tags(_hooks_env):
    from ogham.database import get_backend
    from ogham.tools.memory import contradict_memory

    backend = cast(Any, get_backend())
    row = backend._execute(
        """INSERT INTO memories (content, profile, source, tags, confidence)
           VALUES ('contradict me', 'test-025', 't', ARRAY['m', 'n'], 0.9)
           RETURNING id::text AS id""",
        {},
        fetch="one",
    )
    mid = row["id"]

    with patch("ogham.recompute_executor._run_recompute") as runner:
        contradict_memory(mid, strength=0.1)
        _hooks_env.flush(timeout=5.0)

    assert _enqueued_topics(runner) == {"m", "n"}


def test_store_memory_with_no_tags_is_noop(_hooks_env):
    """Untagged memories don't feed any topic summary -- no enqueue."""
    from ogham.tools.memory import store_memory

    with (
        patch(
            "ogham.service.store_memory_enriched",
            return_value={"id": "00000000-0000-0000-0000-000000000002", "status": "stored"},
        ),
        patch("ogham.recompute_executor._run_recompute") as runner,
    ):
        store_memory(content="orphan memory with no tags", tags=None)
        _hooks_env.flush(timeout=5.0)

    assert runner.call_count == 0


def test_store_memory_enriched_failure_does_not_enqueue(_hooks_env):
    """If the underlying write raises, no enqueue should fire."""
    from ogham.tools.memory import store_memory

    with patch(
        "ogham.service.store_memory_enriched",
        side_effect=RuntimeError("write blew up"),
    ):
        with patch("ogham.recompute_executor._run_recompute") as runner:
            with pytest.raises(RuntimeError, match="blew up"):
                store_memory(content="this will fail", tags=["anything"])
            _hooks_env.flush(timeout=5.0)

    assert runner.call_count == 0
