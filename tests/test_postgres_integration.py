"""Integration tests for PostgresBackend against a real Postgres database.

Run with:
    DATABASE_BACKEND=postgres DATABASE_URL="postgres://..." \
        uv run pytest tests/test_postgres_integration.py -v

Skip with: uv run pytest -m 'not postgres_integration'
"""

import pytest

TEST_PROFILE = "_test_postgres"


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
        not _can_connect(), reason="Postgres backend not configured or unreachable"
    ),
]


@pytest.fixture(autouse=True)
def cleanup():
    """Delete all test data after each test."""
    yield
    from ogham.database import get_backend

    backend = get_backend()
    backend._execute(
        "DELETE FROM memories WHERE profile = %(p)s",
        {"p": TEST_PROFILE},
        fetch="none",
    )
    try:
        backend._execute(
            "DELETE FROM profile_settings WHERE profile = %(p)s",
            {"p": TEST_PROFILE},
            fetch="none",
        )
    except Exception:
        pass


def test_store_and_retrieve():
    """Store a memory and retrieve it via list_recent."""
    from ogham.database import get_backend

    backend = get_backend()
    fake_emb = [0.1] * 768
    result = backend.store_memory(
        content="test memory for postgres",
        embedding=fake_emb,
        profile=TEST_PROFILE,
        tags=["test"],
    )
    assert result["content"] == "test memory for postgres"
    assert result["profile"] == TEST_PROFILE

    recent = backend.list_recent_memories(profile=TEST_PROFILE, limit=5)
    assert len(recent) == 1
    assert recent[0]["content"] == "test memory for postgres"


def test_search_memories():
    """Store and search via match_memories RPC."""
    from ogham.database import get_backend

    backend = get_backend()
    fake_emb = [0.1] * 768
    backend.store_memory(
        content="searchable postgres memory",
        embedding=fake_emb,
        profile=TEST_PROFILE,
    )
    results = backend.search_memories(
        query_embedding=fake_emb,
        profile=TEST_PROFILE,
        threshold=0.1,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0]["content"] == "searchable postgres memory"


def test_hybrid_search():
    """Store and search via hybrid_search_memories RPC."""
    from ogham.database import get_backend

    backend = get_backend()
    fake_emb = [0.1] * 768
    backend.store_memory(
        content="hybrid search postgres test",
        embedding=fake_emb,
        profile=TEST_PROFILE,
    )
    results = backend.hybrid_search_memories(
        query_text="hybrid search postgres",
        query_embedding=fake_emb,
        profile=TEST_PROFILE,
        limit=5,
    )
    assert len(results) >= 1


def test_delete_memory():
    """Store and delete a memory."""
    from ogham.database import get_backend

    backend = get_backend()
    fake_emb = [0.1] * 768
    mem = backend.store_memory(
        content="to be deleted",
        embedding=fake_emb,
        profile=TEST_PROFILE,
    )
    deleted = backend.delete_memory(mem["id"], TEST_PROFILE)
    assert deleted is True
    recent = backend.list_recent_memories(profile=TEST_PROFILE)
    assert len(recent) == 0


def test_profile_stats():
    """get_memory_stats works via RPC."""
    from ogham.database import get_backend

    backend = get_backend()
    fake_emb = [0.1] * 768
    backend.store_memory(
        content="stats test",
        embedding=fake_emb,
        profile=TEST_PROFILE,
        tags=["alpha"],
    )
    stats = backend.get_memory_stats(TEST_PROFILE)
    assert stats["total"] == 1
