"""Tests for database facade wiring and delegation."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from ogham import database as db
from ogham.backends.supabase import SupabaseBackend
from ogham.config import Settings


@pytest.fixture(autouse=True)
def _reset_backend():
    """Ensure each test starts and ends with a clean backend singleton."""
    db._backend = None
    yield
    db._backend = None


def test_default_backend_is_supabase():
    """settings.database_backend should default to 'supabase'."""
    from ogham.config import settings

    assert settings.database_backend == "supabase"


def test_get_backend_returns_supabase_by_default():
    """get_backend() should return a SupabaseBackend instance by default."""
    backend = db.get_backend()
    assert isinstance(backend, SupabaseBackend)


def test_facade_delegates_store_memory():
    """store_memory() should forward to the backend's store_memory."""
    mock = MagicMock()
    db._backend = mock

    db.store_memory(content="test", embedding=[0.1, 0.2], profile="default")

    mock.store_memory.assert_called_once_with(
        "test",
        [0.1, 0.2],
        "default",
        None,
        None,
        None,
        None,
        importance=0.5,
        surprise=0.5,
        recurrence_days=None,
    )


def test_facade_delegates_hybrid_search():
    """hybrid_search_memories() should forward to the backend."""
    mock = MagicMock()
    db._backend = mock

    db.hybrid_search_memories(
        query_text="hello",
        query_embedding=[0.1, 0.2],
        profile="default",
    )

    mock.hybrid_search_memories.assert_called_once_with(
        "hello",
        [0.1, 0.2],
        "default",
        None,
        None,
        None,
        None,
        query_entity_tags=None,
        recency_decay=0.0,
    )


def test_facade_delegates_delete_memory():
    """delete_memory() should forward to the backend."""
    mock = MagicMock()
    db._backend = mock

    db.delete_memory(memory_id="abc-123", profile="default")

    mock.delete_memory.assert_called_once_with("abc-123", "default")


def test_get_client_only_works_with_supabase():
    """get_client() should raise RuntimeError when backend has no _get_client."""
    db._backend = MagicMock(spec=[])

    with pytest.raises(RuntimeError, match="does not expose a raw client"):
        db.get_client()


def test_invalid_backend_rejected():
    """Settings should reject unknown database_backend values."""
    with pytest.raises(ValidationError):
        Settings(
            database_backend="mongodb",
            supabase_url="https://example.supabase.co",
            supabase_key="fake-key",
        )


# --- TBU-159: boot-time schema-fingerprint guard ---


def test_validate_schema_fingerprint_raises_on_mismatch():
    """DB column dim disagreeing with settings.embedding_dim must raise RuntimeError
    with an actionable message -- not silently degrade (writes succeed, reads fail)."""
    mock_backend = MagicMock()
    mock_backend._execute.return_value = 1024
    mock_settings = MagicMock()
    mock_settings.embedding_dim = 512

    with pytest.raises(RuntimeError, match="Schema dim mismatch") as exc_info:
        db._validate_schema_fingerprint(mock_backend, mock_settings)

    msg = str(exc_info.value)
    assert "vector(1024)" in msg
    assert "settings.embedding_dim=512" in msg
    assert "EMBEDDING_DIM=1024" in msg  # actionable fix (b)


def test_validate_schema_fingerprint_passes_on_match():
    """No exception when DB column dim matches settings.embedding_dim."""
    mock_backend = MagicMock()
    mock_backend._execute.return_value = 512
    mock_settings = MagicMock()
    mock_settings.embedding_dim = 512

    db._validate_schema_fingerprint(mock_backend, mock_settings)


def test_validate_schema_fingerprint_skips_backend_without_execute():
    """Supabase/PostgREST + gateway backends have no raw SQL introspection path --
    guard must skip (log + return) rather than raise AttributeError."""
    mock_backend = MagicMock(spec=[])
    mock_settings = MagicMock()
    mock_settings.embedding_dim = 512

    db._validate_schema_fingerprint(mock_backend, mock_settings)


def test_validate_schema_fingerprint_skips_when_schema_not_applied():
    """Fresh install with no schema applied yet -- don't block startup on a
    guard that assumes the memories table already exists."""
    mock_backend = MagicMock()
    mock_backend._execute.side_effect = Exception('relation "memories" does not exist')
    mock_settings = MagicMock()
    mock_settings.embedding_dim = 512

    db._validate_schema_fingerprint(mock_backend, mock_settings)


def test_get_backend_runs_fingerprint_guard_for_supabase():
    """get_backend() should call the guard on first construction; a SupabaseBackend
    has no _execute so this is a no-op skip, but must not raise or block."""
    backend = db.get_backend()
    assert isinstance(backend, SupabaseBackend)
