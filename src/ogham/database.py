"""Database facade — delegates to the configured backend driver.

All callers import from this module; the backend is selected at runtime
based on ``settings.database_backend`` (defaults to ``"supabase"``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_backend = None


def _reset_backend() -> None:
    """Reset the backend singleton. Used by tests."""
    global _backend
    _backend = None


def get_backend():
    """Return (and lazily create) the singleton backend instance."""
    global _backend
    if _backend is None:
        from ogham.config import settings

        backend_name = getattr(settings, "database_backend", "supabase")
        if backend_name == "postgres":
            # Placeholder — PostgresBackend will be implemented in a future task.
            from ogham.backends.postgres import PostgresBackend

            _backend = PostgresBackend()
        else:
            from ogham.backends.supabase import SupabaseBackend

            _backend = SupabaseBackend()
    return _backend


def get_client():
    """Backwards-compatible access to the underlying database client.

    Only works for backends that expose ``_get_client()`` (e.g. SupabaseBackend).
    """
    backend = get_backend()
    if not hasattr(backend, "_get_client"):
        raise RuntimeError(
            f"Backend {type(backend).__name__!r} does not expose a raw client"
        )
    return backend._get_client()


# ── Thin delegates — one per public function ────────────────────────────


def store_memory(
    content: str,
    embedding: list[float],
    profile: str,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    return get_backend().store_memory(
        content, embedding, profile, metadata, source, tags, expires_at
    )


def store_memories_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return get_backend().store_memories_batch(rows)


def search_memories(
    query_embedding: list[float],
    profile: str,
    threshold: float | None = None,
    limit: int | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    return get_backend().search_memories(query_embedding, profile, threshold, limit, tags, source)


def batch_check_duplicates(
    query_embeddings: list[list[float]],
    profile: str,
    threshold: float = 0.8,
) -> list[bool]:
    return get_backend().batch_check_duplicates(query_embeddings, profile, threshold)


def hybrid_search_memories(
    query_text: str,
    query_embedding: list[float],
    profile: str,
    limit: int | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    return get_backend().hybrid_search_memories(
        query_text, query_embedding, profile, limit, tags, source
    )


def list_recent_memories(
    profile: str,
    limit: int = 10,
    source: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    return get_backend().list_recent_memories(profile, limit, source, tags)


def get_memory_stats(profile: str) -> dict[str, Any]:
    return get_backend().get_memory_stats(profile)


def get_all_memories_full(profile: str) -> list[dict[str, Any]]:
    return get_backend().get_all_memories_full(profile)


def get_all_memories_content(profile: str | None = None) -> list[dict[str, Any]]:
    return get_backend().get_all_memories_content(profile)


def list_profiles() -> list[dict[str, Any]]:
    return get_backend().list_profiles()


def batch_update_embeddings(ids: list[str], embeddings: list[list[float]]) -> int:
    return get_backend().batch_update_embeddings(ids, embeddings)


def record_access(memory_ids: list[str]) -> None:
    return get_backend().record_access(memory_ids)


def update_confidence(memory_id: str, signal: float, profile: str) -> float:
    return get_backend().update_confidence(memory_id, signal, profile)


def delete_memory(memory_id: str, profile: str) -> bool:
    return get_backend().delete_memory(memory_id, profile)


def update_memory(memory_id: str, updates: dict[str, Any], profile: str) -> dict[str, Any]:
    return get_backend().update_memory(memory_id, updates, profile)


def get_profile_ttl(profile: str) -> int | None:
    return get_backend().get_profile_ttl(profile)


def set_profile_ttl(profile: str, ttl_days: int | None) -> dict[str, Any]:
    return get_backend().set_profile_ttl(profile, ttl_days)


def cleanup_expired(profile: str) -> int:
    return get_backend().cleanup_expired(profile)


def count_expired(profile: str) -> int:
    return get_backend().count_expired(profile)


def auto_link_memory(
    memory_id: str,
    embedding: list[float],
    profile: str,
    threshold: float = 0.85,
    max_links: int = 5,
) -> int:
    return get_backend().auto_link_memory(memory_id, embedding, profile, threshold, max_links)


def link_unlinked_memories(
    profile: str,
    threshold: float = 0.85,
    max_links: int = 5,
    batch_size: int = 100,
) -> int:
    return get_backend().link_unlinked_memories(profile, threshold, max_links, batch_size)


def explore_memory_graph(
    query_text: str,
    query_embedding: list[float],
    profile: str,
    limit: int = 5,
    depth: int = 1,
    min_strength: float = 0.5,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    return get_backend().explore_memory_graph(
        query_text, query_embedding, profile, limit, depth, min_strength, tags, source
    )


def create_relationship(
    source_id: str,
    target_id: str,
    relationship: str,
    strength: float = 1.0,
    created_by: str = "user",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_backend().create_relationship(
        source_id, target_id, relationship, strength, created_by, metadata
    )


def get_related_memories(
    memory_id: str,
    depth: int = 1,
    min_strength: float = 0.5,
    relationship_types: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return get_backend().get_related_memories(
        memory_id, depth, min_strength, relationship_types, limit
    )
