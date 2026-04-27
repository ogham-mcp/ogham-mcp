"""Topic-summary cache — backend-agnostic CRUD + cascade layer for the
T1.4 wiki-compile path (see docs/plans/2026-04-23-wiki-hybrid-tier1-spec.md).

This module is the orchestration layer: it computes hashes/cursors and
calls into backend.wiki_topic_*() methods which know how to dispatch to
the underlying store. PostgresBackend uses psycopg + the migration 031
functions; SupabaseBackend uses PostgREST rpc() against the same
functions. Both end up at the same SQL on the server side.

Design points enforced here:
  * source_hash pairs with source_cursor: the hash (sha256 of sorted
    source ids) catches membership changes; the cursor identifies the
    newest folded-in memory.
  * Upsert is atomic on the server side via wiki_topic_upsert.
  * No LLM dep, no embedding compute. Callers pass the composed
    content + its embedding; this module only persists.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ogham.database import get_backend

_SOURCE_HASH_DELIM = b"\n"


def compute_source_hash(memory_ids: list[str]) -> bytes:
    """SHA-256 over the sorted source memory ids, newline-delimited."""
    payload = _SOURCE_HASH_DELIM.join(mid.encode("ascii") for mid in sorted(memory_ids))
    return hashlib.sha256(payload).digest()


def _source_cursor(memory_ids: list[str]) -> str | None:
    """Lexical max of the source ids. Stable interface for Phase 4's
    eventual temporal-newest cursor."""
    return max(memory_ids) if memory_ids else None


def upsert_summary(
    *,
    profile: str,
    topic_key: str,
    content: str,
    embedding: list[float],
    source_memory_ids: list[str],
    model_used: str,
    token_count: int | None = None,
    importance: float = 0.5,
) -> dict[str, Any]:
    """Insert or refresh a topic summary atomically (server-side CTE)."""
    return get_backend().wiki_topic_upsert(
        profile=profile,
        topic_key=topic_key,
        content=content,
        embedding=embedding,
        source_memory_ids=source_memory_ids,
        model_used=model_used,
        source_cursor=_source_cursor(source_memory_ids),
        source_hash=compute_source_hash(source_memory_ids),
        token_count=token_count,
        importance=importance,
    )


def get_summary_by_topic(profile: str, topic_key: str) -> dict[str, Any] | None:
    """Fetch the live summary for (profile, topic_key) or None."""
    return get_backend().wiki_topic_get_by_key(profile, topic_key)


def search_summaries(
    profile: str,
    query_embedding: list[float],
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """Vector-search the fresh topic summaries for a profile."""
    if top_k <= 0 or not query_embedding:
        return []
    return get_backend().wiki_topic_search(
        profile=profile,
        query_embedding=query_embedding,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def get_affected_summaries_by_memory_id(memory_id: str) -> list[dict[str, Any]]:
    """Every summary that cites the given memory."""
    return get_backend().wiki_topic_get_affected(memory_id)


def mark_stale(summary_id: str, reason: str | None = None) -> None:
    """Flip a summary to stale, optionally recording why."""
    get_backend().wiki_topic_mark_stale(str(summary_id), reason)


def sweep_stale_summaries(profile: str, older_than_days: int = 30) -> int:
    """Flip fresh -> stale for summaries idle past `older_than_days`.
    Returns the number of rows flipped."""
    return get_backend().wiki_topic_sweep_stale(profile, older_than_days)


def list_stale(
    profile: str | None = None, older_than_days: int | None = None
) -> list[dict[str, Any]]:
    """Stale summaries, optionally scoped to one profile / older than N days."""
    return get_backend().wiki_topic_list_stale(profile, older_than_days)
