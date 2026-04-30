"""Backfill entities + memory_entities for existing memory rows.

The live write path (service.store_memory) calls
``backend.link_memory_entities`` for every new memory after v0.14. This
module covers everything written before v0.14 -- it walks the memories
table, runs ``extract_entities`` on each content body, and feeds the same
RPC to populate the graph.

Idempotent: ``link_memory_entities`` uses ``ON CONFLICT DO NOTHING`` on
the (memory_id, entity_id) unique constraint, so a second run only
touches memories whose extracted entity set has grown since the first.

Operationally this is a one-shot per deployment after applying migration
036. New deployments fresh-installing schema.sql at v0.14+ never need it
(their tables are populated incrementally by the live write path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ogham.database import get_backend
from ogham.extraction import extract_entities

logger = logging.getLogger(__name__)


def backfill_entities(
    profile: str | None = None,
    batch_size: int = 200,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Walk memories and populate entities + memory_entities.

    Args:
        profile: Restrict to one profile, or None for every memory in the
            backend. Single-profile is the safer default for shared
            deployments where you want to pace the work.
        batch_size: How many rows to fetch per page. 200 keeps each
            ``link_memory_entities`` round-trip small while still
            amortising the SELECT cost.
        on_progress: Optional ``(processed, edges_added, total)`` hook
            called after each row. Useful for CLI progress bars.

    Returns:
        ``{"status": "complete", "processed": N, "edges_added": M,
           "memories_with_entities": K, "total": T, "profile": ...}``
    """
    backend = get_backend()
    rows = _select_memory_rows(backend, profile=profile)
    total = len(rows)
    edges_added = 0
    memories_with_entities = 0

    for processed, row in enumerate(rows, start=1):
        memory_id = str(row["id"])
        content = row.get("content") or ""
        row_profile = row.get("profile") or profile or "default"
        entity_tags = extract_entities(content)
        if entity_tags:
            try:
                inserted = backend.link_memory_entities(
                    memory_id=memory_id,
                    profile=row_profile,
                    entity_tags=entity_tags,
                )
                edges_added += int(inserted or 0)
                memories_with_entities += 1
            except Exception as exc:
                # Don't crash the whole backfill on a single bad row;
                # log and keep going. Common cause: link_memory_entities
                # RPC missing because migration 036 wasn't applied.
                logger.warning(
                    "backfill: link_memory_entities failed for %s: %s",
                    memory_id,
                    exc,
                )
        if on_progress:
            on_progress(processed, edges_added, total)

    return {
        "status": "complete",
        "processed": total,
        "edges_added": edges_added,
        "memories_with_entities": memories_with_entities,
        "total": total,
        "profile": profile,
    }


def _select_memory_rows(backend: Any, *, profile: str | None) -> list[dict[str, Any]]:
    """Pull (id, profile, content) for memories the backfill should touch.

    The Postgres backend uses raw SQL; the Supabase backend reads via
    PostgREST. Both return a list of dicts shaped {"id", "profile",
    "content"}. Sorted by ``created_at`` so progress reporting reads
    naturally and partial runs still hit older memories first.
    """
    backend_kind = backend.__class__.__name__.lower()
    if "postgres" in backend_kind:
        if profile is not None:
            sql = (
                "SELECT id::text AS id, profile, content FROM memories "
                "WHERE profile = %(profile)s "
                "  AND (expires_at IS NULL OR expires_at > now()) "
                "ORDER BY created_at"
            )
            params: dict[str, Any] = {"profile": profile}
        else:
            sql = (
                "SELECT id::text AS id, profile, content FROM memories "
                "WHERE expires_at IS NULL OR expires_at > now() "
                "ORDER BY created_at"
            )
            params = {}
        rows = backend._execute(sql, params, fetch="all")
        return rows or []
    if "supabase" in backend_kind:
        # PostgREST default limit is 1000 rows. Paginate explicitly via
        # range() so large profiles (>1000 memories) get walked in full.
        client = backend._get_client()
        page = 1000
        offset = 0
        out: list[dict[str, Any]] = []
        while True:
            query = (
                client.table("memories")
                .select("id,profile,content")
                .order("created_at")
                .range(offset, offset + page - 1)
            )
            if profile is not None:
                query = query.eq("profile", profile)
            result = query.execute()
            chunk = list(result.data or [])
            out.extend(chunk)
            if len(chunk) < page:
                break
            offset += page
        return out
    raise NotImplementedError(
        f"backfill_entities does not support backend {backend.__class__.__name__!r}"
    )
