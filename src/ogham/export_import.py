"""Export and import memory data."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

from ogham.database import (
    batch_check_duplicates,
    get_all_memories_full,
    get_profile_ttl,
    store_memories_batch,
)
from ogham.embeddings import generate_embeddings_batch


def export_memories(profile: str, format: str = "json") -> str:
    """Export all memories in a profile to a string."""
    memories = get_all_memories_full(profile)

    if format == "markdown":
        return _export_markdown(profile, memories)
    return _export_json(profile, memories)


def _export_json(profile: str, memories: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "profile": profile,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(memories),
            "memories": memories,
        },
        indent=2,
        default=str,
    )


def _export_markdown(profile: str, memories: list[dict[str, Any]]) -> str:
    lines = [
        "# Ogham Memory Export",
        "",
        f"**Profile:** {profile}",
        f"**Count:** {len(memories)}",
        f"**Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
    ]
    for mem in memories:
        lines.append(f"## {mem.get('created_at', 'unknown')[:10]}")
        tags = mem.get("tags", [])
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        source = mem.get("source")
        if source:
            lines.append(f"**Source:** {source}")
        lines.append("")
        lines.append(mem["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _build_row(
    mem: dict[str, Any],
    embedding: list[float],
    profile: str,
    expires_at: str | None,
) -> dict[str, Any]:
    """Build a row dict ready for database insertion."""
    row = {
        "content": mem["content"],
        "embedding": str(embedding),
        "profile": profile,
        "metadata": mem.get("metadata") or {},
        "source": mem.get("source"),
        "tags": mem.get("tags") or [],
    }
    if expires_at is not None:
        row["expires_at"] = expires_at
    return row


def import_memories(
    data: str,
    profile: str,
    dedup_threshold: float = 0.0,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_embed_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Import memories from a JSON string into a profile.

    Args:
        on_progress: Optional callback(imported, skipped, total) called after each memory.
        on_embed_progress: Optional callback(embedded, total) called after each batch.
    """
    parsed = json.loads(data)
    memories = parsed.get("memories", [])
    total = len(memories)

    ttl_days = get_profile_ttl(profile)
    expires_at = None
    if ttl_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    # Phase 1: Batch embed all memories upfront
    all_texts = [mem["content"] for mem in memories]
    embeddings = generate_embeddings_batch(
        all_texts, on_progress=on_embed_progress
    )

    # Phase 2: Parallel batch dedup (concurrent RPC batches to use multiple DB cores)
    skipped = 0
    to_insert: list[dict[str, Any]] = []

    if dedup_threshold > 0:
        dedup_batch_size = 50
        is_dup = [False] * total

        # Build batch ranges
        batch_ranges = [
            (start, min(start + dedup_batch_size, total))
            for start in range(0, total, dedup_batch_size)
        ]

        def _check_batch(batch_range: tuple[int, int]) -> tuple[int, int, list[bool]]:
            start, end = batch_range
            batch_embeddings = embeddings[start:end]
            results = batch_check_duplicates(
                query_embeddings=batch_embeddings,
                profile=profile,
                threshold=dedup_threshold,
            )
            return start, end, results

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_check_batch, br) for br in batch_ranges]
            completed = 0
            for future in futures:
                start, end, batch_results = future.result()
                for i, dup in enumerate(batch_results):
                    is_dup[start + i] = dup
                    if dup:
                        skipped += 1
                completed += end - start
                if on_progress:
                    on_progress(completed - skipped, skipped, total)

        for i, (mem, embedding) in enumerate(zip(memories, embeddings)):
            if not is_dup[i]:
                to_insert.append(_build_row(mem, embedding, profile, expires_at))
    else:
        for mem, embedding in zip(memories, embeddings):
            to_insert.append(_build_row(mem, embedding, profile, expires_at))
        if on_progress:
            on_progress(len(to_insert), 0, total)

    # Phase 3: Batch insert non-duplicates
    batch_size = 100
    for start in range(0, len(to_insert), batch_size):
        batch = to_insert[start : start + batch_size]
        store_memories_batch(batch)

    imported = len(to_insert)

    return {
        "status": "complete",
        "profile": profile,
        "imported": imported,
        "skipped": skipped,
        "total": total,
    }
