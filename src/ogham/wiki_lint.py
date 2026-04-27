"""Wiki-layer maintenance lint (T1.6) — backend-agnostic orchestration.

Karpathy's maintenance verb mapped to Ogham. Each find_X function calls
into backend.wiki_lint_* / wiki_topic_list_* methods which dispatch to
the right driver (PostgresBackend uses migration 031 functions via
psycopg; SupabaseBackend uses PostgREST rpc() against the same
functions). Server-side SQL is in sql/migrations/031_wiki_rpc_functions.sql.

Categories:
  * contradictions    -- pairs of memories joined by relationship='contradicts'
  * orphans           -- memories with no edges (5-min grace window)
  * stale_lifecycle   -- memory_lifecycle rows stuck in 'stable' past threshold
  * stale_summaries   -- topic_summaries rows in status='stale'
  * summary_drift     -- topic_summaries whose stored source_hash no longer
                         matches the current set of tagged memories
"""

from __future__ import annotations

import logging
from typing import Any

from ogham.database import get_backend
from ogham.topic_summaries import compute_source_hash, list_stale

logger = logging.getLogger(__name__)


_DEFAULT_SAMPLE_SIZE = 10
_DEFAULT_STABLE_DAYS = 90
_DEFAULT_ORPHAN_GRACE_MINUTES = 5


def find_contradictions(profile: str, sample_size: int = _DEFAULT_SAMPLE_SIZE) -> dict[str, Any]:
    """Pairs of memories joined by relationship='contradicts'."""
    return get_backend().wiki_lint_contradictions(profile, sample_size=sample_size)


def find_orphans(profile: str, sample_size: int = _DEFAULT_SAMPLE_SIZE) -> dict[str, Any]:
    """Memories with no edges. The 5-min grace window excludes
    still-auto-linking rows."""
    return get_backend().wiki_lint_orphans(
        profile,
        sample_size=sample_size,
        grace_minutes=_DEFAULT_ORPHAN_GRACE_MINUTES,
    )


def find_stale_lifecycle(
    profile: str,
    older_than_days: int = _DEFAULT_STABLE_DAYS,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Memories whose lifecycle stage is 'stable' past `older_than_days`."""
    return get_backend().wiki_lint_stale_lifecycle(
        profile,
        older_than_days=older_than_days,
        sample_size=sample_size,
    )


def find_stale_summaries(profile: str, sample_size: int = _DEFAULT_SAMPLE_SIZE) -> dict[str, Any]:
    """topic_summaries rows in status='stale'.

    Reuses topic_summaries.list_stale(); thin wrapper to expose the
    same triage shape as the other lint checks (count + sample).
    """
    rows = list_stale(profile=profile)
    sample_rows = rows[:sample_size]
    return {
        "count": len(rows),
        "sample": [
            {
                "id": str(r["id"]),
                "topic_key": r["topic_key"],
                "version": r["version"],
                "stale_reason": r.get("stale_reason"),
                "updated_at": r["updated_at"],
            }
            for r in sample_rows
        ],
    }


def find_summary_drift(profile: str, sample_size: int = _DEFAULT_SAMPLE_SIZE) -> dict[str, Any]:
    """topic_summaries whose stored source_hash no longer matches current sources.

    Catches the case where the recompute hooks missed an event (manual
    SQL edit, bulk import) so the cached summary is silently out of
    date. Re-hashes the current sources for each fresh topic and
    compares to the stored hash. Only checks 'fresh' rows -- 'stale'
    rows are already flagged by find_stale_summaries.
    """
    backend = get_backend()
    fresh_summaries = backend.wiki_topic_list_fresh_for_drift(profile)

    drifted: list[dict[str, Any]] = []
    for row in fresh_summaries:
        topic_key = row["topic_key"]
        current_ids = backend.wiki_recompute_get_source_ids(profile, topic_key)
        current_hash = compute_source_hash(current_ids)
        stored_hash_raw = row.get("source_hash")
        if isinstance(stored_hash_raw, str):
            # Supabase returns bytea as `\xDEADBEEF`; strip the prefix.
            stored_hash = bytes.fromhex(stored_hash_raw.removeprefix("\\x"))
        elif isinstance(stored_hash_raw, (bytes, bytearray, memoryview)):
            stored_hash = bytes(stored_hash_raw)
        else:
            stored_hash = b""

        if current_hash != stored_hash:
            drifted.append(
                {
                    "id": str(row["id"]),
                    "topic_key": topic_key,
                    "current_source_count": len(current_ids),
                }
            )

    return {
        "count": len(drifted),
        "sample": drifted[:sample_size],
    }


def lint_report(
    profile: str,
    *,
    stable_days: int = _DEFAULT_STABLE_DAYS,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    include_drift: bool = True,
) -> dict[str, Any]:
    """Aggregate health report across all lint checks for a profile."""
    contradictions = find_contradictions(profile, sample_size=sample_size)
    orphans = find_orphans(profile, sample_size=sample_size)
    stale_lifecycle = find_stale_lifecycle(
        profile, older_than_days=stable_days, sample_size=sample_size
    )
    stale_summaries = find_stale_summaries(profile, sample_size=sample_size)

    drift: dict[str, Any] = {"count": 0, "sample": [], "skipped": True}
    if include_drift:
        drift = find_summary_drift(profile, sample_size=sample_size)

    issue_count = (
        contradictions["count"]
        + orphans["count"]
        + stale_lifecycle["count"]
        + stale_summaries["count"]
        + drift["count"]
    )

    return {
        "profile": profile,
        "healthy": issue_count == 0,
        "issue_count": issue_count,
        "contradictions": contradictions,
        "orphans": orphans,
        "stale_lifecycle": stale_lifecycle,
        "stale_summaries": stale_summaries,
        "summary_drift": drift,
    }
