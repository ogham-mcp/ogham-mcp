"""Linear import MCP tool -- fetches issues + stores as memories with dedupe.

Mirrors the TBU-114 amendment pattern used by ``ogham.tools.entity_graph``:
``import_linear_impl`` is pure -- it only imports from ``ogham.importers.linear``
(the domain-ish mapper module) and takes the client + memory service as
injected params, so it's unit-testable without a real Linear token or a
database connection. The ``@mcp.tool`` wrapper below composes the concrete
``LinearClient`` and a memory-service adapter over the ``ogham.database`` /
``ogham.service`` facades and self-registers here; ``ogham.server`` imports
this module once for that side effect, same convention as every other file
under ``ogham/tools/``.
"""

from __future__ import annotations

import os
from typing import Protocol

from ogham import database
from ogham.app import mcp
from ogham.importers.linear import LinearClient, map_issue_to_memory
from ogham.service import store_memory_enriched
from ogham.tools.memory import get_active_profile


class _MemoryService(Protocol):
    def find_by_metadata_kv(self, key: str, value: str, profile: str) -> dict | None: ...
    def store_memory(self, content: str, metadata: dict, tags: list[str], profile: str) -> dict: ...


class _DefaultMemoryService:
    """Adapts ``ogham.database`` + ``ogham.service`` to the ``_MemoryService`` shape."""

    def find_by_metadata_kv(self, key: str, value: str, profile: str) -> dict | None:
        return database.find_by_metadata_kv(key, value, profile)

    def store_memory(self, content: str, metadata: dict, tags: list[str], profile: str) -> dict:
        return store_memory_enriched(
            content=content,
            profile=profile,
            source="linear-import",
            tags=tags,
            metadata=metadata,
        )

    def fetch_all_tracker_ids(self, profile: str) -> set[str]:
        """Fetch every ``metadata.tracker_external_id`` already stored in ``profile``.

        One backend round-trip instead of the N round-trips a per-issue
        ``find_by_metadata_kv`` scan would cost across an N-issue import run.
        """
        ids: set[str] = set()
        for memory in database.get_backend().get_all_memories_full(profile):
            tracker_id = (memory.get("metadata") or {}).get("tracker_external_id")
            if tracker_id is not None:
                ids.add(tracker_id)
        return ids


def _fetch_existing_tracker_ids(service: _MemoryService, profile: str) -> set[str] | None:
    """Fetch all existing tracker_external_ids in one call, if ``service`` supports it.

    Returns ``None`` if ``service`` doesn't implement the batch scan (e.g. a
    hand-rolled ``_MemoryService`` that only satisfies the required Protocol
    methods) -- callers fall back to a per-issue ``find_by_metadata_kv`` check
    in that case, which still works, just at N-fetch cost.
    """
    fetch_all = getattr(service, "fetch_all_tracker_ids", None)
    if fetch_all is None:
        return None
    return fetch_all(profile)


def import_linear_impl(
    *,
    client: LinearClient,
    service: _MemoryService,
    team_key: str,
    since_days: int,
    profile: str,
) -> dict:
    """Fetch Linear issues and upsert as Ogham memories. Dedupe by tracker_external_id.

    Skips issues already stored (matched on ``metadata.tracker_external_id``)
    and separately counts issues that could not be stored because inscribe
    was disabled -- ``store_memory_enriched`` returns ``{"status":
    "disabled"}`` when ``OGHAM_INSCRIBE_ENABLED=false``, same check pattern
    as ``store_decision`` in ``ogham.tools.memory``. Without this check an
    operator running with inscribe disabled would see ``imported=N`` with
    nothing actually landing in Ogham.

    Returns ``{"imported": int, "skipped": int, "disabled": int}``.
    """
    issues = client.fetch_issues(team_key, since_days)
    existing_ids = _fetch_existing_tracker_ids(service, profile)

    imported = skipped = disabled = 0
    for issue in issues:
        payload = map_issue_to_memory(issue)
        tracker_id = payload["metadata"]["tracker_external_id"]

        if existing_ids is not None:
            already_exists = tracker_id in existing_ids
        else:
            already_exists = (
                service.find_by_metadata_kv("tracker_external_id", tracker_id, profile) is not None
            )

        if already_exists:
            skipped += 1
            continue

        result = service.store_memory(
            content=payload["content"],
            metadata=payload["metadata"],
            tags=payload["tags"],
            profile=profile,
        )
        if isinstance(result, dict) and result.get("status") == "disabled":
            disabled += 1
            continue
        imported += 1

    return {"imported": imported, "skipped": skipped, "disabled": disabled}


@mcp.tool
def import_linear(team_key: str, since_days: int = 30, profile: str | None = None) -> dict:
    """Import Linear issues as Ogham memories, deduped by tracker_external_id.

    Read-only import -- ticket writes belong to the coding agent, not the
    memory layer. Use ``linearis``, the Linear MCP, or Linear's own UI to
    move state; Ogham stays the memory layer.

    Args:
        team_key: Linear team key, e.g. "TBU".
        since_days: Days lookback on the issue's updatedAt (default 30).
        profile: Target profile. Defaults to the active profile.

    Returns:
        ``{"imported": int, "skipped": int, "disabled": int}``. ``disabled``
        counts issues that would have imported but inscribe was off.

    Raises:
        ValueError: if the ``LINEAR_API_TOKEN`` env var is not set.
    """
    token = os.environ.get("LINEAR_API_TOKEN")
    if not token:
        raise ValueError("LINEAR_API_TOKEN not set")
    p = profile or get_active_profile()
    client = LinearClient(token=token)
    service = _DefaultMemoryService()
    return import_linear_impl(
        client=client,
        service=service,
        team_key=team_key,
        since_days=since_days,
        profile=p,
    )
