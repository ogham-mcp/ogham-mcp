"""Beads import MCP tool + CLI backend.

``import_beads_impl`` is pure over an injected client + ``IngestService``
(unit-testable with no `bd`, no DB). Tracker full-list model: list every issue
(+ comments), dedup on a namespaced ``beads_key``, ``stop_on_store_error=False``
(re-listable, like Obsidian). Mirrors ``ogham.tools.import_github``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from ogham.app import mcp
from ogham.importers.beads import BeadsClient, map_comment_to_record, map_issue_to_record
from ogham.ingest import DefaultIngestService, IngestService, run_ingest
from ogham.tools.memory import get_active_profile

logger = logging.getLogger(__name__)

_SOURCE = "beads"
_DEDUP_KEY = "beads_key"


class _BeadsClient(Protocol):
    """Structural type for ``import_beads_impl`` -- lets tests inject a fake."""

    def list_issues(self) -> list[dict]: ...
    def list_comments(self, issue_id: str) -> list[dict]: ...


def _to_record(item: dict):
    if item["_type"] == "issue":
        return map_issue_to_record(item["issue"])
    return map_comment_to_record(item["issue_id"], item["comment"])


def _collect_items(client: _BeadsClient) -> list[dict]:
    items: list[dict] = []
    for issue in client.list_issues():
        items.append({"_type": "issue", "issue": issue})
        if issue.get("comment_count") or 0:
            try:
                for comment in client.list_comments(issue["id"]):
                    items.append({"_type": "comment", "issue_id": issue["id"], "comment": comment})
            except Exception as exc:  # noqa: BLE001 -- one issue's comment fetch must not abort the run
                logger.warning("beads: comments fetch failed for %s: %s", issue.get("id"), exc)
    return items


def import_beads_impl(
    *,
    client: _BeadsClient,
    service: IngestService,
    profile: str,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """List all Beads issues + comments and store new ones. Idempotent (dedup on
    ``beads_key``). PRs n/a. A per-issue comment-fetch failure is logged and skipped."""
    existing = service.fetch_existing_keys(profile, source, _DEDUP_KEY)
    items = _collect_items(client)
    summary = run_ingest(
        items=items,
        to_record=_to_record,
        service=service,
        profile=profile,
        source=source,
        dedup_key_field=_DEDUP_KEY,
        existing=existing,
        dry_run=dry_run,
        stop_on_store_error=False,
    )
    # Return the 6 public count keys (drop run_ingest's internal `stopped`),
    # matching every sibling adapter.
    return {
        k: summary[k]
        for k in ("scanned", "stored", "skipped_duplicate", "skipped_ignored", "disabled", "errors")
    }


def _beads_dir_from_env() -> str:
    beads_dir = (os.environ.get("BEADS_DIR") or "").strip()
    if not beads_dir:
        raise ValueError("BEADS_DIR not set (path to the Beads project directory)")
    if not Path(beads_dir).is_dir():
        raise ValueError(f"BEADS_DIR is not a directory: {beads_dir}")
    return beads_dir


@mcp.tool
def import_beads(
    profile: str | None = None,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Import Beads issues + comments into Ogham, deduped by beads_key.

    Reads a local Beads project via the `bd` CLI (no network, no token). Requires
    ``BEADS_DIR`` (the project directory) and `bd` on PATH.

    Returns:
        ``{"scanned","stored","skipped_duplicate","skipped_ignored","disabled","errors"}``.

    Raises:
        ValueError: if ``BEADS_DIR`` is unset or not a directory.
        RuntimeError: if `bd` is missing or fails.
    """
    beads_dir = _beads_dir_from_env()
    target = profile or get_active_profile()
    return import_beads_impl(
        client=BeadsClient(beads_dir=beads_dir),
        service=DefaultIngestService(),
        profile=target,
        source=source,
        dry_run=dry_run,
    )
