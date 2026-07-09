"""GitHub Issues import MCP tool + CLI backend.

``import_github_impl`` is pure over an injected client + ``IngestService``
(unit-testable with no network, no DB). Tracker model: fetch issues updated in
the last ``since_days`` days (+ their comments), dedup on a namespaced
``github_key``, ``stop_on_store_error=False`` (re-fetchable, like Obsidian).
Mirrors ``ogham.tools.import_slack``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Protocol

from ogham.app import mcp
from ogham.importers.github_issues import (
    GitHubClient,
    map_comment_to_record,
    map_issue_to_record,
)
from ogham.ingest import DefaultIngestService, IngestService, run_ingest
from ogham.tools.memory import get_active_profile

logger = logging.getLogger(__name__)

_SOURCE = "github"
_DEDUP_KEY = "github_key"


class _IssuesClient(Protocol):
    """Structural type for ``import_github_impl`` -- lets tests inject a fake."""

    def list_issues(
        self, repo: str, since: str | None = None, page: int = 1, per_page: int = 100
    ) -> tuple[list[dict], int | None]: ...
    def list_comments(
        self, repo: str, issue_number: int, page: int = 1, per_page: int = 100
    ) -> tuple[list[dict], int | None]: ...


def _since_iso(since_days: int) -> str:
    """ISO 8601 ``YYYY-MM-DDTHH:MM:SSZ`` for ``now - since_days`` (UTC)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_record(item: dict):
    if item["_type"] == "issue":
        return map_issue_to_record(item["repo"], item["issue"])
    return map_comment_to_record(item["repo"], item["issue"], item["comment"])


def _collect_repo_items(client: _IssuesClient, repo: str, since: str) -> list[dict]:
    items: list[dict] = []
    page: int | None = 1
    while page is not None:
        issues, page = client.list_issues(repo, since=since, page=page)
        for issue in issues:
            if "pull_request" in issue:
                continue  # PR, not an issue
            items.append({"_type": "issue", "repo": repo, "issue": issue})
            try:
                cpage: int | None = 1
                while cpage is not None:
                    comments, cpage = client.list_comments(repo, issue["number"], page=cpage)
                    for comment in comments:
                        items.append(
                            {"_type": "comment", "repo": repo, "issue": issue, "comment": comment}
                        )
            except Exception as exc:  # noqa: BLE001 -- one issue's comment fetch must not abort the repo
                logger.warning(
                    "github: comments fetch failed for %s#%s: %s", repo, issue.get("number"), exc
                )
    return items


def import_github_impl(
    *,
    client: _IssuesClient,
    service: IngestService,
    profile: str,
    repos: list[str],
    since_days: int = 30,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Fetch issues + comments updated in the last ``since_days`` days and store
    new ones. Idempotent; safe to re-run (dedup on ``github_key``). PRs are
    filtered out. A per-issue comment-fetch failure is logged and skipped. A
    whole-repo ``list_issues`` failure is counted as an error and does not
    abort the other repos in ``repos``."""
    existing = service.fetch_existing_keys(profile, source, _DEDUP_KEY)
    since = _since_iso(since_days)
    totals = {
        "scanned": 0,
        "stored": 0,
        "skipped_duplicate": 0,
        "skipped_ignored": 0,
        "disabled": 0,
        "errors": 0,
    }
    for repo in repos:
        try:
            items = _collect_repo_items(client, repo, since)
        except Exception as exc:  # noqa: BLE001 -- a bad repo (typo / lost access / transient) must not sink the others
            logger.warning("github: repo %s failed: %s", repo, exc)
            totals["errors"] += 1
            continue
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
        for key in totals:
            totals[key] += summary[key]
    return totals


def _repos_from_env() -> list[str]:
    raw = os.environ.get("GITHUB_REPOS")
    repos = [r.strip() for r in (raw or "").split(",") if r.strip()]
    if not repos:
        raise ValueError("GITHUB_REPOS not set (comma-separated owner/repo)")
    return repos


@mcp.tool
def import_github(
    profile: str | None = None,
    since_days: int = 30,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Import GitHub issues + comments into Ogham, deduped by github_key.

    Fetches issues updated in the last ``since_days`` days from ``GITHUB_REPOS``.
    Requires ``GITHUB_TOKEN``. Outbound REST -- no webhook, no exposed endpoint.

    Returns:
        ``{"scanned","stored","skipped_duplicate","skipped_ignored","disabled","errors"}``.

    Raises:
        ValueError: if ``GITHUB_TOKEN`` or ``GITHUB_REPOS`` is not set.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not set")
    target = profile or get_active_profile()
    return import_github_impl(
        client=GitHubClient(token=token),
        service=DefaultIngestService(),
        profile=target,
        repos=_repos_from_env(),
        since_days=since_days,
        source=source,
        dry_run=dry_run,
    )
