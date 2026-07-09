"""GitHub Issues importer -- fetch issues + comments via REST, map to IngestRecord.

Outbound-only: calls out to api.github.com with a Bearer token header (no token
in the URL). Mappers are pure; the client wraps httpx (injectable for tests),
same shape as ``ogham.importers.slack.SlackClient``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ogham.ingest import IngestRecord

GITHUB_API = "https://api.github.com"


class GitHubClient:
    """Thin issues/comments client scoped to what the importer needs."""

    def __init__(self, token: str, http_client: httpx.Client | None = None):
        self._token = token
        self._http = http_client or httpx.Client(timeout=30.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            resp = self._http.get(f"{GITHUB_API}{path}", params=params, headers=self._headers())
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"github {path} HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise RuntimeError(f"github {path} request failed: {type(exc).__name__}") from None
        return resp.json()

    def list_issues(
        self, repo: str, since: str | None = None, page: int = 1, per_page: int = 100
    ) -> tuple[list[dict], int | None]:
        """Return ``(issues, next_page)`` for one page. ``state=all``; ``since`` is an
        ISO 8601 updated-after filter. A full page implies there may be more."""
        params: dict[str, Any] = {"state": "all", "per_page": per_page, "page": page}
        if since is not None:
            params["since"] = since
        issues = self._get(f"/repos/{repo}/issues", params)
        issues = issues if isinstance(issues, list) else []
        next_page = page + 1 if len(issues) == per_page else None
        return issues, next_page

    def list_comments(
        self, repo: str, issue_number: int, page: int = 1, per_page: int = 100
    ) -> tuple[list[dict], int | None]:
        params: dict[str, Any] = {"per_page": per_page, "page": page}
        comments = self._get(f"/repos/{repo}/issues/{issue_number}/comments", params)
        comments = comments if isinstance(comments, list) else []
        next_page = page + 1 if len(comments) == per_page else None
        return comments, next_page


def map_issue_to_record(repo: str, issue: dict) -> IngestRecord | None:
    """Map a GitHub issue to an IngestRecord, or None if it is a pull request or empty."""
    if "pull_request" in issue:
        return None  # the issues endpoint also returns PRs; skip them
    issue_id = issue.get("id")
    if issue_id is None:
        return None
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    content = f"{title}\n\n{body}".strip()
    if not content:
        return None
    labels = [
        lbl["name"]
        for lbl in (issue.get("labels") or [])
        if isinstance(lbl, dict) and lbl.get("name")
    ]
    metadata = {
        "github_key": f"issue:{issue_id}",
        "github_id": issue_id,
        "repo": repo,
        "number": issue.get("number"),
        "state": issue.get("state"),
        "url": issue.get("html_url"),
        "author": (issue.get("user") or {}).get("login"),
        "updated_at": issue.get("updated_at"),
    }
    return IngestRecord(content=content, tags=labels, metadata=metadata)


def map_comment_to_record(repo: str, issue: dict, comment: dict) -> IngestRecord | None:
    """Map an issue comment to an IngestRecord, or None if it has no body."""
    comment_id = comment.get("id")
    if comment_id is None:
        return None
    body = comment.get("body")
    if not body:
        return None
    metadata = {
        "github_key": f"comment:{comment_id}",
        "github_id": comment_id,
        "repo": repo,
        "issue_number": issue.get("number"),
        "url": comment.get("html_url"),
        "author": (comment.get("user") or {}).get("login"),
        "created_at": comment.get("created_at"),
    }
    return IngestRecord(content=body, tags=[], metadata=metadata)
