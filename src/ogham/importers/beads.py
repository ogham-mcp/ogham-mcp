"""Beads importer -- read issues + comments from a local Beads project via the `bd` CLI.

The client shells out to `bd` (a local CLI over a local Dolt DB -- no network, no
token) through an injectable ``run`` callable so tests need no real `bd` and no
Beads project. Mappers are pure. Sibling of ``ogham.importers.github_issues``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

from ogham.ingest import IngestRecord

Runner = Callable[[list[str], str], str]  # (argv, cwd) -> stdout


def _default_run(args: list[str], cwd: str) -> str:
    try:
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError:
        raise RuntimeError(f"{args[0]} not found on PATH") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:200]
        cmd = " ".join(args[:2])
        raise RuntimeError(f"{cmd} failed (exit {proc.returncode}): {detail}") from None
    return proc.stdout


class BeadsClient:
    """Thin `bd` CLI wrapper scoped to what the importer needs."""

    def __init__(self, beads_dir: str, run: Runner | None = None, bd_bin: str = "bd"):
        self._dir = beads_dir
        self._run = run or _default_run
        self._bd = bd_bin

    def _json(self, args: list[str]) -> list[dict]:
        out = self._run([self._bd, *args], self._dir)
        if not out.strip():
            return []
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            raise RuntimeError(f"bd {' '.join(args)}: invalid JSON output") from None
        return data if isinstance(data, list) else []

    def list_issues(self) -> list[dict]:
        # --limit 0 == unlimited (default is 50); --all includes closed issues.
        return self._json(["list", "--all", "--json", "--limit", "0"])

    def list_comments(self, issue_id: str) -> list[dict]:
        return self._json(["comments", issue_id, "--json"])


def map_issue_to_record(issue: dict) -> IngestRecord | None:
    """Map a `bd list` issue object to an IngestRecord, or None if empty."""
    issue_id = issue.get("id")
    if not issue_id:
        return None
    title = issue.get("title") or ""
    description = issue.get("description") or ""
    content = f"{title}\n\n{description}".strip()
    if not content:
        return None
    labels = [lbl for lbl in (issue.get("labels") or []) if isinstance(lbl, str) and lbl]
    metadata = {
        "beads_key": f"issue:{issue_id}",
        "beads_id": issue_id,
        "status": issue.get("status"),
        "priority": issue.get("priority"),
        "issue_type": issue.get("issue_type"),
        "assignee": issue.get("assignee"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "dependency_count": issue.get("dependency_count"),
        "dependent_count": issue.get("dependent_count"),
        "comment_count": issue.get("comment_count"),
    }
    return IngestRecord(content=content, tags=labels, metadata=metadata)


def map_comment_to_record(issue_id: str, comment: dict) -> IngestRecord | None:
    """Map a `bd comments` object to an IngestRecord, or None if it has no text."""
    comment_id = comment.get("id")
    if not comment_id:
        return None
    text = comment.get("text")
    if not text:
        return None
    metadata = {
        "beads_key": f"comment:{comment_id}",
        "comment_id": comment_id,
        "issue_id": issue_id,
        "author": comment.get("author"),
        "created_at": comment.get("created_at"),
    }
    return IngestRecord(content=text, tags=[], metadata=metadata)
