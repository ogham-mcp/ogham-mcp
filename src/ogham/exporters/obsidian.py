"""Obsidian vault exporter for the wiki layer.

Dumps `topic_summaries` for a profile as an Obsidian-compatible vault:
one markdown file per topic with YAML frontmatter, plus a README index.
Inter-topic references in the body are rewritten as `[[wikilinks]]`
when they match a known topic_key, giving Obsidian's graph view
something to chew on.

Read-only by design. The exporter never writes to the database --
the vault is a snapshot, not a sync target. Bidirectional sync (vault
edits flowing back into Ogham) is its own product.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ogham.database import get_backend

logger = logging.getLogger(__name__)


# Obsidian / cross-platform filesystem reserved characters. Strip these
# rather than URL-encoding -- the vault is meant to be human-readable.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_HYPHEN = re.compile(r"-{2,}")


def slugify(topic_key: str) -> str:
    """Turn a topic_key into a filesystem-safe filename stem.

    Preserves case + non-ASCII (Obsidian handles unicode filenames
    fine) -- only strips reserved chars and collapses runs of hyphens.
    Empty or all-unsafe input falls back to "untitled".
    """
    cleaned = _UNSAFE_CHARS.sub("-", topic_key.strip())
    cleaned = _MULTI_HYPHEN.sub("-", cleaned).strip("-")
    return cleaned or "untitled"


@dataclass
class ExportResult:
    vault_path: Path
    profile: str
    topics_written: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _format_frontmatter(summary: dict[str, Any], topic_keys: set[str]) -> str:
    """YAML frontmatter block with provenance + Obsidian-friendly extras."""
    source_hash = summary.get("source_hash")
    if isinstance(source_hash, (bytes, bytearray, memoryview)):
        source_hash_hex = bytes(source_hash).hex()
    elif isinstance(source_hash, str):
        source_hash_hex = source_hash.removeprefix("\\x") or None
    else:
        source_hash_hex = None

    lines = [
        "---",
        f"ogham_id: {summary['id']}",
        f"topic_key: {summary['topic_key']}",
        f"profile: {summary['profile_id']}",
        f"version: {summary['version']}",
        f"status: {summary['status']}",
        f"source_count: {summary['source_count']}",
        f"model_used: {summary['model_used']}",
        f"updated_at: {summary['updated_at']}",
    ]
    if source_hash_hex:
        lines.append(f"source_hash: {source_hash_hex}")
    # Obsidian convention: tags as a YAML list. Surface the topic_key
    # itself plus a generic "ogham/wiki" tag so the vault's tag pane
    # has a sensible top-level grouping.
    lines.append("tags:")
    lines.append("  - ogham/wiki")
    lines.append(f"  - {slugify(summary['topic_key']).lower()}")
    lines.append("---")
    return "\n".join(lines)


_WIKILINK_SAFE = re.compile(r"[A-Za-z0-9_\-]+")


def _rewrite_wikilinks(body: str, topic_keys: set[str], self_key: str) -> str:
    """Wrap recognised topic_key tokens in [[ ]].

    Conservative match: only rewrites tokens that already look like
    valid identifiers and that aren't already inside a markdown link
    or code span. We do a simple word-boundary substitution per known
    topic_key. Skips the topic's own key so a page doesn't link to
    itself, and skips occurrences inside backtick spans / fenced blocks
    via a coarse heuristic (line starts with ``` or token is wrapped
    in backticks).
    """
    if not topic_keys:
        return body

    out_lines: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        rewritten = line
        for key in topic_keys:
            if key == self_key:
                continue
            if not _WIKILINK_SAFE.fullmatch(key):
                # Skip topic_keys with chars that would confuse the regex
                # (e.g. slashes). Listing them in [[ ]] still works in
                # Obsidian but the boundary regex gets fragile.
                continue
            pattern = re.compile(rf"(?<![\w\[`]){re.escape(key)}(?![\w\]`])")
            rewritten = pattern.sub(f"[[{key}]]", rewritten)
        out_lines.append(rewritten)
    return "\n".join(out_lines)


def _format_topic_file(summary: dict[str, Any], topic_keys: set[str]) -> str:
    frontmatter = _format_frontmatter(summary, topic_keys)
    body = summary.get("content") or ""
    body = _rewrite_wikilinks(body, topic_keys, summary["topic_key"])
    return f"{frontmatter}\n\n{body}\n"


def _format_index(profile: str, summaries: list[dict[str, Any]]) -> str:
    """Vault-root README listing every exported topic."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"# Ogham wiki -- profile `{profile}`",
        "",
        f"Exported {now}. {len(summaries)} topic(s).",
        "",
        "## Topics",
        "",
    ]
    for s in summaries:
        key = s["topic_key"]
        slug = slugify(key)
        status = s.get("status", "?")
        count = s.get("source_count", "?")
        lines.append(f"- [[{slug}|{key}]] -- {status}, {count} source(s)")
    lines.append("")
    return "\n".join(lines)


def export_to_vault(
    vault_path: Path,
    profile: str,
    *,
    force: bool = False,
) -> ExportResult:
    """Dump topic_summaries for `profile` as an Obsidian vault at `vault_path`.

    Creates `<vault>/<topic-slug>.md` per topic_summary and a `README.md`
    index at the root. Idempotent: rerunning replaces the previous
    snapshot. With `force=False`, refuses to write into a directory
    that already contains files we did not previously create -- this
    is a safety net so users don't accidentally clobber a hand-curated
    vault.
    """
    vault_path = Path(vault_path).expanduser().resolve()
    result = ExportResult(vault_path=vault_path, profile=profile)

    summaries = get_backend().wiki_topic_list_all(profile)
    if not summaries:
        result.errors.append(
            f"profile {profile!r} has no topic_summaries -- "
            "compile some with the `compile_wiki` MCP tool first"
        )
        return result

    vault_path.mkdir(parents=True, exist_ok=True)

    if not force and any(_unfamiliar_files(vault_path)):
        result.errors.append(
            f"vault directory {vault_path} already contains files we did not "
            "create. Re-run with --force to overwrite, or pick an empty path."
        )
        return result

    topic_keys = {s["topic_key"] for s in summaries}

    for summary in summaries:
        key = summary["topic_key"]
        slug = slugify(key)
        target = vault_path / f"{slug}.md"
        try:
            target.write_text(_format_topic_file(summary, topic_keys), encoding="utf-8")
            result.topics_written += 1
        except OSError as exc:
            result.errors.append(f"{key}: {exc}")
            result.skipped.append(key)

    (vault_path / "README.md").write_text(_format_index(profile, summaries), encoding="utf-8")

    return result


def _unfamiliar_files(vault_path: Path) -> list[Path]:
    """Files in the vault root that don't look like a previous export.

    A previous export writes only `*.md` files. Any other file (or any
    subdirectory we didn't create) is treated as user content and the
    exporter refuses to overwrite without --force.
    """
    out: list[Path] = []
    for entry in vault_path.iterdir():
        if entry.is_dir() and entry.name not in {".obsidian"}:
            out.append(entry)
        elif entry.is_file() and entry.suffix != ".md":
            out.append(entry)
    return out
