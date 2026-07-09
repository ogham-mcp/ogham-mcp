"""Obsidian vault importer -- pure walk + markdown->memory mapper + dedup fingerprint.

Depends on nothing beyond stdlib + pyyaml: no database, no MCP, no embedding.
The orchestration and I/O boundary live in ``ogham.tools.import_obsidian``.
Mirrors the ``ogham.importers.linear`` seam.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_MAX_NOTE_BYTES = 1_000_000


def _normalize_body(text: str) -> str:
    """Normalize line endings to ``\\n``, strip per-line trailing whitespace,
    and collapse leading/trailing blank lines.

    Keeps whitespace-only edits from forging a new fingerprint while leaving
    every real content change visible.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped = [line.rstrip() for line in unified.split("\n")]
    return "\n".join(stripped).strip("\n")


def compute_fingerprint(text: str) -> str:
    """Return the SHA-256 hex digest of the normalized body.

    Stable across whitespace-only edits, sensitive to any real content change.
    Used as the dedup key stored at ``metadata['content_fingerprint']``.
    """
    normalized = _normalize_body(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ParsedNote:
    """One Obsidian note mapped to the OM store shape."""

    content: str
    tags: list[str]
    metadata: dict
    fingerprint: str


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split a leading YAML frontmatter block from the body.

    Returns ``({}, raw)`` unchanged when there is no well-formed ``---`` fenced
    block at the very top (including on YAML parse errors), so a malformed or
    absent header degrades to "the whole file is body".
    """
    lines = raw.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            try:
                parsed = yaml.safe_load("\n".join(lines[1:i]))
            except yaml.YAMLError:
                return {}, raw
            if not isinstance(parsed, dict):
                return {}, raw
            return parsed, "\n".join(lines[i + 1 :])
    return {}, raw


def _json_safe(value):
    """Coerce YAML-native scalars that aren't JSON-serializable (date/datetime/
    time/Decimal) to strings, recursing through dicts/lists.

    Keeps ``metadata`` safe for psycopg ``Jsonb`` / ``json.dumps`` -- without
    this, an Obsidian daily note with dated frontmatter (``date: 2026-07-06``)
    would crash the store call with a non-serializable-type error.
    """
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def parse_note(rel_path: str, raw: str) -> ParsedNote:
    """Map a raw note (vault-relative path + file text) to a ``ParsedNote``."""
    frontmatter, body = _split_frontmatter(raw)

    tags: list[str] = []
    fm_tags = frontmatter.get("tags")
    if isinstance(fm_tags, list):
        tags = [str(t) for t in fm_tags]
    elif isinstance(fm_tags, str):
        tags = [fm_tags]

    fingerprint = compute_fingerprint(body)
    metadata: dict = {"vault_path": rel_path, "content_fingerprint": fingerprint}
    remaining = _json_safe({k: v for k, v in frontmatter.items() if k != "tags"})
    if remaining:
        metadata["frontmatter"] = remaining

    return ParsedNote(
        content=_normalize_body(body),
        tags=tags,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def iter_vault_notes(vault: str, *, max_bytes: int = _MAX_NOTE_BYTES) -> Iterator[Path]:
    """Yield every ingestable ``*.md`` under ``vault`` in sorted order.

    Skips dot-directories (``.obsidian``, ``.trash``, ``.git`` ...), empty
    files, and notes larger than ``max_bytes`` (logged). Sorted so a scan is
    deterministic and diffable.
    """
    root = Path(vault)
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        if size > max_bytes:
            logger.warning("obsidian: skipping oversize note %s (%d bytes)", rel, size)
            continue
        yield path
