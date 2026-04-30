"""Parser for Claude Code's local-memory markdown files.

Claude Code stores per-project auto-memory at
``~/.claude/projects/<encoded-cwd>/memory/`` as one ``MEMORY.md`` index
plus N child ``.md`` files. Each child has a YAML frontmatter block::

    ---
    name: <title>
    description: <one-line hook used in the index>
    type: user | feedback | project | reference
    originSessionId: <optional uuid>
    ---
    <markdown body>

This module walks such a directory, parses the frontmatter, and emits a
list of memory dicts in the shape ``export_import.import_memories``
expects (``{"content", "tags", "source", "metadata"}``). Mapping:

* ``content``     -- the markdown body after the frontmatter
* ``tags``        -- ``["source:claude-code-memory", f"type:{frontmatter.type}",
                       f"project:{<inferred>}"]``
* ``source``      -- ``"claude-code-memory"``
* ``metadata``    -- ``{"name", "description", "origin_session_id",
                       "source_file", "claude_code_type"}``

``MEMORY.md`` is the index file and is skipped (it lists the others by
title -- not content worth storing on its own).

Frontmatter parsing uses pyyaml (already a dependency). Files without a
recognisable frontmatter block are skipped with a warning so a partial
walk still yields something useful.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ogham.export_import import import_memories as _import_memories

logger = logging.getLogger(__name__)

# Frontmatter block at the top of the file: opening "---" line, body
# lines, closing "---" line. Non-greedy on the body so a stray "---"
# elsewhere doesn't expand the match.
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

# Skip these exact filenames -- they are index/structural files, not
# memory content.
_SKIP_FILENAMES = frozenset({"MEMORY.md"})

# The Claude Code project-directory naming scheme replaces every ``/`` in
# the absolute cwd with ``-`` and prepends one leading ``-``. Inverse
# mapping below recovers a usable project tag.
_ENCODED_CWD_PREFIX = "-Users-"


def _infer_project_tag(directory: Path) -> str | None:
    """Recover a project tag from Claude Code's encoded directory name.

    For ``~/.claude/projects/-Users-someone-Developer-web-projects-foo/memory/``
    we want ``project:foo``. Walks up to find the encoded segment, then
    takes the last path-like piece. Returns ``None`` if the directory
    doesn't look like a Claude Code projects path -- callers can still
    import, just without the project tag.
    """
    for parent in (directory, *directory.parents):
        name = parent.name
        if name.startswith(_ENCODED_CWD_PREFIX):
            # Last hyphen-separated segment is the project basename. We
            # lower-case it because tags are case-sensitive in search and
            # most tag schemes use lowercase.
            tail = name.rsplit("-", maxsplit=1)[-1]
            return tail.lower() if tail else None
    return None


def parse_memory_file(path: Path, *, project_tag: str | None = None) -> dict[str, Any] | None:
    """Parse a single Claude Code memory ``.md`` file.

    Returns a memory dict ready for ``import_memories``, or ``None`` if
    the file lacks a frontmatter block or carries a corrupt one. Errors
    are logged at WARNING -- a single bad file doesn't kill the walk.

    ``project_tag`` overrides the directory-inferred project name (the
    encoded-cwd heuristic is lossy when the original path contains
    hyphens, e.g. ``openbrain-sharedmemory`` decodes to ``sharedmemory``;
    pass ``project_tag="ogham"`` to keep tags consistent across imports).
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        logger.warning("claude-code import: no frontmatter in %s -- skipping", path)
        return None

    raw_frontmatter, body = match.group(1), match.group(2)
    try:
        fm = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError as e:
        logger.warning("claude-code import: invalid YAML in %s: %s", path, e)
        return None
    if not isinstance(fm, dict):
        logger.warning("claude-code import: frontmatter is not a mapping in %s -- skipping", path)
        return None

    # Body is the actual memory content. Strip leading blank lines so
    # the embedding doesn't see decorative whitespace.
    content = body.lstrip("\n").rstrip()
    if not content:
        logger.warning("claude-code import: empty body in %s -- skipping", path)
        return None

    cc_type = fm.get("type") or "unknown"
    tags = ["source:claude-code-memory", f"type:{cc_type}"]

    effective_project = project_tag or _infer_project_tag(path.parent)
    if effective_project:
        tags.append(f"project:{effective_project}")

    metadata = {
        "name": fm.get("name") or path.stem,
        "description": fm.get("description") or "",
        "claude_code_type": cc_type,
        "source_file": str(path),
    }
    origin = fm.get("originSessionId")
    if origin:
        metadata["origin_session_id"] = origin

    return {
        "content": content,
        "tags": tags,
        "source": "claude-code-memory",
        "metadata": metadata,
    }


def find_memory_files(directory: Path) -> list[Path]:
    """List importable ``.md`` files under ``directory``.

    Excludes ``MEMORY.md`` (the index) and dotfiles. Sorted by name so
    repeated runs touch the same files in the same order -- useful for
    deterministic dedup and predictable progress reporting.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(
        p
        for p in directory.glob("*.md")
        if p.name not in _SKIP_FILENAMES and not p.name.startswith(".")
    )


def parse_claude_code_memories(
    directory: Path, *, project_tag: str | None = None
) -> list[dict[str, Any]]:
    """Walk ``directory`` and return parsed memory dicts.

    Files without recognisable frontmatter are silently skipped (a
    WARNING log entry is emitted per file). Empty bodies are skipped.
    Order matches ``find_memory_files`` (sorted by filename).

    ``project_tag`` overrides the directory-inferred project name on every
    parsed memory.
    """
    out: list[dict[str, Any]] = []
    for path in find_memory_files(directory):
        parsed = parse_memory_file(path, project_tag=project_tag)
        if parsed is not None:
            out.append(parsed)
    return out


def import_claude_code_memories(
    directory: Path | str,
    profile: str,
    dedup_threshold: float = 0.8,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_embed_progress: Callable[[int, int], None] | None = None,
    project_tag: str | None = None,
) -> dict[str, Any]:
    """Parse a Claude Code memory directory and import into a profile.

    Wraps ``parse_claude_code_memories`` -> JSON envelope ->
    ``export_import.import_memories``. Dedup defaults to 0.8 (cosine
    similarity) so re-running the importer against the same directory
    is cheap.

    Pass ``project_tag`` to override the directory-inferred project name
    (the encoded-cwd heuristic is lossy on hyphenated repo names).
    """
    directory = Path(directory).expanduser()
    parsed = parse_claude_code_memories(directory, project_tag=project_tag)

    if not parsed:
        return {
            "status": "complete",
            "profile": profile,
            "imported": 0,
            "skipped": 0,
            "total": 0,
            "directory": str(directory),
            "warning": "no memory files found",
        }

    envelope = json.dumps({"memories": parsed})
    result = _import_memories(
        envelope,
        profile=profile,
        dedup_threshold=dedup_threshold,
        on_progress=on_progress,
        on_embed_progress=on_embed_progress,
    )
    result["directory"] = str(directory)
    return result
