"""Filename + slug generation for OKF concepts."""

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(content: str, max_len: int = 60) -> str:
    """Convert content to a filesystem-safe slug.

    Lowercase ASCII letters + digits only, hyphen-separated. Falls back to
    "untitled" for empty/whitespace-only content. Caps at max_len chars,
    splitting at the last hyphen so we don't truncate mid-word.
    """
    lowered = content.lower()
    cleaned = _NON_ALNUM.sub("-", lowered).strip("-")
    if not cleaned:
        return "untitled"
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 0:
        return truncated[:last_hyphen]
    return truncated


def make_filename(memory: dict) -> str:
    """Build the OKF concept filename for a memory.

    Pattern: {slug}-{first-8-hex-of-uuid}.md
    Locked in the v0.15 design plan; do not change without a migration.
    """
    slug = slugify(memory.get("content", ""))
    # str() because the two backends disagree on this column's Python type:
    # psycopg maps a uuid column to uuid.UUID, PostgREST returns a string. The
    # raw value went straight into .replace() from v0.15.0, so OKF export
    # crashed on the first memory on every Postgres install while working fine
    # on Supabase. Every fixture in the suite used a string id, so nothing
    # caught it until an export was run against a real database.
    memory_id = str(memory["id"])
    uuid8 = memory_id.replace("-", "")[:8]
    return f"{slug}-{uuid8}.md"
