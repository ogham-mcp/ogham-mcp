"""Memory <-> OKF concept frontmatter marshalling."""

from datetime import datetime

DEFAULT_OKF_TYPE = "Memory"
TYPE_TAG_PREFIX = "type:"

# OKF spec-recognised fields that map to memory columns or have special handling.
# Other fields fall through to metadata per spec §4.1.
# NOTE: title/description/resource are intentionally NOT listed here so that
# non-Ogham OKF bundles (which use these spec-recommended fields) preserve them
# in metadata on import and re-emit them in frontmatter on export.
_RECOGNISED_FIELDS = {
    "type",
    "id",
    "tags",
    "timestamp",
    "source",
}


def derive_okf_type(tags: list[str]) -> str:
    """Pick the OKF concept type from a memory's tags.

    Rule (locked in v0.15 plan): the first `type:X` tag alphabetically wins,
    title-cased. Memories with no `type:X` tag default to "Memory".
    Tags of the form `type:` (empty value) are skipped -- they would violate
    spec §9 (type MUST be non-empty string).
    """
    type_tags = sorted(
        t for t in tags if t.startswith(TYPE_TAG_PREFIX) and len(t) > len(TYPE_TAG_PREFIX)
    )
    if not type_tags:
        return DEFAULT_OKF_TYPE
    raw = type_tags[0][len(TYPE_TAG_PREFIX) :]
    return raw[:1].upper() + raw[1:]


def strip_type_tags(tags: list[str]) -> list[str]:
    """Remove the winning type:X tag (the one that became the OKF type).

    Other type:X tags are preserved as tags so the round-trip can reconstruct them.
    Empty-value `type:` tags are not type tags per derive_okf_type and are kept as-is.
    """
    type_tags = sorted(
        t for t in tags if t.startswith(TYPE_TAG_PREFIX) and len(t) > len(TYPE_TAG_PREFIX)
    )
    if not type_tags:
        return list(tags)
    winner = type_tags[0]
    return [t for t in tags if t != winner]


def memory_to_frontmatter(memory: dict) -> dict:
    """Convert an Ogham memory record to an OKF concept frontmatter dict.

    Required: type (per OKF spec §9).
    Extensions: id, source -- preserved on round-trip per spec §4.1.
    Metadata is flattened to top-level keys so consumers can reason about
    them without knowing our convention.
    """
    tags = list(memory.get("tags") or [])
    # created_at is a datetime on the Postgres backend, an ISO string on the
    # Supabase REST backend (TBU-162 root cause). Coerce to a string so the
    # exported frontmatter is backend-independent -- otherwise yaml.safe_dump
    # writes it as an unquoted YAML timestamp scalar for one backend and a
    # quoted string for the other. Match export_import.py's manifest
    # timestamps (`datetime.now(timezone.utc).isoformat()`) so there's one
    # ISO shape across the whole OKF bundle.
    created_at = memory.get("created_at")
    timestamp = created_at.isoformat() if isinstance(created_at, datetime) else created_at
    fm: dict = {
        "type": derive_okf_type(tags),
        "id": memory["id"],
        "tags": strip_type_tags(tags),
        "timestamp": timestamp,
    }
    source = memory.get("source")
    if source:
        fm["source"] = source
    # Flatten metadata first so that stored title/description/resource (which
    # survive import round-trips via metadata) take precedence over the
    # auto-derived title below.
    metadata = memory.get("metadata") or {}
    for k, v in metadata.items():
        if k not in fm:  # never let metadata override spec/extension fields
            fm[k] = v
    # Only derive a title from content when one wasn't already provided by metadata.
    if "title" not in fm:
        title = _derive_title(memory.get("content") or "")
        if title:
            fm["title"] = title
    return fm


def frontmatter_to_memory(fm: dict, body: str) -> dict:
    """Convert an OKF concept (frontmatter + body) back to a memory record.

    `id` is None if absent (caller mints a new UUID); the OKF type is converted
    back to a `type:X` tag unless it's the default ("Memory"). Unknown frontmatter
    keys are stored in metadata to satisfy spec §4.1 round-trip preservation.
    """
    # Tag order is not preserved across round-trip; consumers should compare as sets.
    tags = list(fm.get("tags") or [])
    okf_type = fm.get("type") or DEFAULT_OKF_TYPE
    if okf_type != DEFAULT_OKF_TYPE:
        tag = f"{TYPE_TAG_PREFIX}{okf_type.lower()}"
        if tag not in tags:
            tags.append(tag)
    metadata: dict = {}
    for k, v in fm.items():
        if k not in _RECOGNISED_FIELDS:
            metadata[k] = v
    return {
        "id": fm.get("id"),
        "content": body,
        "tags": tags,
        "timestamp": fm.get("timestamp"),
        "source": fm.get("source"),
        "metadata": metadata,
    }


def _derive_title(content: str) -> str:
    """First non-empty line, capped at 80 chars."""
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""
