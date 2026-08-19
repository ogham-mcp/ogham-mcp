"""Memory <-> OKF concept frontmatter marshalling."""

from datetime import datetime

from ogham.okf.entities import wiki_link

DEFAULT_OKF_TYPE = "Memory"
TYPE_TAG_PREFIX = "type:"

# Frontmatter keys this module DERIVES on every export. They are never accepted
# from stored metadata, even when the derivation produces nothing -- a stale
# value is worse than an absent one, because it names a concept the bundle does
# not contain.
_DERIVED_ONLY_FIELDS = frozenset({"MENTIONS"})

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


def memory_to_frontmatter(memory: dict, *, entity_paths: list[str] | None = None) -> dict:
    """Convert an Ogham memory record to an OKF concept frontmatter dict.

    Required: type (per OKF spec §9).
    Extensions: id, source -- preserved on round-trip per spec §4.1.
    Metadata is flattened to top-level keys so consumers can reason about
    them without knowing our convention.

    ``entity_paths`` are the bundle-relative paths (no ``.md``) of the entity
    concepts this memory links to via ``memory_entities``. Keyword-only with a
    None default so every existing caller keeps the frontmatter it had before
    the graph layer existed -- a bundle with no entities is still a valid
    bundle, and MENTIONS is purely additive on top of it.
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
        # str() for the same backend split as make_filename: psycopg hands back
        # uuid.UUID and yaml.safe_dump raises RepresenterError on it, so the
        # concept file could not be written at all on Postgres.
        "id": str(memory["id"]),
        "tags": strip_type_tags(tags),
        "timestamp": timestamp,
    }
    source = memory.get("source")
    if source:
        fm["source"] = source
    # memory_entities -> MENTIONS (D7). One of the five verified Schema.org
    # alignments (schema:mentions), and a genuine equivalence rather than a
    # stretch: the join row records that a memory mentions an entity, which is
    # exactly what schema:mentions asserts. Emitting it here puts the alignment
    # on the concepts themselves instead of only in a vocabulary file.
    # Placed BEFORE the metadata flatten so the existing `if k not in fm` guard
    # stops producer metadata clobbering a triple.
    if entity_paths:
        fm["MENTIONS"] = [wiki_link(p) for p in entity_paths]
    # Flatten metadata first so that stored title/description/resource (which
    # survive import round-trips via metadata) take precedence over the
    # auto-derived title below.
    metadata = memory.get("metadata") or {}
    for k, v in metadata.items():
        if k in _DERIVED_ONLY_FIELDS:
            # MENTIONS is computed from memory_entities, never carried. Without
            # this, a memory whose metadata happens to hold a MENTIONS key emits
            # it verbatim whenever entity_paths is empty -- and that is a real
            # round-trip path, not a hypothetical: frontmatter_to_memory files
            # every unrecognised frontmatter key into metadata, so
            # export -> import -> export re-emits the FIRST export's MENTIONS
            # as producer data, pointing at entity concepts the second bundle
            # never wrote. The `k not in fm` guard below cannot catch it,
            # because when there are no entities the key is not in fm at all.
            continue
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
