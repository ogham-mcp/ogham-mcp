"""Memory <-> OKF concept frontmatter marshalling.

OKF v0.3 shape (TBU-297). Ogham-specific keys are `x-ogham-*` (§4.1), the
creation time is `generated: {by, at}` (§5.2, replacing v0.1 `timestamp`), a
superseded memory is `status: deprecated` with a `superseded_by` path (§5.4),
and entity links are carried both as `x-ogham-mentions` and as a marked body
section of Markdown links (§6.1). The importer reads every v0.1 spelling too.
"""

import functools
import importlib.metadata
from datetime import datetime, timezone

from ogham.okf.entities import predicate_key
from ogham.okf.links import (
    MENTIONS_SECTION,
    append_section,
    build_section,
    relative_target,
    resolve_link_note_name,
    strip_section,
    x_key,
)

DEFAULT_OKF_TYPE = "Memory"
TYPE_TAG_PREFIX = "type:"

#: Directory memory concepts are written to, bundle-relative.
MEMORIES_DIR = "memories"

#: Frontmatter keys. v0.3 name, then the v0.1 name the importer still reads.
ID_KEYS = (x_key("id"), "id")
SOURCE_KEYS = (x_key("source"), "source")
MENTIONS_KEY = predicate_key("MENTIONS")  # x-ogham-mentions
LEGACY_MENTIONS_KEY = "MENTIONS"
IMPORTANCE_KEY = x_key("importance")
#: The replacing memory's UUID. `superseded_by` is a PATH (§5.4) and is omitted
#: when the replacement is not in the bundle, so the id needs its own carrier or
#: an import -> export cycle could not reproduce `status: deprecated`.
SUPERSEDED_ID_KEY = x_key("superseded-by")
# MENTIONS_KEY and MENTIONS_SECTION are the same string on purpose: the body
# marker names the key it mirrors. tests/test_okf_links.py pins it.

# Frontmatter keys this module DERIVES on every export. They are never accepted
# from stored metadata, even when the derivation produces nothing -- a stale
# value is worse than an absent one, because it names a concept the bundle does
# not contain.
_DERIVED_ONLY_FIELDS = frozenset({LEGACY_MENTIONS_KEY, MENTIONS_KEY, SUPERSEDED_ID_KEY})

# OKF spec-recognised fields that map to memory columns or have special handling.
# Other fields fall through to metadata per spec §4.1.
# NOTE: title/description/resource are intentionally NOT listed here so that
# non-Ogham OKF bundles (which use these spec-recommended fields) preserve them
# in metadata on import and re-emit them in frontmatter on export. `status` and
# `superseded_by` are handled separately: they are consumed only when Ogham
# wrote them (see frontmatter_to_memory).
_RECOGNISED_FIELDS = {
    "type",
    "tags",
    "generated",
    "timestamp",  # v0.1
    IMPORTANCE_KEY,
    SUPERSEDED_ID_KEY,
    MENTIONS_KEY,
    LEGACY_MENTIONS_KEY,
    *ID_KEYS,
    *SOURCE_KEYS,
}

_GENERATOR_NAME = "ogham"


@functools.lru_cache(maxsize=1)
def generator_actor() -> str:
    """`ogham/<package version>` -- an OKF §7 actor for `generated.by`.

    Falls back to `ogham/dev` when the package metadata is unavailable (an
    uninstalled source tree), which is still a well-formed actor.
    """
    try:
        version = importlib.metadata.version("ogham-mcp")
    except Exception:
        version = "dev"
    return f"{_GENERATOR_NAME}/{version}"


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


def _iso_with_offset(created_at: object) -> object:
    """created_at as an ISO string, keeping the offset.

    created_at is a datetime on the Postgres backend and an ISO string on the
    Supabase REST backend (TBU-162 root cause); coercing to a string keeps the
    bundle backend-independent. A naive datetime is read as UTC -- the column is
    timestamptz, and OKF §5 rejects a date-time with no offset. Strings pass
    through untouched: they came from the database already formatted.
    """
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at.isoformat()
    return created_at


def memory_to_frontmatter(
    memory: dict,
    *,
    entity_paths: list[str] | None = None,
    superseded_by_path: str | None = None,
    concept_dir: str = MEMORIES_DIR,
) -> dict:
    """Convert an Ogham memory record to an OKF concept frontmatter dict.

    Required: type (per OKF spec §4.1).
    Extensions: x-ogham-id, x-ogham-source, x-ogham-importance -- preserved on
    round-trip per spec §4.1.
    Metadata is flattened to top-level keys so consumers can reason about
    them without knowing our convention.

    ``entity_paths`` are the bundle-relative paths (no ``.md``) of the entity
    concepts this memory links to via ``memory_entities``. They are written as
    paths relative to ``concept_dir`` (OKF §6 rule 3).

    A memory with a truthy ``superseded_by`` (the id of the newer memory that
    contradicts it) is written ``status: deprecated``. ``superseded_by_path``
    is the bundle-relative path of the replacement's concept file, or None
    when the replacement is not in the bundle -- then the path is omitted and
    only the status says the memory is not current.
    """
    tags = list(memory.get("tags") or [])
    generated: dict = {"by": generator_actor()}
    at = _iso_with_offset(memory.get("created_at"))
    if at:
        generated["at"] = at
    fm: dict = {
        "type": derive_okf_type(tags),
        # str() for the same backend split as make_filename: psycopg hands back
        # uuid.UUID and yaml.safe_dump raises RepresenterError on it, so the
        # concept file could not be written at all on Postgres.
        ID_KEYS[0]: str(memory["id"]),
        "tags": strip_type_tags(tags),
        "generated": generated,
    }
    source = memory.get("source")
    if source:
        fm[SOURCE_KEYS[0]] = source
    importance = memory.get("importance")
    if isinstance(importance, (int, float)) and not isinstance(importance, bool):
        # 4 dp on export: compute_importance sums floats, so 0.2 + 0.1 would
        # otherwise reach the bundle as 0.30000000000000004.
        fm[IMPORTANCE_KEY] = round(float(importance), 4)
    superseded_id = memory.get("superseded_by")
    if superseded_id:
        fm["status"] = "deprecated"
        if superseded_by_path:
            fm["superseded_by"] = relative_target(concept_dir, superseded_by_path)
        fm[SUPERSEDED_ID_KEY] = str(superseded_id)
    # memory_entities -> x-ogham-mentions (D7). schema:mentions is a genuine
    # equivalence: the join row records that a memory mentions an entity.
    # Placed BEFORE the metadata flatten so the `if k not in fm` guard stops
    # producer metadata clobbering it.
    if entity_paths:
        fm[MENTIONS_KEY] = [relative_target(concept_dir, f"{p}.md") for p in entity_paths]
    # Flatten metadata first so that stored title/description/resource (which
    # survive import round-trips via metadata) take precedence over the
    # auto-derived title below.
    metadata = memory.get("metadata") or {}
    for k, v in metadata.items():
        if k in _DERIVED_ONLY_FIELDS:
            # Mentions are computed from memory_entities and the superseding
            # id from the contradiction graph, never carried. Without this, a
            # memory whose metadata holds one of these keys (a v0.1 import
            # filed MENTIONS into metadata) emits it verbatim whenever the
            # derivation is empty, pointing at concepts the bundle never wrote.
            continue
        if k not in fm:  # never let metadata override spec/extension fields
            fm[k] = v
    # Only derive a title from content when one wasn't already provided by metadata.
    if "title" not in fm:
        title = _derive_title(memory.get("content") or "")
        if title:
            fm["title"] = title
    return fm


def memory_to_body(
    content: str,
    mentions: list[tuple[str, str]] | None = None,
    *,
    concept_dir: str = MEMORIES_DIR,
) -> str:
    """Concept body: the memory content plus the marked Mentions section.

    ``mentions`` are ``(link text, bundle-relative entity path without .md)``.
    Without mentions the body is the content, exactly as before.
    """
    if not mentions:
        return content
    items = [("", text, relative_target(concept_dir, f"{path}.md")) for text, path in mentions]
    return append_section(content, build_section(MENTIONS_SECTION, "Mentions", items))


def _first(fm: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in fm:
            return fm[key]
    return None


def _importance(value: object) -> float | None:
    """A usable importance from frontmatter, or None. Import is a trust boundary."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def frontmatter_to_memory(fm: dict, body: str) -> dict:
    """Convert an OKF concept (frontmatter + body) back to a memory record.

    `id` is None if absent (caller mints a new UUID); the OKF type is converted
    back to a `type:X` tag unless it's the default ("Memory"). Unknown frontmatter
    keys are stored in metadata to satisfy spec §4.1 round-trip preservation.

    Reads both shapes: v0.3 (`x-ogham-id`, `x-ogham-source`, `generated.at`,
    `x-ogham-mentions` paths) and v0.1 (`id`, `source`, `timestamp`, `MENTIONS`
    wiki links). The new key wins when a file carries both.

    The returned ``mentions`` are target note names. ``superseded_by`` is the
    replacing memory's id; `status`/`superseded_by` are consumed only when
    Ogham wrote them (the `x-ogham-superseded-by` id is present) and otherwise
    stay in metadata as foreign data.
    """
    # Tag order is not preserved across round-trip; consumers should compare as sets.
    tags = list(fm.get("tags") or [])
    okf_type = fm.get("type") or DEFAULT_OKF_TYPE
    if okf_type != DEFAULT_OKF_TYPE:
        tag = f"{TYPE_TAG_PREFIX}{okf_type.lower()}"
        if tag not in tags:
            tags.append(tag)

    generated = fm.get("generated")
    timestamp = generated.get("at") if isinstance(generated, dict) else None
    if timestamp is None:
        timestamp = fm.get("timestamp")  # v0.1, OKF Appendix B fallback

    raw_mentions = fm.get(MENTIONS_KEY)
    has_section_key = MENTIONS_KEY in fm
    if raw_mentions is None:
        raw_mentions = fm.get(LEGACY_MENTIONS_KEY)
    mentions = [
        name
        for name in (resolve_link_note_name(v) for v in _as_list(raw_mentions))
        if name is not None
    ]

    superseded_id = fm.get(SUPERSEDED_ID_KEY)
    ogham_lifecycle = bool(superseded_id)

    metadata: dict = {}
    for k, v in fm.items():
        if k in _RECOGNISED_FIELDS:
            continue
        if ogham_lifecycle and k in ("status", "superseded_by"):
            continue
        metadata[k] = v

    # Strip the body section only when the structured key says one was written.
    # A body that ends in a hand-written block of the same shape, with no key,
    # is content and is left alone.
    content = strip_section(body, MENTIONS_SECTION) if has_section_key else body

    return {
        "id": _first(fm, ID_KEYS),
        "content": content,
        "tags": tags,
        "timestamp": timestamp,
        "source": _first(fm, SOURCE_KEYS),
        "importance": _importance(fm.get(IMPORTANCE_KEY)),
        "superseded_by": str(superseded_id) if superseded_id else None,
        "mentions": mentions,
        "metadata": metadata,
    }


def _as_list(value: object) -> list:
    """OKF §11.2: a single item stands for a list of one; null means absent."""
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _derive_title(content: str) -> str:
    """First non-empty line, capped at 80 chars."""
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


__all__ = [
    "DEFAULT_OKF_TYPE",
    "MEMORIES_DIR",
    "derive_okf_type",
    "frontmatter_to_memory",
    "generator_actor",
    "memory_to_body",
    "memory_to_frontmatter",
    "strip_type_tags",
]
