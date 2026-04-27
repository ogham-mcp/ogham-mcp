import functools
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Any

from fastmcp import Context
from pydantic import BeforeValidator

from ogham.app import mcp
from ogham.config import settings
from ogham.database import (
    batch_update_embeddings,
    get_all_memories_content,
    list_recent_memories,
    record_access,
)
from ogham.database import cleanup_expired as db_cleanup_expired
from ogham.database import count_expired as db_count_expired
from ogham.database import create_relationship as db_create_relationship
from ogham.database import delete_memory as db_delete
from ogham.database import explore_memory_graph as db_explore_graph
from ogham.database import get_related_memories as db_get_related
from ogham.database import link_unlinked_memories as db_link_unlinked
from ogham.database import list_profiles as db_list_profiles
from ogham.database import set_profile_ttl as db_set_profile_ttl
from ogham.database import update_confidence as db_update_confidence
from ogham.database import update_memory as db_update
from ogham.embeddings import clear_embedding_cache, generate_embedding, generate_embeddings_batch
from ogham.export_import import export_memories as _export_memories
from ogham.export_import import import_memories as _import_memories
from ogham.extraction import extract_dates
from ogham.health import full_health_check
from ogham.lifecycle import advance_stages

logger = logging.getLogger(__name__)

# === FastMCP list/dict coercion wrappers ===
# Some FastMCP clients serialise list[str] and dict[str, Any] tool parameters
# as JSON strings before the transport layer. Pydantic on the server then sees
# a `str` and fails list_type / dict_type validation, surfacing as
# JSON-RPC -32602 Invalid params. BeforeValidator-based coercion accepts
# either the native shape or its JSON-string form and returns the canonical
# Python shape. Same idiom as fastmcp/utilities/components.py:82.


def _coerce_list(v):
    if v is None or isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: a bare string that isn't JSON becomes a 1-element list.
        # Pragmatic for tag-like fields where a single bare tag is common.
        return [v]
    raise TypeError(f"Cannot coerce {type(v).__name__} to list")


def _coerce_dict(v):
    if v is None or isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    raise TypeError(f"Cannot coerce {type(v).__name__} to dict")


ListStr = Annotated[list[str] | None, BeforeValidator(_coerce_list)]
DictAny = Annotated[dict[str, Any] | None, BeforeValidator(_coerce_dict)]


MAX_CONTENT_LEN = 100_000
MAX_LIMIT = 1_000


def log_timing(tool_name: str):
    """Decorator to log tool execution time."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("[%s] completed in %.1fms", tool_name, elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error("[%s] failed after %.1fms: %s", tool_name, elapsed_ms, e)
                raise

        return wrapper

    return decorator


MIN_CONTENT_LEN = 10

# Patterns that indicate noise rather than useful memory content
_NOISE_PATTERNS = [
    re.compile(r"^diff --git ", re.MULTILINE),  # git diff output
    re.compile(r"^(\+\+\+|---) [ab]/", re.MULTILINE),  # diff headers
    re.compile(r"^@@\s", re.MULTILINE),  # diff hunks
    re.compile(r"^\$\s", re.MULTILINE),  # shell prompts
    re.compile(r"^[\s\S]*\x00", re.MULTILINE),  # binary content
]


def _require_content(content: str) -> None:
    if not content or not content.strip():
        raise ValueError("content must be a non-empty string")
    if len(content) > MAX_CONTENT_LEN:
        raise ValueError(f"content exceeds maximum length of {MAX_CONTENT_LEN} characters")
    stripped = content.strip()
    if len(stripped) < MIN_CONTENT_LEN:
        raise ValueError(
            f"content too short ({len(stripped)} chars). "
            f"Minimum {MIN_CONTENT_LEN} characters for a useful memory."
        )
    # Check for noise patterns (diff output, shell dumps, binary)
    noise_matches = sum(1 for p in _NOISE_PATTERNS if p.search(stripped))
    if noise_matches >= 2:
        raise ValueError(
            "content looks like a diff, shell dump, or binary output. "
            "Store a summary of what happened instead of raw output."
        )


def _require_limit(limit: int) -> None:
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")


# Session-level active profile. Defaults lazily from config on first use so
# helper imports do not require a fully configured runtime environment.
_active_profile: str | None = None


def _read_active_profile_sentinel() -> str | None:
    """Read ~/.ogham/active_profile if present.

    The Go CLI ogham-cli writes this file when the user runs `ogham profile
    switch` or calls the MCP switch_profile tool. By reading it here, both
    stacks agree on the active profile across process boundaries without
    having to mutate the user's config.toml. Returns None when the file is
    absent, unreadable, or empty so callers can fall through to the legacy
    in-memory / settings path.
    """
    import os

    path = os.path.expanduser("~/.ogham/active_profile")
    try:
        with open(path, encoding="utf-8") as f:
            name = f.read().strip()
    except OSError:
        return None
    return name or None


def get_active_profile() -> str:
    # Precedence: OGHAM_PROFILE env > Go-written sentinel file > in-memory
    # switch_profile call this session > settings.default_profile. Matches
    # the Go CLI's ActiveProfile() resolution exactly.
    import os

    if env := os.environ.get("OGHAM_PROFILE", "").strip():
        return env
    if sentinel := _read_active_profile_sentinel():
        return sentinel
    return _active_profile or settings.default_profile


@mcp.tool
def switch_profile(profile: str) -> dict[str, Any]:
    """Switch to a different memory profile. Like Severance -- each profile is
    a separate memory partition. Memories from other profiles are invisible.

    Args:
        profile: The profile to switch to (e.g. "work", "personal", "default").
    """
    global _active_profile
    old = get_active_profile()
    _active_profile = profile
    return {"status": "switched", "from": old, "to": profile}


@mcp.tool
def current_profile() -> dict[str, str]:
    """Show which memory profile is currently active."""
    return {"profile": get_active_profile()}


@mcp.tool
def list_profiles() -> list[dict[str, Any]]:
    """List all memory profiles and how many memories each has."""
    active_profile = get_active_profile()
    profiles = db_list_profiles()
    for p in profiles:
        if p["profile"] == active_profile:
            p["active"] = True
    return profiles


@mcp.tool
@log_timing("store_memory")
def store_memory(
    content: str,
    source: str | None = None,
    tags: ListStr = None,
    metadata: DictAny = None,
    auto_link: bool = True,
) -> dict[str, Any]:
    """Store a memory in the active profile with automatic embedding generation.

    Args:
        content: The text content to remember.
        source: Where this came from (e.g. "claude-desktop", "cursor", "claude-code").
        tags: Categorization tags for filtering (e.g. ["project:foo", "type:decision"]).
        metadata: Additional structured data to store alongside the memory.
        auto_link: Automatically link to similar existing memories (default True).
    """
    from ogham.recompute_executor import enqueue_for_tags
    from ogham.service import store_memory_enriched

    active_profile = get_active_profile()
    result = store_memory_enriched(
        content=content,
        profile=active_profile,
        source=source,
        tags=tags,
        metadata=metadata,
        auto_link=auto_link,
    )
    # Debounced topic-summary recompute per tag (T1.4 Phase 6 hook).
    enqueue_for_tags(active_profile, tags)
    return result


@mcp.tool
@log_timing("store_decision")
def store_decision(
    decision: str,
    rationale: str,
    alternatives: ListStr = None,
    reasoning_trace: str | None = None,
    tags: ListStr = None,
    related_memories: ListStr = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store an architectural decision with rationale. Creates a memory with
    type:decision tag and structured metadata, auto-linked to similar context.

    Args:
        decision: What was decided.
        rationale: Why — the reasoning behind the decision.
        alternatives: What was considered and rejected.
        reasoning_trace: The full chain of thought that led to this decision.
        tags: Additional tags (type:decision is always added).
        related_memories: UUIDs of memories this decision relates to (creates supports edges).
        source: Where this decision was made.
    """
    _require_content(decision)
    _require_content(rationale)

    parts = [f"Decision: {decision}", f"Rationale: {rationale}"]
    if alternatives:
        parts.append(f"Alternatives considered: {', '.join(alternatives)}")
    if reasoning_trace:
        parts.append(f"Reasoning: {reasoning_trace}")
    content = "\n".join(parts)

    decision_tags = list(tags or [])
    if "type:decision" not in decision_tags:
        decision_tags.append("type:decision")

    # Extract dates from decision content
    dates = extract_dates(content)

    metadata = {
        "type": "decision",
        "alternatives": alternatives or [],
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    if dates:
        metadata["dates"] = dates
    if reasoning_trace:
        metadata["reasoning_trace"] = reasoning_trace

    result = store_memory(
        content=content,
        source=source,
        tags=decision_tags,
        metadata=metadata,
    )

    if related_memories:
        for rel_id in related_memories:
            db_create_relationship(
                source_id=result["id"],
                target_id=rel_id,
                relationship="supports",
                strength=1.0,
                created_by="user",
                metadata={},
            )

    return result


@mcp.tool
@log_timing("store_preference")
def store_preference(
    preference: str,
    subject: str | None = None,
    alternatives: ListStr = None,
    strength: str = "normal",
    tags: ListStr = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store a user preference with structured metadata.

    Formats the preference as prose memory content with `type:preference` tag
    so later queries like "what does the user prefer for X?" surface it cleanly.
    Strength lets callers distinguish "always" from "usually from "sometimes".

    Args:
        preference: What is preferred (e.g. "dark mode", "PostgreSQL over MySQL").
        subject: Optional subject/context the preference applies to.
        alternatives: Optional list of alternatives that were rejected.
        strength: Preference strength -- "strong", "normal" (default), or "weak".
        tags: Additional tags; type:preference is always added.
        source: Where this preference was stated.
    """
    _require_content(preference)

    if strength not in ("strong", "normal", "weak"):
        raise ValueError("strength must be one of: strong, normal, weak")

    parts = [f"Preference: {preference}"]
    if subject:
        parts.append(f"Subject: {subject}")
    if alternatives:
        parts.append(f"Rejected alternatives: {', '.join(alternatives)}")
    parts.append(f"Strength: {strength}")
    content = "\n".join(parts)

    pref_tags = list(tags or [])
    if "type:preference" not in pref_tags:
        pref_tags.append("type:preference")

    metadata = {
        "type": "preference",
        "subject": subject,
        "alternatives": alternatives or [],
        "strength": strength,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    return store_memory(
        content=content,
        source=source,
        tags=pref_tags,
        metadata=metadata,
    )


@mcp.tool
@log_timing("store_fact")
def store_fact(
    fact: str,
    subject: str | None = None,
    confidence: float = 1.0,
    source_citation: str | None = None,
    tags: ListStr = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store a factual statement with confidence and optional citation.

    Formats the fact as a memory with `type:fact` tag. Confidence lets callers
    downweight uncertain facts at retrieval time; source_citation preserves
    provenance (paper, URL, conversation reference) in metadata.

    Args:
        fact: The factual statement.
        subject: Optional subject/entity the fact is about.
        confidence: Confidence score 0.0-1.0 (default 1.0).
        source_citation: Optional citation string (paper, URL, who-said-it).
        tags: Additional tags; type:fact is always added.
        source: Where this fact was recorded (the MCP client / tool).
    """
    _require_content(fact)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    parts = [f"Fact: {fact}"]
    if subject:
        parts.append(f"Subject: {subject}")
    if source_citation:
        parts.append(f"Source: {source_citation}")
    content = "\n".join(parts)

    fact_tags = list(tags or [])
    if "type:fact" not in fact_tags:
        fact_tags.append("type:fact")

    metadata = {
        "type": "fact",
        "subject": subject,
        "confidence": confidence,
        "source_citation": source_citation,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    return store_memory(
        content=content,
        source=source,
        tags=fact_tags,
        metadata=metadata,
    )


@mcp.tool
@log_timing("store_event")
def store_event(
    event: str,
    when: str | None = None,
    participants: ListStr = None,
    location: str | None = None,
    tags: ListStr = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store an event with structured temporal + participant metadata.

    Formats the event as memory content with `type:event` tag, plus structured
    metadata so temporal queries can match by `when`, social queries by
    `participants`, and location queries by `location`.

    Args:
        event: What happened.
        when: Optional time expression ("2026-04-15 14:00", "yesterday at 3pm",
              "last Tuesday"). Stored verbatim; the extraction layer will
              parse and tag dates.
        participants: Optional list of people or entities involved.
        location: Optional place name.
        tags: Additional tags; type:event is always added.
        source: Where this event was recorded.
    """
    _require_content(event)

    parts = [f"Event: {event}"]
    if when:
        parts.append(f"When: {when}")
    if participants:
        parts.append(f"Participants: {', '.join(participants)}")
    if location:
        parts.append(f"Location: {location}")
    content = "\n".join(parts)

    event_tags = list(tags or [])
    if "type:event" not in event_tags:
        event_tags.append("type:event")

    metadata = {
        "type": "event",
        "when": when,
        "participants": participants or [],
        "location": location,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    dates = extract_dates(content)
    if dates:
        metadata["dates"] = dates

    return store_memory(
        content=content,
        source=source,
        tags=event_tags,
        metadata=metadata,
    )


@mcp.tool
@log_timing("hybrid_search")
def hybrid_search(
    query: str,
    limit: int = 10,
    tags: ListStr = None,
    source: str | None = None,
    graph_depth: int = 0,
    profiles: ListStr = None,
    extract_facts: bool = False,
) -> dict[str, Any]:
    """Search memories in the active profile by meaning and keywords (hybrid search).

    Combines semantic similarity (embeddings) with keyword matching (full-text search)
    using Reciprocal Rank Fusion. Finds both conceptually similar memories and exact
    keyword matches.

    Returns ``{"results": [...memory rows...], "wiki_preamble": [...wiki summaries...]}``.
    The split (new in v0.12.1) keeps benchmark scorers and downstream pipelines clean:
    they see a deterministic list of retrieval hits in ``results``, and a separate
    optional list of compiled topic summaries in ``wiki_preamble``. ``wiki_preamble``
    is always present; it's an empty list when wiki injection is off, the embedding
    can't be generated, or no summary clears the similarity threshold.

    Args:
        query: Natural language search query. Also used for keyword matching.
        limit: Maximum number of results to return.
        tags: Filter results to memories with any of these tags.
        source: Filter results to memories from this source.
        graph_depth: Follow relationship edges N hops deep (0 = no graph, default).
        profiles: Search across multiple profiles (e.g. ["personal", "shared"]).
                  When set, overrides the active profile for this search.
        extract_facts: When True, runs an LLM over retrieved memories to extract
                  query-relevant facts. Returns focused facts instead of raw
                  memories. Requires OGHAM_EXTRACT_PROVIDER and API key.
                  Default: False (returns verbatim memories). Bypasses wiki
                  injection entirely so the extractor sees raw memories only.
    """
    _require_limit(limit)
    from ogham.embeddings import generate_embedding
    from ogham.service import _wiki_injection_results, search_memories_enriched

    profile = get_active_profile()
    results = search_memories_enriched(
        query=query,
        profile=profile,
        limit=limit,
        tags=tags,
        source=source,
        graph_depth=graph_depth,
        profiles=profiles,
        extract_facts=extract_facts,
    )

    wiki_preamble: list[dict[str, Any]] = []
    if not extract_facts and getattr(settings, "wiki_injection_enabled", False):
        embedding = generate_embedding(query)
        wiki_preamble = _wiki_injection_results(profile, embedding)

    return {"results": results, "wiki_preamble": wiki_preamble}


@mcp.tool
def list_recent(
    limit: int = 10,
    source: str | None = None,
    tags: ListStr = None,
) -> list[dict[str, Any]]:
    """List recent memories in the active profile.

    Args:
        limit: Maximum number of memories to return.
        source: Filter to memories from this source.
        tags: Filter to memories with any of these tags.
    """
    _require_limit(limit)
    return list_recent_memories(
        profile=get_active_profile(),
        limit=limit,
        source=source,
        tags=tags,
    )


@mcp.tool
def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a memory by its ID.

    Args:
        memory_id: The UUID of the memory to delete.
    """
    from ogham.database import emit_audit_event, get_memory_by_id
    from ogham.recompute_executor import enqueue_for_tags

    active_profile = get_active_profile()
    # Fetch tags BEFORE the delete -- after the row is gone we can't
    # recover which summaries cited it. Summaries that referenced this
    # memory need recompute so the next compile drops the source.
    # Use the backend facade (get_memory_by_id) rather than raw _execute
    # so this works on SupabaseBackend (PostgREST) too, not just psycopg.
    pre_row = get_memory_by_id(memory_id, active_profile)
    pre_tags = list(pre_row["tags"]) if pre_row and pre_row.get("tags") else []

    success = db_delete(memory_id, profile=active_profile)
    emit_audit_event(
        profile=active_profile,
        operation="delete",
        resource_id=memory_id,
        outcome="success" if success else "not_found",
    )
    if success:
        enqueue_for_tags(active_profile, pre_tags)
        return {"status": "deleted", "id": memory_id}
    return {"status": "not_found", "id": memory_id}


@mcp.tool
def update_memory(
    memory_id: str,
    content: str | None = None,
    tags: ListStr = None,
    metadata: DictAny = None,
) -> dict[str, Any]:
    """Update an existing memory. Re-embeds if content changes.

    Args:
        memory_id: The UUID of the memory to update.
        content: New content (triggers re-embedding).
        tags: New tags (replaces existing tags).
        metadata: New metadata (replaces existing metadata).
    """
    updates: dict[str, Any] = {}
    if content is not None:
        updates["content"] = content
        updates["embedding"] = str(generate_embedding(content))
    if tags is not None:
        updates["tags"] = tags
    if metadata is not None:
        updates["metadata"] = metadata

    if not updates:
        return {"status": "no_changes", "id": memory_id}

    from ogham.database import emit_audit_event, get_memory_by_id
    from ogham.recompute_executor import enqueue_for_tags

    active_profile = get_active_profile()
    # Fetch old tags FIRST so a tag-replace update (e.g. [a,b] -> [b,c])
    # enqueues for the dropped tag `a` too. The `a` summary needs to
    # recompile to drop this memory as a source. Use the backend facade
    # so this works on both PostgresBackend (psycopg) and SupabaseBackend
    # (PostgREST) -- raw _execute only exists on PostgresBackend.
    pre_row = get_memory_by_id(memory_id, active_profile)
    pre_tags = list(pre_row["tags"]) if pre_row and pre_row.get("tags") else []

    result = db_update(memory_id, updates, profile=active_profile)
    emit_audit_event(
        profile=active_profile,
        operation="update",
        resource_id=memory_id,
        metadata={"fields_updated": list(updates.keys())},
    )

    post_tags = tags if tags is not None else pre_tags
    affected = set(pre_tags) | set(post_tags or [])
    enqueue_for_tags(active_profile, list(affected))
    return {"status": "updated", "id": result["id"], "updated_at": result["updated_at"]}


@mcp.tool
@log_timing("reinforce_memory")
def reinforce_memory(
    memory_id: str,
    strength: float = 0.85,
) -> dict[str, Any]:
    """Reinforce a memory's confidence -- mark it as verified or confirmed.

    Increases the memory's confidence score, making it rank higher in future searches.
    Call this when a memory has been validated as accurate.

    Args:
        memory_id: The UUID of the memory to reinforce.
        strength: How strongly to reinforce (0.5-1.0, default 0.85). Higher = stronger boost.
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be between 0.0 (exclusive) and 1.0, got {strength}")
    from ogham.database import get_memory_by_id
    from ogham.recompute_executor import enqueue_for_tags

    active_profile = get_active_profile()
    new_confidence = db_update_confidence(memory_id, strength, active_profile)
    # Confidence shifts feed importance ranking inside future summaries.
    # Enqueue; the hash-match short-circuit will skip the LLM call when
    # the source set is unchanged (common case). Backend facade so this
    # works on Supabase (PostgREST) as well as psycopg.
    row = get_memory_by_id(memory_id, active_profile)
    tags = list(row["tags"]) if row and row.get("tags") else []
    enqueue_for_tags(active_profile, tags)
    return {
        "status": "reinforced",
        "id": memory_id,
        "confidence": new_confidence,
        "profile": active_profile,
    }


@mcp.tool
@log_timing("contradict_memory")
def contradict_memory(
    memory_id: str,
    strength: float = 0.15,
) -> dict[str, Any]:
    """Contradict a memory's confidence -- mark it as disputed or outdated.

    Decreases the memory's confidence score, making it rank lower in future searches.
    The memory isn't deleted, just deprioritised. Call this when a memory is found to be
    inaccurate or superseded.

    Args:
        memory_id: The UUID of the memory to contradict.
        strength: How strongly to contradict (0.0-0.5, default 0.15). Lower = stronger.
    """
    if not 0.0 <= strength < 1.0:
        raise ValueError(f"strength must be between 0.0 and 1.0 (exclusive), got {strength}")
    from ogham.database import get_memory_by_id
    from ogham.recompute_executor import enqueue_for_tags

    active_profile = get_active_profile()
    new_confidence = db_update_confidence(memory_id, strength, active_profile)
    # Same rationale as reinforce_memory: contradiction flips confidence,
    # cited summaries may rank differently. Cheap via hash-match. Backend
    # facade so this works on Supabase (PostgREST) as well as psycopg.
    row = get_memory_by_id(memory_id, active_profile)
    tags = list(row["tags"]) if row and row.get("tags") else []
    enqueue_for_tags(active_profile, tags)
    return {
        "status": "contradicted",
        "id": memory_id,
        "confidence": new_confidence,
        "profile": active_profile,
    }


@mcp.tool
async def re_embed_all(ctx: Context) -> dict[str, Any]:
    """Re-generate embeddings for all memories in the active profile.

    Run this after switching embedding providers or models (e.g. Ollama to OpenAI,
    or mxbai-embed-large to bge-m3). Vectors from different models are incompatible,
    so all existing embeddings need to be regenerated to keep search working.

    Reports progress via MCP progress notifications.
    """
    active_profile = get_active_profile()
    memories = get_all_memories_content(profile=active_profile)
    total = len(memories)
    if total == 0:
        return {"status": "nothing_to_do", "profile": active_profile, "total": 0}

    clear_embedding_cache()
    await ctx.info(f"Re-embedding {total} memories...")

    texts = [mem["content"] for mem in memories]
    ids = [mem["id"] for mem in memories]

    embeddings = generate_embeddings_batch(texts)
    await ctx.report_progress(total, total)

    # Write embeddings to DB in batches
    failed = 0
    batch_ids: list[str] = []
    batch_embs: list[list[float]] = []
    db_batch_size = settings.embedding_batch_size

    for mem_id, emb in zip(ids, embeddings):
        batch_ids.append(mem_id)
        batch_embs.append(emb)

        if len(batch_ids) >= db_batch_size:
            batch_update_embeddings(batch_ids, batch_embs)
            batch_ids = []
            batch_embs = []

    if batch_ids:
        batch_update_embeddings(batch_ids, batch_embs)

    provider = settings.embedding_provider
    model = settings.ollama_embed_model if provider == "ollama" else "text-embedding-3-small"

    return {
        "status": "complete",
        "profile": active_profile,
        "total": total,
        "succeeded": total - failed,
        "failed": failed,
        "provider": provider,
        "model": model,
    }


@mcp.tool
def health_check() -> dict[str, dict]:
    """Check connectivity to Supabase and the embedding provider.

    Returns status of each dependency and the active config.
    Use this to diagnose connection issues.
    """
    return full_health_check()


@mcp.tool
def set_profile_ttl(profile: str, ttl_days: int | None = None) -> dict[str, Any]:
    """Set a time-to-live (TTL) for memories in a profile. New memories stored in this
    profile will automatically expire after ttl_days. Existing memories are unaffected.

    Args:
        profile: The profile to configure (e.g. "work", "personal").
        ttl_days: Number of days before memories expire. Pass None to remove TTL.
    """
    if ttl_days is not None and ttl_days < 1:
        raise ValueError("ttl_days must be at least 1, or None to disable")
    result = db_set_profile_ttl(profile, ttl_days)
    return {
        "status": "configured",
        "profile": profile,
        "ttl_days": result.get("ttl_days"),
    }


@mcp.tool
@log_timing("export_profile")
def export_profile(format: str = "json") -> dict[str, Any]:
    """Export all memories in the active profile.

    Args:
        format: Output format — "json" or "markdown".
    """
    if format not in ("json", "markdown"):
        raise ValueError("format must be 'json' or 'markdown'")
    active_profile = get_active_profile()
    data = _export_memories(active_profile, format=format)
    return {"status": "exported", "profile": active_profile, "format": format, "data": data}


@mcp.tool
@log_timing("import_memories")
def import_memories_tool(data: str, dedup_threshold: float = 0.8) -> dict[str, Any]:
    """Import memories into the active profile from a JSON export.

    Args:
        data: JSON string from a previous export_profile call.
        dedup_threshold: Skip memories with similarity above this (0 to disable dedup).
    """
    return _import_memories(data, profile=get_active_profile(), dedup_threshold=dedup_threshold)


@mcp.tool
@log_timing("cleanup_expired")
def cleanup_expired() -> dict[str, Any]:
    """Delete expired memories in the active profile. Expired memories are already
    hidden from searches and listings — this permanently removes them.
    """
    active_profile = get_active_profile()
    count = db_count_expired(active_profile)
    if count == 0:
        return {"status": "nothing_to_clean", "profile": active_profile, "deleted": 0}

    deleted = db_cleanup_expired(active_profile)
    return {"status": "cleaned", "profile": active_profile, "deleted": deleted}


@mcp.tool
@log_timing("link_unlinked")
def link_unlinked(
    batch_size: int = 100,
    threshold: float = 0.85,
    max_links: int = 5,
) -> dict[str, Any]:
    """Backfill auto-links for memories that don't have any yet.

    Run this after batch imports or to populate the relationship graph
    for existing memories. Call repeatedly until processed returns 0.

    Args:
        batch_size: Number of memories to process per call (default 100).
        threshold: Minimum similarity to create a link (default 0.85).
        max_links: Maximum links per memory (default 5).
    """
    processed = db_link_unlinked(
        profile=get_active_profile(),
        threshold=threshold,
        max_links=max_links,
        batch_size=batch_size,
    )
    return {
        "status": "linked" if processed > 0 else "nothing_to_link",
        "profile": get_active_profile(),
        "processed": processed,
        "batch_size": batch_size,
    }


@mcp.tool
@log_timing("explore_knowledge")
def explore_knowledge(
    query: str,
    depth: int = 1,
    min_strength: float = 0.5,
    limit: int = 5,
    tags: ListStr = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Explore what you know about a topic. Finds relevant memories then
    expands via relationship graph to pull in connected context.

    Returns direct matches (depth=0) and related memories (depth=1+)
    with relationship type and strength.

    Args:
        query: Natural language query — "what do I know about X?"
        depth: How many relationship hops to traverse (0=search only, 1=default).
        min_strength: Minimum edge strength to traverse (default 0.5).
        limit: Number of seed results from hybrid search.
        tags: Filter seed results to memories with any of these tags.
        source: Filter seed results to memories from this source.
    """
    _require_limit(limit)
    embedding = generate_embedding(query)
    results = db_explore_graph(
        query_text=query,
        query_embedding=embedding,
        profile=get_active_profile(),
        limit=limit,
        depth=depth,
        min_strength=min_strength,
        tags=tags,
        source=source,
    )
    if results:
        record_access([r["id"] for r in results if r["depth"] == 0])
    return results


@mcp.tool
@log_timing("find_related")
def find_related(
    memory_id: str,
    relationship_types: ListStr = None,
    depth: int = 1,
    min_strength: float = 0.5,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find everything related to a specific memory by traversing relationship edges.

    Use for impact analysis — "what else connects to this memory?" Optionally
    filter by relationship type (e.g. only decisions that support this memory).

    Args:
        memory_id: The UUID of the starting memory.
        relationship_types: Filter to specific edge types (e.g. ["supports", "contradicts"]).
        depth: How many hops to traverse (default 1).
        min_strength: Minimum edge strength to follow (default 0.5).
        limit: Maximum results to return (default 20).
    """
    return db_get_related(
        memory_id=memory_id,
        depth=depth,
        min_strength=min_strength,
        relationship_types=relationship_types,
        limit=limit,
    )


@mcp.tool
@log_timing("suggest_connections")
def suggest_connections(
    memory_id: str,
    min_shared_entities: int = 2,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Suggest memories that share entities but have no explicit relationship.

    Surfaces "hidden" connections through the entity graph -- memories that
    mention the same people, projects, or concepts but were never explicitly
    linked. Useful for discovering cross-session context an agent might miss.

    Args:
        memory_id: The UUID of the starting memory.
        min_shared_entities: Minimum entities in common (default 2).
        limit: Maximum suggestions (default 10).
    """
    from ogham.database import get_backend

    backend = get_backend()
    try:
        rows = backend._execute(
            """
            WITH target_entities AS (
                SELECT entity_id FROM memory_entities
                WHERE memory_id = %(memory_id)s::uuid
            ),
            shared AS (
                SELECT
                    me.memory_id,
                    count(*) as shared_count,
                    array_agg(e.entity_type || ':' || e.canonical_name) as shared_entities
                FROM memory_entities me
                JOIN target_entities te ON te.entity_id = me.entity_id
                JOIN entities e ON e.id = me.entity_id
                WHERE me.memory_id != %(memory_id)s::uuid
                  AND me.profile = %(profile)s
                GROUP BY me.memory_id
                HAVING count(*) >= %(min_shared)s
            ),
            unlinked AS (
                SELECT s.*
                FROM shared s
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_relationships mr
                    WHERE (mr.source_id = %(memory_id)s::uuid AND mr.target_id = s.memory_id)
                       OR (mr.target_id = %(memory_id)s::uuid AND mr.source_id = s.memory_id)
                )
            )
            SELECT
                u.memory_id::text as id,
                u.shared_count,
                u.shared_entities,
                m.content,
                m.created_at,
                m.tags
            FROM unlinked u
            JOIN memories m ON m.id = u.memory_id
            WHERE m.expires_at IS NULL OR m.expires_at > now()
            ORDER BY u.shared_count DESC, m.created_at DESC
            LIMIT %(limit)s
            """,
            {
                "memory_id": memory_id,
                "profile": get_active_profile(),
                "min_shared": min_shared_entities,
                "limit": limit,
            },
            fetch="all",
        )
    except Exception as e:
        logger.debug("suggest_connections failed: %s", e)
        return []

    return rows or []


@mcp.tool
@log_timing("compress_old_memories")
def compress_old_memories() -> dict[str, Any]:
    """Compress old, inactive memories to save space and reduce search noise.

    Memories compress gradually:
    - Level 0 (recent): full text preserved
    - Level 1 (7+ days, low activity): compressed to key sentences (~30%)
    - Level 2 (30+ days, low activity): compressed to one-line summary + tags

    High-importance, frequently-accessed, or high-confidence memories
    resist compression. Original content is always preserved for restoration.
    """
    from ogham.compression import compress_to_gist, compress_to_tags, get_compression_target
    from ogham.database import get_all_memories_full

    active_profile = get_active_profile()
    memories = get_all_memories_full(profile=active_profile)
    stats = {"compressed_to_gist": 0, "compressed_to_tags": 0, "skipped": 0, "total": len(memories)}

    for mem in memories:
        current = mem.get("compression_level", 0)
        target = get_compression_target(mem)

        if target <= current:
            stats["skipped"] += 1
            continue

        content = mem["content"]
        tags = mem.get("tags", [])

        if target == 1 and current == 0:
            gist = compress_to_gist(content)
            db_update(
                mem["id"],
                content=gist,
                profile=active_profile,
            )
            # Store original and update compression level via direct update
            _update_compression(mem["id"], compression_level=1, original_content=content)
            stats["compressed_to_gist"] += 1

        elif target == 2:
            if current == 0:
                # Save original before any compression
                _update_compression(mem["id"], original_content=content)
            tag_repr = compress_to_tags(content, tags)
            db_update(
                mem["id"],
                content=tag_repr,
                profile=active_profile,
            )
            _update_compression(mem["id"], compression_level=2)
            stats["compressed_to_tags"] += 1

    return stats


def _update_compression(
    memory_id: str,
    compression_level: int | None = None,
    original_content: str | None = None,
) -> None:
    """Update compression columns directly via backend."""
    from ogham.database import get_backend

    backend = get_backend()
    updates = {}
    if compression_level is not None:
        updates["compression_level"] = compression_level
    if original_content is not None:
        updates["original_content"] = original_content

    if hasattr(backend, "_get_client"):
        # Supabase
        client = backend._get_client()
        client.table("memories").update(updates).eq("id", memory_id).execute()
    else:
        # Postgres
        set_clauses = [f"{k} = %({k})s" for k in updates]
        updates["id"] = memory_id
        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = %(id)s"
        backend._execute(sql, updates)


@mcp.tool
@log_timing("advance_lifecycle")
def advance_lifecycle(profile: str | None = None) -> dict[str, Any]:
    """Run the lifecycle advancement sweep for a profile.

    Ordinarily the sweep fires at session start. Call this to force
    it (useful after bulk imports or dashboard testing).

    Args:
        profile: Target profile. Defaults to the active profile.

    Returns:
        dict with counts: fresh_to_stable, editing_closed.
    """
    p = profile or get_active_profile()
    report = advance_stages(p)
    return {
        "profile": p,
        "fresh_to_stable": report.fresh_to_stable,
        "editing_closed": report.editing_closed,
    }
