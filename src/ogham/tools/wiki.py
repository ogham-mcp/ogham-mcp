"""MCP tools for the wiki layer (T1.1, v0.12).

`compile_wiki` is the user-facing read/write tool: ask for a topic, get
a synthesized markdown page back. It reuses the recompute pipeline so
the cache the executor populates and the cache an explicit user request
populates are the same rows -- no shadow paths, no drift.

`query_topic_summary` is the cheap read-only fetch: returns whatever the
cache already holds for a topic, no LLM call, no recompute. Used by the
dashboard Wiki tab and by any caller that only wants what's already
synthesized. v0.13 added a `level` parameter so callers can request the
short paragraph or one-line form instead of the full body.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ogham.app import mcp
from ogham.config import settings
from ogham.data.loader import get_wiki_message
from ogham.database import walk_memory_graph
from ogham.recompute import recompute_topic_summary
from ogham.tools.memory import get_active_profile
from ogham.topic_summaries import get_summary_by_topic
from ogham.wiki_lint import lint_report

# v0.13: progressive-recall resolution levels. The default `body` preserves
# v0.12 behaviour; `short` and `one_line` target tighter context budgets.
LevelType = Literal["body", "short", "one_line"]
_LEVEL_TO_COLUMN: dict[str, str] = {
    "body": "content",
    "short": "tldr_short",
    "one_line": "tldr_one_line",
}


def _wiki_message(key: str, **fields: Any) -> str:
    """Format a wiki user-facing message in the active locale."""
    template = get_wiki_message(key, lang=getattr(settings, "locale", "en"))
    return template.format(**fields) if fields else template


logger = logging.getLogger(__name__)


def _format_summary_response(
    summary: dict[str, Any],
    *,
    level: LevelType = "body",
) -> dict[str, Any]:
    """Shape a topic_summaries row for MCP return.

    Stamps the markdown content with a YAML frontmatter block so callers
    can see provenance (source ids, hash, model, version) without a
    second round-trip. Frontmatter is at the top of the markdown body
    rather than in a sibling field because most readers (Obsidian,
    static-site generators, plain `cat`) treat it as part of the page.

    v0.13: `level` selects which content column populates `markdown`.
    Falls back to body when the requested column is NULL on the row
    (pre-033 schemas, or rows that failed three-form generation).
    """
    source_hash = summary.get("source_hash")
    if isinstance(source_hash, (bytes, bytearray, memoryview)):
        source_hash_hex = bytes(source_hash).hex()
    elif isinstance(source_hash, str):
        # PostgREST returns bytea as `\xDEADBEEF` -- strip the prefix.
        source_hash_hex = source_hash.removeprefix("\\x") or None
    else:
        source_hash_hex = None

    column = _LEVEL_TO_COLUMN[level]
    body_text = summary.get(column)
    served_level: LevelType = level
    fallback_reason: str | None = None
    if not body_text and level != "body":
        # Pre-033 rows have NULL TLDR fields; fall back to body so callers
        # always get *something* readable rather than a blank page.
        body_text = summary.get("content") or ""
        served_level = "body"
        fallback_reason = f"{column} is null on this row"

    if body_text is None:
        body_text = ""

    frontmatter_lines = [
        "---",
        f"ogham_id: {summary['id']}",
        f"topic_key: {summary['topic_key']}",
        f"profile: {summary['profile_id']}",
        f"version: {summary['version']}",
        f"status: {summary['status']}",
        f"source_count: {summary['source_count']}",
        f"model_used: {summary['model_used']}",
        f"updated_at: {summary['updated_at']}",
        f"level: {served_level}",
    ]
    if source_hash_hex:
        frontmatter_lines.append(f"source_hash: {source_hash_hex}")
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    stamped = f"{frontmatter}\n\n{body_text}"

    response: dict[str, Any] = {
        "id": str(summary["id"]),
        "topic_key": summary["topic_key"],
        "profile": summary["profile_id"],
        "version": summary["version"],
        "status": summary["status"],
        "source_count": summary["source_count"],
        "model_used": summary["model_used"],
        "updated_at": str(summary["updated_at"]),
        "source_hash": source_hash_hex,
        "markdown": stamped,
        "content": body_text,
        "level": served_level,
    }
    if fallback_reason is not None:
        response["requested_level"] = level
        response["fallback_reason"] = fallback_reason
    return response


@mcp.tool
def compile_wiki(
    topic: str,
    provider: str | None = None,
    model: str | None = None,
    force: bool = False,
    force_oversize: bool = False,
) -> dict[str, Any]:
    """Compile a topic into a synthesized markdown wiki page.

    Resolves source memories tagged with `topic`, runs the same compile
    pipeline the background executor uses (hash-check short-circuit,
    LLM synthesize, embed, atomic upsert), and returns the resulting
    markdown with a YAML frontmatter provenance stamp.

    Args:
        topic: Tag string. Memories carrying this tag in the active
            profile are the synthesis sources.
        provider: Optional LLM provider override (e.g. "openai",
            "gemini"). Defaults to settings.llm_provider.
        model: Optional model override (e.g. "gpt-4o-mini",
            "gemini-2.5-flash"). Defaults to settings.llm_model.
        force: When True, bypass the source-hash short-circuit and
            re-synthesize even if sources haven't changed. Use to
            re-compile with a different provider/model on the same
            source set without manually marking the existing summary
            stale. Default False (cheapest behaviour).
        force_oversize: When True, override the
            settings.compile_max_sources cap. Default False -- topics
            with more sources than the cap are refused with
            `status="skipped_oversize"` to protect against mega-rollup
            tags producing JSON-malformed LLM outputs.

    Returns:
        A dict with the stamped markdown plus structured fields. If no
        memories carry this tag in the active profile, returns
        `{"status": "no_sources", ...}` without writing anything. If the
        tag's source set exceeds `settings.compile_max_sources` and
        force_oversize is False, returns `{"status": "skipped_oversize",
        "source_count": N, "max_sources": M, ...}` without an LLM call.
    """
    from ogham.flow_control import disabled_payload, inscribe_enabled

    profile = get_active_profile()
    if not inscribe_enabled():
        return disabled_payload("inscribe", topic_key=topic, profile=profile)

    # `force=True` is implemented by marking the existing summary stale
    # before recompute. The recompute pipeline's short-circuit checks
    # status='fresh', so flipping to 'stale' bypasses it cleanly without
    # touching recompute.py. The atomic upsert resets status='fresh' on
    # successful synthesis.
    if force:
        existing = get_summary_by_topic(profile, topic)
        if existing and existing.get("status") == "fresh":
            from ogham.topic_summaries import mark_stale

            mark_stale(str(existing["id"]), reason="forced re-compile")

    result = recompute_topic_summary(
        profile=profile,
        topic_key=topic,
        provider=provider,
        model=model,
        force_oversize=force_oversize,
    )

    if result.get("action") == "no_sources":
        return {
            "status": "no_sources",
            "topic_key": topic,
            "profile": profile,
            "message": _wiki_message("no_sources", topic=topic, profile=profile),
        }

    if result.get("action") == "skipped_oversize":
        return {
            "status": "skipped_oversize",
            "topic_key": topic,
            "profile": profile,
            "source_count": result.get("source_count"),
            "max_sources": result.get("max_sources"),
            "message": (
                f"Topic {topic!r} has {result.get('source_count')} source memories, "
                f"exceeding the compile_max_sources cap of {result.get('max_sources')}. "
                "Mega-rollup tags produce LLM outputs that fail JSON escaping; pass "
                "force_oversize=True to override, or use a more targeted tag."
            ),
        }

    summary = get_summary_by_topic(profile, topic)
    if summary is None:
        # Recompute claims success but the row vanished -- shouldn't
        # happen unless something deletes mid-flight. Surface clearly
        # rather than hand back an empty payload.
        logger.error(
            "compile_wiki: recompute action=%r but no row found for %s/%s",
            result.get("action"),
            profile,
            topic,
        )
        return {
            "status": "error",
            "topic_key": topic,
            "profile": profile,
            "message": _wiki_message("row_missing"),
        }

    response = _format_summary_response(summary)
    response["action"] = result.get("action")  # "skipped" or "recomputed"
    return response


_VALID_DIRECTIONS = ("outgoing", "incoming", "both")


@mcp.tool
def walk_knowledge(
    start_id: str,
    depth: int = 1,
    direction: str = "both",
    min_strength: float = 0.0,
    relationship_types: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Walk the memory-relationships graph from a known memory.

    Pure-code traversal -- no LLM, no embedding, no new schema. Lets an
    agent follow `memory_relationships` edges like wiki-links to assemble
    multi-hop context cheaply.

    Args:
        start_id: UUID of the memory to walk from.
        depth: Hops to follow (0-5). Default 1 = direct neighbours only.
        direction: 'outgoing' (this->other), 'incoming' (other->this),
            or 'both' (default; bidirectional, like get_related_memories).
        min_strength: Drop edges below this strength (0.0-1.0).
        relationship_types: Optional filter (e.g. ['similar_to', 'contradicts']).
        limit: Max nodes returned across all depths.

    Returns:
        `{"start_id": ..., "depth": ..., "direction": ..., "nodes": [...]}`
        where each node carries id, content, depth, relationship,
        edge_strength, connected_from (parent in the path), and the
        direction the edge was followed.
    """
    from ogham.flow_control import disabled_payload, recall_enabled

    if not recall_enabled():
        return disabled_payload(
            "recall",
            start_id=start_id,
            depth=depth,
            direction=direction,
            node_count=0,
            nodes=[],
        )

    if direction not in _VALID_DIRECTIONS:
        return {
            "status": "error",
            "message": _wiki_message(
                "invalid_direction",
                valid=list(_VALID_DIRECTIONS),
                got=direction,
            ),
        }

    try:
        rows = walk_memory_graph(
            start_id=start_id,
            depth=depth,
            direction=direction,
            min_strength=min_strength,
            relationship_types=relationship_types,
            limit=limit,
        )
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    nodes = []
    for r in rows:
        nodes.append(
            {
                "id": str(r["id"]),
                "content": r.get("content"),
                "tags": r.get("tags") or [],
                "source": r.get("source"),
                "confidence": r.get("confidence"),
                "depth": r.get("depth"),
                "relationship": r.get("relationship"),
                "edge_strength": r.get("edge_strength"),
                "connected_from": (str(r["connected_from"]) if r.get("connected_from") else None),
                "direction_used": r.get("direction_used"),
            }
        )

    return {
        "start_id": start_id,
        "depth": depth,
        "direction": direction,
        "node_count": len(nodes),
        "nodes": nodes,
    }


@mcp.tool
def query_topic_summary(topic: str, level: LevelType = "body") -> dict[str, Any]:
    """Fetch the cached topic summary without triggering recompute.

    Cheap read-only path -- returns whatever is in `topic_summaries`
    for (active_profile, topic) right now. Use `compile_wiki` if you
    want to ensure the summary reflects the latest sources.

    Args:
        topic: Tag string for the topic.
        level: Resolution level (v0.13 progressive recall):
            - 'body' (default, ~1000-2000 tokens): full markdown body.
              Preserves v0.12 behaviour for callers that don't pass
              the parameter.
            - 'short' (~150-300 tokens): one-paragraph TLDR. The cheap
              context-preamble form.
            - 'one_line' (~30-50 tokens): single-sentence glanceable form.
            If the requested level is NULL on the row (pre-033 rows or
            ones that failed three-form generation), the response falls
            back to body and reports `requested_level` + `fallback_reason`.

    Returns:
        Dict with the cached markdown + provenance, or
        `{"status": "not_cached", ...}` if no summary exists yet. The
        `level` field reports which form actually populated `content`.
    """
    if level not in _LEVEL_TO_COLUMN:
        raise ValueError(f"unknown level {level!r}; expected one of {sorted(_LEVEL_TO_COLUMN)}")

    from ogham.flow_control import disabled_payload, recall_enabled

    profile = get_active_profile()
    if not recall_enabled():
        return disabled_payload("recall", topic_key=topic, profile=profile)

    summary = get_summary_by_topic(profile, topic)
    if summary is None:
        return {
            "status": "not_cached",
            "topic_key": topic,
            "profile": profile,
            "message": _wiki_message("not_cached", topic=topic, profile=profile),
        }
    return _format_summary_response(summary, level=level)


@mcp.tool
def lint_wiki(
    stable_days: int = 90,
    sample_size: int = 10,
    include_drift: bool = True,
) -> dict[str, Any]:
    """Health report across the active profile's memory + wiki layer.

    Karpathy's "maintenance" verb. One structured response covering:
      * contradictions -- pairs joined by relationship='contradicts'
      * orphans -- memories with no edges (5-min-old grace window
        excludes still-auto-linking rows)
      * stale_lifecycle -- memories stuck in stage='stable' past
        `stable_days` (default 90)
      * stale_summaries -- topic_summaries in status='stale'
      * summary_drift -- fresh summaries whose stored source_hash no
        longer matches their current tagged-memory set (catches the
        case where a recompute hook missed an event)

    Args:
        stable_days: Lifecycle "stuck in stable" threshold in days.
        sample_size: Per-category preview rows. Counts are unbounded.
        include_drift: Set False on huge profiles where the per-topic
            re-hash loop would be slow.

    Returns:
        Aggregate report with `healthy: bool`, `issue_count: int`, and
        a per-category breakdown carrying count + sample.
    """
    profile = get_active_profile()
    return lint_report(
        profile=profile,
        stable_days=stable_days,
        sample_size=sample_size,
        include_drift=include_drift,
    )
