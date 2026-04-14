"""Shared memory service pipeline.

Used by both the MCP tool layer (tools/memory.py) and the gateway REST API.
Handles: content validation, date extraction, entity extraction,
importance scoring, embedding generation, surprise scoring,
storage, auto-linking, and optional read-time fact extraction.
"""

import hashlib
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from ogham.data.loader import get_direction_words
from ogham.database import auto_link_memory as db_auto_link
from ogham.database import (
    emit_audit_event,
    hybrid_search_memories,
    record_access,
    spread_entity_activation,
)
from ogham.database import get_profile_ttl as db_get_profile_ttl
from ogham.database import store_memory as db_store
from ogham.embeddings import EmbeddingUsage, generate_embedding
from ogham.extraction import (
    compute_importance,
    extract_dates,
    extract_entities,
    extract_query_anchors,
    extract_recurrence,
    has_temporal_intent,
    is_broad_summary_query,
    is_cross_reference_query,
    is_multi_hop_temporal,
    is_ordering_query,
    resolve_temporal_query,
)
from ogham.pricing import calculate_embedding_cost

logger = logging.getLogger(__name__)


def _merge_embedding_usage(
    total: EmbeddingUsage | None, current: EmbeddingUsage | None
) -> EmbeddingUsage | None:
    if current is None:
        return total
    if total is None:
        return dict(current)

    merged: EmbeddingUsage = dict(total)
    current_tokens = current.get("input_tokens")
    if current_tokens is not None:
        merged["input_tokens"] = merged.get("input_tokens", 0) + current_tokens
    if not merged.get("model"):
        merged["model"] = current.get("model", "")
    return merged


def _audit_usage_fields(usage: EmbeddingUsage | None) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "embedding_model": usage.get("model"),
        "tokens_used": usage.get("input_tokens"),
        "cost_usd": calculate_embedding_cost(usage),
    }


def store_memory_enriched(
    content: str,
    profile: str,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    auto_link: bool = True,
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Full store pipeline: validation, extraction, embedding, scoring, store, link.

    Returns the stored memory dict with id, created_at, links_created, etc.
    """
    # Lazy import to avoid circular dependency with tools/memory.py
    from ogham.tools.memory import _require_content

    _require_content(content)

    # Mask secrets before storing (protects all paths: hooks, MCP tools, gateway, CLI)
    from ogham.hooks import _mask_secrets

    content = _mask_secrets(content)

    # Auto-extract dates into metadata
    dates = extract_dates(content)
    if dates:
        if metadata is None:
            metadata = {}
        metadata["dates"] = dates

    # Auto-extract entities as tags
    entity_tags = extract_entities(content)
    if entity_tags:
        if tags is None:
            tags = []
        else:
            tags = list(tags)
        tags.extend(entity_tags)

    # Auto-extract recurrence (multilingual, 16 languages)
    recurrence_days = extract_recurrence(content)

    # Compute importance score from content signals
    importance = compute_importance(content, tags)
    # Preserve original importance for Hebbian decay recovery
    if metadata is None:
        metadata = {}
    metadata["original_importance"] = importance

    # Generate embedding (skip if pre-computed, e.g. from gateway cache)
    embedding_usage = None
    if embedding is None:
        embedding_usage = {}
        embedding = generate_embedding(content, usage_out=embedding_usage)

    # Compute surprise score + detect conflicts (>75% similarity)
    surprise = 0.5
    conflicts: list[dict[str, Any]] = []
    conflict_threshold = float(os.environ.get("OGHAM_CONFLICT_THRESHOLD", "0.75"))
    try:
        existing = hybrid_search_memories(
            query_text=content[:200],
            query_embedding=embedding,
            profile=profile,
            limit=3,
        )
        if existing:
            max_sim = max(r.get("similarity", 0) for r in existing)
            surprise = round(1.0 - max_sim, 3)
            for r in existing:
                sim = r.get("similarity", 0)
                if sim >= conflict_threshold:
                    conflicts.append(
                        {
                            "id": r.get("id", ""),
                            "similarity": round(sim, 3),
                            "content_preview": r.get("content", "")[:200],
                        }
                    )
    except Exception:
        logger.debug("Surprise scoring skipped: search failed, using default 0.5")

    # TTL
    ttl_days = db_get_profile_ttl(profile)
    expires_at = None
    if ttl_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    # Store
    result = db_store(
        content=content,
        embedding=embedding,
        profile=profile,
        metadata=metadata,
        source=source,
        tags=tags,
        expires_at=expires_at,
        importance=importance,
        recurrence_days=recurrence_days,
        surprise=surprise,
    )

    response: dict[str, Any] = {
        "status": "stored",
        "id": result["id"],
        "profile": profile,
        "created_at": result["created_at"],
        "expires_at": expires_at,
        "importance": importance,
        "surprise": surprise,
    }

    if conflicts:
        response["conflicts"] = conflicts
        response["conflict_warning"] = (
            f"Found {len(conflicts)} existing memory(s) with >{int(conflict_threshold * 100)}% "
            "similarity. Consider using update_memory instead of storing a duplicate."
        )

    # Auto-link
    if auto_link:
        links_created = db_auto_link(
            memory_id=result["id"],
            embedding=embedding,
            profile=profile,
        )
        response["links_created"] = links_created

    # Audit trail
    emit_audit_event(
        profile=profile,
        operation="store",
        resource_id=str(result["id"]),
        source=source,
        metadata={"importance": importance, "surprise": surprise},
        **_audit_usage_fields(embedding_usage),
    )

    return response


def _read_time_extract(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract query-relevant facts from retrieved memories using an LLM.

    Opt-in read-time extraction: the extractor sees both the query and the
    retrieved context, producing focused facts for the caller. The raw
    memories are still stored verbatim -- this is a presentation optimisation,
    not a storage transformation.

    Provider/model controlled via env vars:
        OGHAM_EXTRACT_PROVIDER: "ollama", "gemini" (default), or "openai"
        OGHAM_EXTRACT_MODEL: model name (defaults per provider)
    """
    if not results:
        return results

    parts = []
    for i, r in enumerate(results):
        content = r.get("content", "")
        parts.append(f"## Memory {i + 1}\n{content}")
    bundle = "\n\n".join(parts)

    prompt = (
        "Given a user's question and retrieved memory context, extract the facts "
        "most relevant to answering the question.\n\n"
        f"Question: {query}\n\n"
        f"Memory context:\n{bundle}\n\n"
        "Extract relevant facts as a concise bulleted list. Preserve specific "
        "details: names, numbers, dates, locations. If the context contains no "
        'relevant information, respond with "No relevant facts found."'
    )

    provider = os.environ.get("OGHAM_EXTRACT_PROVIDER", "gemini")
    model = os.environ.get("OGHAM_EXTRACT_MODEL", "")

    try:
        if provider == "ollama":
            import httpx

            model = model or "gemma3:1b"
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = httpx.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            facts = resp.json().get("message", {}).get("content", "")
        elif provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = model or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            facts = response.choices[0].message.content or ""
        else:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)
            model = model or "gemini-2.5-flash"
            response = client.models.generate_content(model=model, contents=prompt)
            facts = response.text or ""
    except Exception:
        logger.warning("Read-time extraction failed, returning raw results")
        return results

    return [
        {
            "id": "extracted-facts",
            "content": facts,
            "metadata": {
                "source_ids": [r.get("id", "") for r in results],
                "extraction_model": f"{provider}:{model}",
            },
            "tags": ["extracted"],
        }
    ]


def search_memories_enriched(
    query: str,
    profile: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
    graph_depth: int = 0,
    embedding: list[float] | None = None,
    profiles: list[str] | None = None,
    extract_facts: bool = False,
) -> list[dict[str, Any]]:
    """Full search pipeline: retrieve, rerank (optional), record access.

    Pipeline stages:
    1. _search_memories_raw: intent detection, retrieval, temporal/entity enrichment
    2. _maybe_rerank: optional FlashRank cross-encoder reranking (RERANK_ENABLED=true)
    3. Record access for retrieved memories
    4. (opt-in) _read_time_extract: LLM-powered fact extraction from results

    Args:
        extract_facts: When True, runs retrieved memories through an LLM to
            extract query-relevant facts. Returns a single extracted-facts
            result instead of raw memories. Default: False (verbatim results).
    """
    embedding_usage: EmbeddingUsage | None = None

    if embedding is None:
        def _generate_embedding_with_usage_tracking(text: str) -> list[float]:
            nonlocal embedding_usage
            current_usage: EmbeddingUsage = {}
            result = generate_embedding(text, usage_out=current_usage)
            embedding_usage = _merge_embedding_usage(embedding_usage, current_usage or None)
            return result

        embedding = _generate_embedding_with_usage_tracking(query)
        embedding_generator = _generate_embedding_with_usage_tracking
    else:
        embedding_generator = generate_embedding

    results = _search_memories_raw(
        query,
        profile,
        limit,
        tags,
        source,
        graph_depth,
        embedding,
        profiles,
        embedding_generator=embedding_generator,
    )

    if results:
        results = _maybe_rerank(query, results, limit)
        results = _reorder_for_attention(results)
        record_access([r["id"] for r in results])

    # Audit trail
    result_ids = [str(r["id"]) for r in results] if results else []
    emit_audit_event(
        profile=profile,
        operation="search",
        result_ids=result_ids or None,
        result_count=len(results),
        query_hash=hashlib.sha256(query.encode()).hexdigest()[:16],
        **_audit_usage_fields(embedding_usage),
    )

    if extract_facts and results:
        results = _read_time_extract(query, results)

    return results


def build_timeline_table(
    results: list[dict],
    reference_date: datetime | None = None,
) -> str:
    """Build a chronological timeline table from retrieved memories.

    Extracts dates from each memory, sorts chronologically, computes
    "days ago" relative to reference_date (defaults to now), and creates
    a Markdown table with memory ID hard links (M1, M3, etc.).

    Returns an empty string if fewer than 2 dated events are found.
    """
    ref_dt = reference_date or datetime.now(timezone.utc)
    ref_dt_naive = ref_dt.replace(tzinfo=None)

    # Collect and parse dates
    dated_events = []
    for idx, r in enumerate(results, 1):
        content = r.get("content", "")
        meta = r.get("metadata") or {}
        dates = meta.get("dates") or extract_dates(content)

        if dates:
            summary = content[:100].replace("\n", " ")
            for d in dates:
                dated_events.append({"date": d, "summary": summary, "idx": idx})

    if len(dated_events) < 2:
        return ""

    # Group by date (dict preserves insertion order in Python 3.7+)
    dated_events.sort(key=lambda x: x["date"])
    day_groups: dict[str, list[dict]] = {}
    for ev in dated_events:
        day_groups.setdefault(ev["date"], []).append(ev)

    # Build table
    ref_str = ref_dt.strftime("%Y-%m-%d")
    lines = [
        f"### CHRONOLOGICAL TIMELINE (Today = {ref_str}) ###",
        "| Date | Event Summary | Days Ago | Ref |",
        "|:---|:---|:---|:---|",
    ]

    for date_str, entries in list(day_groups.items())[:20]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
            delta = (ref_dt_naive - dt).days
            ago = f"{delta} days" if delta > 0 else "TODAY"
        except ValueError:
            ago = "---"

        summary = " + ".join(e["summary"][:35] for e in entries)
        refs = ", ".join(f"M{e['idx']}" for e in entries)
        lines.append(f"| {date_str} | {summary[:38]} | {ago} | {refs} |")

    lines.append(f"| {ref_str} | >>> TODAY (reference point) <<< | TODAY | |")

    return "\n".join(lines)


def format_results_with_sessions(
    results: list[dict],
    reference_date: datetime | None = None,
    include_timeline: bool = True,
) -> str:
    """Format search results with session headers, entity tags, and optional timeline.

    Produces a structured context string ready for LLM consumption:
    - Timeline table at the top (if enough dated events)
    - Session boundary headers between different conversation dates
    - Entity and date annotations per memory

    Args:
        results: Search results from search_memories_enriched.
        reference_date: Reference date for timeline "days ago" computation.
        include_timeline: Whether to include the timeline table.

    Returns:
        Formatted context string with session headers and annotations.
    """
    parts = []

    if include_timeline:
        timeline = build_timeline_table(results, reference_date=reference_date)
        if timeline:
            parts.append(timeline)
            parts.append("")

    current_session = None
    for idx, r in enumerate(results, 1):
        content = r.get("content", "")
        if not content:
            continue

        # Detect session date from metadata or content prefix
        meta = r.get("metadata") or {}
        session_date = meta.get("date")
        if not session_date:
            import re

            date_match = re.match(r"\[Date:\s*([^\]]+)\]", content)
            if date_match:
                session_date = date_match.group(1).strip()
            elif r.get("created_at"):
                session_date = str(r["created_at"])[:10]

        if session_date and session_date != current_session:
            parts.append(f"\n=== SESSION: {session_date} ===")
            current_session = session_date

        # Entity and date annotations
        entity_tags = extract_entities(content)
        dates = extract_dates(content)

        annotations = []
        if entity_tags:
            annotations.append(f"Entities: {', '.join(entity_tags)}")
        if dates:
            annotations.append(f"Dates: {', '.join(dates)}")

        if annotations:
            parts.append(f"[Memory {idx}] {content}\n[{' | '.join(annotations)}]")
        else:
            parts.append(f"[Memory {idx}] {content}")

    return "\n---\n".join(parts)


def _reorder_for_attention(results: list[dict]) -> list[dict]:
    """Reorder results to combat 'Lost in the Middle' (Liu et al. 2023).

    LLMs attend better to items at the start and end of the context.
    Put top 30% first, bottom 20% last, middle items in between.
    """
    if len(results) < 5:
        return results
    n = len(results)
    top = results[: max(1, n * 3 // 10)]
    bottom = results[max(1, n * 8 // 10) :]
    middle = results[max(1, n * 3 // 10) : max(1, n * 8 // 10)]
    return top + middle + bottom


def _maybe_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Apply cross-encoder reranking if enabled via RERANK_ENABLED."""
    from ogham.config import settings

    if not settings.rerank_enabled:
        return results
    from ogham.reranker import rerank_results

    return rerank_results(query, results, top_k=limit, alpha=settings.rerank_alpha)


def _search_memories_raw(
    query: str,
    profile: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
    graph_depth: int = 0,
    embedding: list[float] | None = None,
    profiles: list[str] | None = None,
    embedding_generator: Callable[[str], list[float]] = generate_embedding,
) -> list[dict[str, Any]]:
    """Retrieve memories via intent-aware search paths. No reranking, no access recording."""
    if embedding is None:
        embedding = embedding_generator(query)

    # Query reformulation disabled — global application regressed MRR (2026-04-10).
    search_query = query

    # Entity overlap boost: extract entity tags from the query and pass to SQL.
    # Memories sharing entities with the query get up to 1.3x relevance boost.
    # See docs/plans/2026-04-06-entity-graph-spreading-activation.md Stage 1.
    query_entities = extract_entities(query)
    query_entity_tags = query_entities if query_entities else None

    # Elastic K: set-queries (ordering, summary, multi-session) get 2x limit
    # for broader coverage of scattered facts across the timeline.
    elastic_limit = limit * 2

    # Ordering queries: strided retrieval + activation + chronological sort
    if is_ordering_query(query):
        results = hybrid_search_memories(
            query_text=search_query,
            query_embedding=embedding,
            profile=profile,
            limit=elastic_limit * 5,
            tags=tags,
            source=source,
            profiles=profiles,
            query_entity_tags=query_entity_tags,
        )
        if results:
            results = _strided_retrieval(results, elastic_limit * 2)
        results = _merge_activation_results(
            results or [],
            query_entity_tags,
            profile,
            elastic_limit * 2,
            graph_fraction=0.5,
        )
        if results:
            for r in results:
                r["_sort_date"] = _extract_memory_date(r) or "9999"
            results.sort(key=lambda r: r["_sort_date"])
            results = results[:elastic_limit]
        return results

    # Multi-hop temporal: entity-centric bridge retrieval + threading
    if is_multi_hop_temporal(query):
        bridge_results = _bridge_retrieval(
            query,
            profile,
            elastic_limit,
            tags,
            source,
            embedding_generator,
        )
        if bridge_results:
            results = _merge_bridge_results(
                bridge_results,
                query,
                embedding,
                profile,
                elastic_limit,
                tags,
                source,
            )
            if results:
                results = _entity_thread(
                    results, query, embedding, profile, elastic_limit, tags, source
                )
            return results

    # Cross-reference queries: spreading activation for entity bridging.
    # Two parallel signals: hybrid search (semantic + keyword) + entity graph
    # walk (spreading activation). Merge boosts hybrid results by activation
    # score but limits bridge doc injection (graph_fraction=0.15) to avoid
    # displacing the gold answer at rank 1.
    if is_cross_reference_query(query):
        results = hybrid_search_memories(
            query_text=search_query,
            query_embedding=embedding,
            profile=profile,
            limit=elastic_limit * 3,
            tags=tags,
            source=source,
            profiles=profiles,
            query_entity_tags=query_entity_tags,
        )
        return _merge_activation_results(
            results or [],
            query_entity_tags,
            profile,
            elastic_limit,
            graph_fraction=0.15,
        )

    # Broad summary queries: strided retrieval + activation for entity diversity.
    if is_broad_summary_query(query):
        results = hybrid_search_memories(
            query_text=search_query,
            query_embedding=embedding,
            profile=profile,
            limit=elastic_limit * 5,
            tags=tags,
            source=source,
            profiles=profiles,
            query_entity_tags=query_entity_tags,
        )
        if results:
            results = _strided_retrieval(results, elastic_limit)
        results = _merge_activation_results(
            results or [],
            query_entity_tags,
            profile,
            elastic_limit,
            graph_fraction=0.4,
        )
        return results

    # Standard search path — fetch wider pool for TDR density check.
    # NOTE: temporal queries keep limit*3 (not higher) — increasing to 5x
    # regressed temporal-reasoning by -1.5pp because extra candidates dilute
    # _temporal_rerank's top-k. Non-temporal gets 3x for TDR headroom.
    fetch_limit = limit * 3

    if graph_depth > 0:
        from ogham.database import graph_augmented_search

        results = graph_augmented_search(
            query_text=search_query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            graph_depth=graph_depth,
            tags=tags,
            source=source,
        )
    else:
        # Standard path gets gated recency decay (0.01 = ~69-day half-life).
        # Ordering and summary paths above do NOT get recency decay because
        # they need evidence from across the full timeline.
        results = hybrid_search_memories(
            query_text=search_query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            tags=tags,
            source=source,
            profiles=profiles,
            query_entity_tags=query_entity_tags,
            recency_decay=0.01,
        )

    # Temporal Diversity Re-ranking (TDR): density-gated soft penalty.
    # Fires only when top-20 results are temporally clustered (≥80% in ≤5%
    # of the total time-range). Prevents semantic density collapse on
    # multi-session counting queries without forced injection.
    if results and len(results) > limit:
        results = _tdr_rerank(results, limit)

    # Single-anchor temporal re-ranking
    if results and has_temporal_intent(query):
        results = _temporal_rerank(results, query)
        results = results[:limit]

    return results


def _bridge_retrieval(
    query: str,
    profile: str,
    limit: int,
    tags: list[str] | None,
    source: str | None,
    embedding_generator: Callable[[str], list[float]] = generate_embedding,
) -> list[dict[str, Any]]:
    """Entity-centric bridge retrieval for multi-hop temporal queries.

    Extracts entity anchors from the query, runs separate keyword searches
    for each anchor, and returns results grouped by anchor.
    """
    anchors = extract_query_anchors(query)
    if not anchors:
        return []

    # Split limit evenly across anchors
    per_anchor_limit = max(limit, 10) // len(anchors) if anchors else limit

    all_results = []
    for anchor in anchors:
        # Path A: Semantic + keyword hybrid search
        anchor_embedding = embedding_generator(anchor)
        results = hybrid_search_memories(
            query_text=anchor,
            query_embedding=anchor_embedding,
            profile=profile,
            limit=per_anchor_limit * 2,
            tags=tags,
            source=source,
        )

        if results:
            for r in results:
                r["_bridge_anchor"] = anchor
            all_results.append(results)

    if not all_results:
        return []

    # Interleave results from each anchor (round-robin)
    interleaved = []
    max_len = max(len(group) for group in all_results)
    for i in range(max_len):
        for group in all_results:
            if i < len(group):
                interleaved.append(group[i])

    return interleaved


def _merge_bridge_results(
    bridge_results: list[dict[str, Any]],
    original_query: str,
    original_embedding: list[float],
    profile: str,
    limit: int,
    tags: list[str] | None,
    source: str | None,
) -> list[dict[str, Any]]:
    """Merge bridge retrieval results with standard search, deduplicating.

    Bridge results get a 1.5x boost. The final list is deduped by ID
    and trimmed to limit.
    """
    # Also run the standard search as fallback
    standard_results = hybrid_search_memories(
        query_text=original_query,
        query_embedding=original_embedding,
        profile=profile,
        limit=limit * 2,
        tags=tags,
        source=source,
    )

    # Logarithmic boost for bridge results (run 7 config -- best MRR)
    import math

    for r in bridge_results:
        if "relevance" in r and r["relevance"] is not None:
            ccf = r["relevance"]
            r["relevance"] = ccf + 0.3 * math.log1p(ccf)

    # Merge: bridge results first, then standard, dedup by ID
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for r in bridge_results + standard_results:
        rid = r.get("id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)

    # Timestamp tiebreaker for same-score results
    for r in merged:
        created = r.get("created_at", "")
        if created and "relevance" in r and r["relevance"] is not None:
            try:
                ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                r["relevance"] += ts.timestamp() * 1e-15
            except (ValueError, TypeError):
                pass

    # Sort by relevance
    merged.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return merged[:limit]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _exact_content_search(anchor: str, profile: str, limit: int) -> list[dict[str, Any]]:
    """Direct content search using ILIKE -- bypasses ts_rank normalisation.

    Finds entity mentions buried in long documents where tsvector
    scores them too low (e.g. "Emma" in 18K chars of marketing content).
    """
    from ogham.database import get_backend

    backend = get_backend()

    # Extract the most specific word from the anchor (longest, likely a name)
    words = [w for w in anchor.split() if len(w) > 2]
    if not words:
        return []

    # Use the most distinctive word (longest, likely a proper noun)
    search_term = max(words, key=len)

    try:
        if hasattr(backend, "_execute"):
            # Postgres backend -- direct SQL
            sql = """
                SELECT id, content, metadata, source, profile, tags,
                       confidence, created_at, updated_at,
                       0.5::float AS similarity, 0.5::float AS relevance
                FROM memories
                WHERE profile = %(profile)s
                  AND content ILIKE %(pattern)s
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY created_at DESC
                LIMIT %(limit)s
            """
            results = backend._execute(
                sql,
                {"profile": profile, "pattern": f"%{search_term}%", "limit": limit},
                fetch="all",
            )
            return results if results else []
        elif hasattr(backend, "_get_client"):
            # Supabase/PostgREST backend
            result = (
                backend._get_client()
                .from_("memories")
                .select("*")
                .eq("profile", profile)
                .ilike("content", f"*{search_term}*")
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []
    except Exception as e:
        logger.debug("Exact content search failed for '%s': %s", search_term, e)

    return []


def _tdr_rerank(
    results: list[dict[str, Any]],
    limit: int,
    density_threshold: float = 0.80,
    spread_threshold: float = 0.05,
    decay_lambda: float = 0.8,
    null_date_penalty: float = 0.95,
) -> list[dict[str, Any]]:
    """Temporal Diversity Re-ranking with density gating.

    Prevents semantic density collapse on multi-session counting queries
    by soft-penalising results from already-represented dates. Only fires
    when the top-20 results are temporally clustered.

    Design: Gemini 3.1 Pro stress-test (2026-04-11), refined via three
    peer-review rounds. Key principle: SOFT re-ranking only, never forced
    injection (boundary-anchoring regressed -1.5pp on temporal-reasoning).

    Args:
        density_threshold: gate fires when this fraction of top-20 falls
            within spread_threshold of the total time-range (default 0.80)
        spread_threshold: what fraction of total time-range counts as
            "clustered" (default 0.05 = 5%)
        decay_lambda: penalty base per duplicate date (default 0.8)
        null_date_penalty: baseline multiplier for date-less memories (0.95)
    """
    if len(results) <= limit:
        return results

    # Extract dates for all candidates
    dated_results: list[tuple[str | None, int, dict]] = []
    for i, r in enumerate(results):
        d = _extract_memory_date(r)
        dated_results.append((d, i, r))

    # Compute temporal spread ratio on top-20
    top_20 = dated_results[:20]
    all_dates = [d for d, _, _ in dated_results if d is not None]
    top_20_dates = [d for d, _, _ in top_20 if d is not None]

    if len(all_dates) < 2 or len(top_20_dates) < 2:
        # Not enough dated memories to compute spread — pass through
        return results[:limit]

    all_dates_sorted = sorted(all_dates)
    total_range_days = max(
        1,
        (_date_to_ordinal(all_dates_sorted[-1]) - _date_to_ordinal(all_dates_sorted[0])),
    )

    top_20_sorted = sorted(top_20_dates)
    top_20_range_days = _date_to_ordinal(top_20_sorted[-1]) - _date_to_ordinal(top_20_sorted[0])

    spread_ratio = top_20_range_days / total_range_days if total_range_days > 0 else 1.0
    top_20_concentration = len(top_20_dates) / max(len(top_20), 1)

    # Density gate: do top-20 cluster in a narrow time band?
    if not (top_20_concentration >= density_threshold and spread_ratio <= spread_threshold):
        # Top-20 are already diverse — no TDR needed
        return results[:limit]

    logger.debug(
        "TDR gate fired: %.0f%% of top-20 in %.1f%% of time-range (threshold: %.0f%%/%.0f%%)",
        top_20_concentration * 100,
        spread_ratio * 100,
        density_threshold * 100,
        spread_threshold * 100,
    )

    # Greedy selection with soft date-penalty
    selected: list[dict] = []
    date_counts: dict[str, int] = {}
    _null_counter = 0

    for date_str, _orig_idx, r in dated_results:
        if len(selected) >= limit:
            break

        relevance = r.get("relevance", 0.0)

        if date_str is None:
            # Null-date: unique virtual bucket per memory + baseline penalty
            _null_counter += 1
            bucket = f"_null_{_null_counter}"
            penalty = null_date_penalty
        else:
            bucket = date_str
            count = date_counts.get(bucket, 0)
            penalty = decay_lambda**count

        adjusted_score = relevance * penalty

        # Only select if the adjusted score is positive
        if adjusted_score > 0:
            r["_tdr_adjusted"] = adjusted_score
            r["_tdr_bucket"] = bucket
            selected.append(r)
            date_counts[bucket] = date_counts.get(bucket, 0) + 1

    # Re-sort by adjusted score (highest first)
    selected.sort(key=lambda r: r.get("_tdr_adjusted", 0), reverse=True)

    return selected[:limit]


def _date_to_ordinal(date_str: str) -> int:
    """Convert YYYY-MM-DD string to an ordinal day number for arithmetic."""
    try:
        parts = date_str.split("-")
        from datetime import date

        return date(int(parts[0]), int(parts[1]), int(parts[2])).toordinal()
    except (ValueError, IndexError):
        return 0


_CONTENT_DATE_RE = re.compile(r"\[Date:\s*(\d{4}-\d{2}-\d{2})\]")

# Direction keywords for asymmetric temporal decay
_dir = get_direction_words("en")
_AFTER_WORDS = frozenset(_dir.get("after", []))
_BEFORE_WORDS = frozenset(_dir.get("before", []))


def _extract_memory_date(r: dict[str, Any]) -> str | None:
    """Extract a date from a memory (metadata > content prefix > created_at)."""
    meta_dates = r.get("metadata", {}).get("dates", [])
    if meta_dates:
        return meta_dates[0]

    content = r.get("content", "")
    date_match = _CONTENT_DATE_RE.search(content)
    if date_match:
        return date_match.group(1)

    created = str(r.get("created_at", ""))[:10]
    if created and len(created) == 10:
        return created

    return None


def _boundary_anchored_inject(
    results: list[dict[str, Any]],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Force-inject the chronological boundary nodes into the result set.

    For temporal-reasoning queries ("how long since X", "how many weeks between"),
    the reader needs the earliest and latest mentions relative to the query's
    temporal anchor to establish the timeline boundaries. Without them it
    guesses dates or returns "no information".

    Algorithm (per Gemini 3.1 Pro stress-test, 2026-04-11):
      1. Extract dates from all candidates
      2. Find top-1 chronologically BEFORE the anchor (or earliest overall)
      3. Find top-1 chronologically AFTER the anchor (or latest overall)
      4. Force-inject both into the result set (deduplicated)
      5. Fill remaining slots from the semantic ranker

    This is a deterministic boundary lookup, not a diversity re-ranker.
    Immune to thin-profile risk because it doesn't care about density.
    """
    if not results or len(results) <= 2:
        return results

    # Date every candidate
    dated: list[tuple[str, dict]] = []
    for r in results:
        d = _extract_memory_date(r)
        if d:
            dated.append((d, r))

    if len(dated) < 2:
        return results  # not enough dated memories to establish boundaries

    dated.sort(key=lambda x: x[0])

    # Try to find anchor date from query via extract_dates
    from ogham.extraction import extract_dates

    query_dates = extract_dates(query)
    if query_dates:
        # Use the first extracted date as anchor
        anchor = query_dates[0]
    else:
        # No explicit anchor — use the midpoint of the date range
        anchor = dated[len(dated) // 2][0]

    # Boundary: latest memory BEFORE anchor
    before = None
    for date_str, r in reversed(dated):
        if date_str <= anchor:
            before = r
            break

    # Boundary: earliest memory AFTER anchor
    after = None
    for date_str, r in dated:
        if date_str >= anchor:
            after = r
            break

    # If anchor is outside the range, use the edges
    if before is None:
        before = dated[0][1]
    if after is None:
        after = dated[-1][1]

    # Force-inject boundary nodes at the front, deduped
    seen_ids: set[str] = set()
    injected: list[dict] = []

    for boundary in [before, after]:
        rid = str(boundary.get("id", ""))
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            injected.append(boundary)

    # Fill remaining slots from the original order, deduped
    for r in results:
        if len(injected) >= limit:
            break
        rid = str(r.get("id", ""))
        if rid not in seen_ids:
            seen_ids.add(rid)
            injected.append(r)

    return injected[:limit]


def _detect_direction(query: str) -> str:
    """Detect temporal direction: 'future', 'past', or 'near'."""
    words = set(query.lower().split())
    if words & _AFTER_WORDS:
        return "future"
    if words & _BEFORE_WORDS:
        return "past"
    return "near"


def _temporal_rerank(
    results: list[dict[str, Any]], query: str, sigma: float = 3.0
) -> list[dict[str, Any]]:
    """Gaussian decay + directional hard penalty temporal re-ranking.

    Uses Gaussian decay centered on the anchor date. σ controls the window
    width (3 = tight week-scale, 7 = loose). Wrong-direction results get
    hard 0.1x penalty (the "temporal cliff"). Sub-day precision via
    fractional days from timestamps.

    Decay: exp(-delta²/2σ²) concentrates boost near anchor
    Directional: 0.1x for wrong side of anchor (squash, not just decay)
    Same-day grace: delta < 1 day gets 1.5x boost
    Tiebreaker: 1e-6 * timestamp for same-score results
    """
    import math

    date_range = resolve_temporal_query(query)
    if not date_range:
        return results

    range_start, range_end = date_range
    try:
        anchor_start = datetime.fromisoformat(range_start)
        anchor_end = datetime.fromisoformat(range_end)
        anchor = anchor_start + (anchor_end - anchor_start) / 2
    except (ValueError, TypeError):
        return results

    direction = _detect_direction(query)

    for r in results:
        mem_date_str = _extract_memory_date(r)
        if not mem_date_str:
            continue

        try:
            # Use full timestamp precision when available
            if len(mem_date_str) > 10:
                mem_dt = datetime.fromisoformat(mem_date_str.replace("Z", "+00:00"))
            else:
                mem_dt = datetime.fromisoformat(mem_date_str)
        except (ValueError, TypeError):
            continue

        # Delta in fractional days (sub-day precision)
        delta_days = (mem_dt - anchor).total_seconds() / 86400.0

        # 1. Directional hard penalty (the "cliff")
        dir_multiplier = 1.0
        if direction == "future" and delta_days < -0.5:
            dir_multiplier = 0.1
        elif direction == "past" and delta_days > 0.5:
            dir_multiplier = 0.1

        # 2. Gaussian decay (proximity boost)
        if abs(delta_days) < 0.5:
            # Same-day grace period
            decay = 1.5
        else:
            decay = 1.0 + 0.5 * math.exp(-(delta_days**2) / (2 * sigma**2))

        # 3. Tiebreaker: tiny timestamp fraction
        tiebreak = mem_dt.timestamp() * 1e-15 if hasattr(mem_dt, "timestamp") else 0

        # 4. Apply
        if "relevance" in r and r["relevance"] is not None:
            r["relevance"] = r["relevance"] * dir_multiplier * decay + tiebreak

    results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return results


def _entity_thread(
    pass1_results: list[dict],
    query: str,
    embedding: list[float],
    profile: str,
    limit: int,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    """Entity-anchored threading: two-pass retrieval.

    Pass 1: Use the top semantic results to identify core entities (nouns).
    Pass 2: Search for ALL occurrences of those entities across the index.
    Interleave Pass 1 (answer) with Pass 2 (history) for full coverage.
    """
    import re

    if not pass1_results:
        return pass1_results

    # Extract key nouns from top-3 results + query
    # Look for capitalised phrases, quoted terms, technical nouns
    entity_sources = [query]
    for r in pass1_results[:3]:
        content = r.get("content", "")[:500]
        entity_sources.append(content)

    combined = " ".join(entity_sources)

    # Extract candidate entities: multi-word capitalised phrases, technical terms
    # Skip common words that appear in every conversation
    _STOP = frozenset(
        {
            "the",
            "a",
            "an",
            "i",
            "my",
            "me",
            "you",
            "your",
            "we",
            "our",
            "this",
            "that",
            "it",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "not",
            "no",
            "and",
            "or",
            "but",
            "if",
            "then",
            "so",
            "for",
            "with",
            "from",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "as",
            "how",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "all",
            "each",
            "every",
            "some",
            "any",
            "many",
            "much",
            "more",
            "most",
            "other",
            "new",
            "old",
            "first",
            "last",
            "next",
            "different",
            "same",
            "user",
            "assistant",
            "date",
            "time",
            "project",
            "app",
            "code",
            "feature",
            "system",
        }
    )

    # Find 2-3 word capitalised phrases (e.g. "Flask Login", "Transactions Table")
    cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", combined)
    # Find technical terms (hyphenated, camelCase, or ALL_CAPS)
    tech_terms = re.findall(
        r"\b([A-Z][a-zA-Z]+(?:-[a-zA-Z]+)+|[a-z]+[A-Z]\w+|[A-Z]{2,}[a-z]\w*)\b", combined
    )
    # Find quoted terms
    quoted = re.findall(r'"([^"]+)"', combined)

    # Also extract significant single words (appear in query, not stopwords)
    query_words = [w for w in re.findall(r"\b\w{4,}\b", query.lower()) if w not in _STOP]

    # Build entity query string (top 5 most specific terms)
    entities = []
    for phrase in cap_phrases[:3]:
        if phrase.lower() not in _STOP:
            entities.append(phrase)
    for term in tech_terms[:2]:
        entities.append(term)
    for term in quoted[:2]:
        entities.append(term)
    # Add top query words as fallback
    for w in query_words[:3]:
        if w not in [e.lower() for e in entities]:
            entities.append(w)

    if not entities:
        return pass1_results[:limit]

    # Pass 2: search for entity occurrences
    entity_query = " ".join(entities[:5])
    pass2_results = hybrid_search_memories(
        query_text=entity_query,
        query_embedding=embedding,
        profile=profile,
        limit=limit * 2,
        tags=tags,
        source=source,
    )

    # Interleave: alternate Pass 1 (answer) and Pass 2 (history), deduped
    merged: list[dict] = []
    seen: set[str] = set()

    p1_iter = iter(pass1_results)
    p2_iter = iter(pass2_results)
    p1_done = False
    p2_done = False

    while len(merged) < limit:
        # Take from Pass 1
        if not p1_done:
            try:
                r = next(p1_iter)
                rid = r.get("id", "")
                if rid not in seen:
                    seen.add(rid)
                    merged.append(r)
            except StopIteration:
                p1_done = True

        if len(merged) >= limit:
            break

        # Take from Pass 2
        if not p2_done:
            try:
                r = next(p2_iter)
                rid = r.get("id", "")
                if rid not in seen:
                    seen.add(rid)
                    merged.append(r)
            except StopIteration:
                p2_done = True

        if p1_done and p2_done:
            break

    return merged[:limit]


def _strided_retrieval(results: list[dict], limit: int) -> list[dict]:
    """Diversify results across temporal buckets to prevent clumping.

    Divides the timeline into N equal buckets (N = limit) and takes
    the top-1 result from each bucket by relevance, round-robin.
    """
    if len(results) <= limit:
        return results

    dated = []
    undated = []
    for r in results:
        d = _extract_memory_date(r)
        if d:
            dated.append((d, r))
        else:
            undated.append(r)

    if not dated:
        return results[:limit]

    # Sort chronologically
    dated.sort(key=lambda x: x[0])

    # Split into N equal temporal buckets
    n_buckets = min(limit, len(dated))
    bucket_size = max(1, len(dated) // n_buckets)
    buckets = []
    for i in range(0, len(dated), bucket_size):
        # Sort each bucket by relevance (highest first)
        bucket = sorted(
            [r for _, r in dated[i : i + bucket_size]],
            key=lambda r: r.get("relevance", 0),
            reverse=True,
        )
        buckets.append(bucket)

    # Round-robin: take best from each bucket
    diversified: list[dict] = []
    seen: set[str] = set()
    max_rounds = max(len(b) for b in buckets) if buckets else 0
    for round_num in range(max_rounds):
        for bucket in buckets:
            if round_num < len(bucket):
                r = bucket[round_num]
                rid = r.get("id", "")
                if rid not in seen:
                    seen.add(rid)
                    diversified.append(r)
                    if len(diversified) >= limit:
                        return diversified

    # Fill from undated if needed
    for r in undated:
        if len(diversified) >= limit:
            break
        rid = r.get("id", "")
        if rid not in seen:
            seen.add(rid)
            diversified.append(r)

    return diversified[:limit]


def _graph_rerank(
    results: list[dict],
    query_entity_tags: list[str] | None,
    profile: str,
    boost_weight: float = 0.3,
) -> list[dict]:
    """Re-rank results using structured entity graph connectivity.

    Uses the memory_entities table to boost results that share entities
    with the query and with each other. Unlike _entity_thread (which does
    a second text search), this only re-ranks existing results -- no new
    candidates, no dilution.

    Boost = (query_overlap + cross_connectivity) * boost_weight
    - query_overlap: fraction of query entities found in this result's entities
    - cross_connectivity: fraction of result's entities shared by 2+ other results
    """
    if not results or len(results) <= 1:
        return results

    from ogham.backends.postgres import PostgresBackend
    from ogham.database import get_backend

    backend = get_backend()
    if not isinstance(backend, PostgresBackend):
        return results

    # Batch lookup: get entity IDs for all result memory IDs
    mem_ids = [r.get("id") for r in results if r.get("id")]
    if not mem_ids:
        return results

    try:
        pool = backend._get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Get entity sets for each memory in the result set
                cur.execute(
                    """
                    SELECT me.memory_id, array_agg(e.canonical_name || ':' || e.entity_type)
                    FROM memory_entities me
                    JOIN entities e ON e.id = me.entity_id
                    WHERE me.memory_id = ANY(%s) AND me.profile = %s
                    GROUP BY me.memory_id
                    """,
                    (mem_ids, profile),
                )
                mem_entities: dict[str, set[str]] = {}
                for mid, ents in cur.fetchall():
                    mem_entities[str(mid)] = set(ents)
    except Exception:
        return results

    if not mem_entities:
        return results

    # Build entity frequency map across all results (for connectivity scoring)
    entity_freq: dict[str, int] = {}
    for ents in mem_entities.values():
        for e in ents:
            entity_freq[e] = entity_freq.get(e, 0) + 1

    # Normalise query entity tags for matching (e.g. "person:John" -> "John:person")
    query_ent_set: set[str] = set()
    if query_entity_tags:
        for tag in query_entity_tags:
            parts = tag.split(":", 1)
            if len(parts) == 2:
                query_ent_set.add(f"{parts[1]}:{parts[0]}")

    # Score each result
    for r in results:
        rid = str(r.get("id", ""))
        ents = mem_entities.get(rid, set())
        if not ents:
            continue

        # 1. Query overlap: what fraction of query entities does this memory have?
        query_overlap = 0.0
        if query_ent_set:
            overlap = len(ents & query_ent_set)
            query_overlap = overlap / len(query_ent_set)

        # 2. Cross-connectivity: what fraction of this memory's entities appear
        #    in 2+ other results? (shared entities = topical cluster)
        shared = sum(1 for e in ents if entity_freq.get(e, 0) >= 2)
        connectivity = shared / len(ents) if ents else 0.0

        # Combined boost (capped at 1.0 to prevent over-boosting)
        boost = min(1.0, query_overlap + connectivity * 0.5) * boost_weight

        if "relevance" in r and r["relevance"] is not None:
            r["relevance"] = r["relevance"] * (1.0 + boost)

    results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return results


_DENSITY_CACHE: dict[str, tuple[float, float]] = {}
_DENSITY_TTL_SECONDS = 300.0


def _profile_graph_density(profile: str) -> float:
    """Measure entity graph density (edges per entity) for a profile.

    Returns edges-per-entity ratio. Dense profiles (single-chat, BEAM-style)
    typically return 2-4. Sparse multi-session profiles return closer to 1.

    Cached for 5 minutes per profile to avoid repeated queries. Falls back
    to 2.0 (neutral) on error.
    """
    import time as _time

    now = _time.time()
    cached = _DENSITY_CACHE.get(profile)
    if cached and (now - cached[1]) < _DENSITY_TTL_SECONDS:
        return cached[0]

    try:
        from ogham.database import get_backend

        backend = get_backend()
        rows = backend._execute(
            """SELECT
                 count(distinct entity_id)::float as entities,
                 count(*)::float as edges
               FROM memory_entities
               WHERE profile = %(profile)s""",
            {"profile": profile},
            fetch="one",
        )
        if rows and rows.get("entities", 0) > 0:
            density = float(rows["edges"]) / float(rows["entities"])
        else:
            density = 2.0
    except Exception:
        density = 2.0

    _DENSITY_CACHE[profile] = (density, now)
    return density


def _density_adaptive_activation_weight(
    profile: str,
    base_weight: float = 0.15,
    min_weight: float = 0.05,
    max_weight: float = 0.30,
) -> float:
    """Scale activation weight inversely with graph density.

    Pattern from Shodh (varun29ankuS/shodh-memory): sparse entity graphs
    have higher-signal edges (Hebbian-pruned), dense graphs have more
    noise. This measures the profile's edges/entity ratio and scales
    activation_weight accordingly:

    - Sparse (<=1.5 edges/entity): scale to max_weight (0.30) -- trust graph more
    - Medium (1.5-2.5): use base_weight (0.15)
    - Dense (>=3.5 edges/entity): scale to min_weight (0.05) -- reduce graph influence

    Directly mitigates cluster saturation on single-chat benchmarks.
    """
    density = _profile_graph_density(profile)

    if density <= 1.5:
        return max_weight
    if density >= 3.5:
        return min_weight
    # Linear interpolation between max_weight @ 1.5 and min_weight @ 3.5
    # range width = 2.0, distance from sparse end = (density - 1.5)
    frac = (density - 1.5) / 2.0
    return max_weight - frac * (max_weight - min_weight)


def _merge_activation_results(
    hybrid_results: list[dict],
    query_entity_tags: list[str] | None,
    profile: str,
    limit: int,
    graph_fraction: float = 0.3,
    activation_weight: float | None = None,
) -> list[dict]:
    """Merge hybrid search results with spreading activation graph walk.

    Two parallel signals:
    1. Hybrid search (semantic + keyword) -> relevance score
    2. Entity graph walk (spreading activation) -> activation score

    When activation_weight is None (default), measures profile graph density
    and scales weight adaptively (dense profiles get less graph influence).
    Pass a float to override.

    Memories in both sets get boosted. Graph-only memories (no hybrid match)
    enter as "bridge" documents at graph_fraction of the result set.
    """
    if not query_entity_tags:
        return hybrid_results[:limit]

    # Density-adaptive activation weight (Shodh-inspired).
    if activation_weight is None:
        activation_weight = _density_adaptive_activation_weight(profile)
        logger.debug(
            "Density-adaptive activation_weight for profile=%s: %.3f",
            profile,
            activation_weight,
        )

    try:
        activated = spread_entity_activation(
            entity_tags=query_entity_tags,
            profile=profile,
            max_depth=2,
            decay=0.65,
            min_activation=0.05,
            max_results=limit * 3,
        )
    except Exception:
        logger.debug("Spreading activation failed, falling back to hybrid only")
        return hybrid_results[:limit]

    if not activated:
        return hybrid_results[:limit]

    # Build activation lookup: memory_id -> activation score
    act_map: dict[str, float] = {}
    for row in activated:
        mid = str(row.get("memory_id", ""))
        act = float(row.get("activation", 0))
        act_map[mid] = max(act_map.get(mid, 0), act)

    # Squash activation via tanh to prevent runaway MRR-theft from
    # highly connected clusters. tanh caps influence while preserving
    # differential signal from bridge entities.
    import math

    act_map = {k: math.tanh(v) for k, v in act_map.items()}

    # Score hybrid results: relevance + activation_weight * activation
    hybrid_ids: set[str] = set()
    for r in hybrid_results:
        rid = str(r.get("id", ""))
        hybrid_ids.add(rid)
        act = act_map.get(rid, 0.0)
        base = r.get("relevance", 0) or 0
        r["relevance"] = base + activation_weight * act

    # Re-sort by boosted relevance (re-ranking only — no eviction risk)
    hybrid_results.sort(key=lambda r: r.get("relevance", 0), reverse=True)

    # Conditional expansion: for cross-reference queries, append a small
    # number of graph-only candidates at the END (not interleaved).
    # These are "bridge" documents with zero semantic overlap but strong
    # entity connections. Appended, not interleaved, so they never displace
    # the golden semantic result at Rank 1.
    if graph_fraction > 0 and act_map:
        from ogham.database import get_memory_by_id

        n_expand = min(3, int(limit * graph_fraction))
        for mid, act in sorted(act_map.items(), key=lambda x: -x[1]):
            if mid not in hybrid_ids and n_expand > 0:
                mem = get_memory_by_id(mid, profile)
                if mem:
                    mem["relevance"] = activation_weight * act
                    mem["_graph_only"] = True
                    hybrid_results.append(mem)
                    n_expand -= 1

    return hybrid_results[:limit]


def _mmr_rerank(
    candidates: list[dict],
    query_embedding: list[float],
    limit: int,
    lambda_param: float = 0.5,
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking for diversity.

    Iteratively selects documents balancing relevance to query
    with diversity from already-selected documents.

    lambda_param: 0.5 = balanced, 0.3 = high diversity (contradiction queries),
                  0.7 = high relevance (standard queries)
    """
    if len(candidates) <= limit:
        return candidates

    # Use relevance scores as Sim1 (normalized)
    scores = [c.get("relevance", c.get("similarity", 0.0)) or 0.0 for c in candidates]
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        sim1 = [s / max_score for s in scores]
    else:
        sim1 = [0.0] * len(candidates)

    # For Sim2 (inter-document similarity), use content overlap as proxy
    # since we don't have stored embeddings in results
    contents = [c.get("content", "") for c in candidates]

    def _content_overlap(a: str, b: str) -> float:
        """Quick word-level Jaccard similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # Greedy MMR selection
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    # First pick: highest relevance
    best_idx = max(remaining, key=lambda i: sim1[i])
    selected.append(best_idx)
    remaining.remove(best_idx)

    while len(selected) < limit and remaining:
        best_mmr = -float("inf")
        best_candidate = remaining[0]

        for i in remaining:
            # Max similarity to any already-selected document
            max_sim2 = max(_content_overlap(contents[i], contents[j]) for j in selected)
            mmr_score = lambda_param * sim1[i] - (1 - lambda_param) * max_sim2
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = i

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return [candidates[i] for i in selected]
