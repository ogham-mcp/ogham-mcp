"""Shared memory service pipeline.

Used by both the MCP tool layer (tools/memory.py) and the gateway REST API.
Handles: content validation, date extraction, entity extraction,
importance scoring, embedding generation, surprise scoring,
storage, and auto-linking.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ogham.data.loader import get_direction_words
from ogham.database import auto_link_memory as db_auto_link
from ogham.database import get_profile_ttl as db_get_profile_ttl
from ogham.database import hybrid_search_memories, record_access
from ogham.database import store_memory as db_store
from ogham.embeddings import generate_embedding
from ogham.extraction import (
    compute_importance,
    extract_dates,
    extract_entities,
    extract_query_anchors,
    extract_recurrence,
    has_temporal_intent,
    is_broad_summary_query,
    is_multi_hop_temporal,
    is_ordering_query,
    resolve_temporal_query,
)

logger = logging.getLogger(__name__)


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

    # Generate embedding (skip if pre-computed, e.g. from gateway cache)
    if embedding is None:
        embedding = generate_embedding(content)

    # Compute surprise score: how novel is this vs existing memories?
    surprise = 0.5
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

    # Auto-link
    if auto_link:
        links_created = db_auto_link(
            memory_id=result["id"],
            embedding=embedding,
            profile=profile,
        )
        response["links_created"] = links_created

    return response


def search_memories_enriched(
    query: str,
    profile: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
    graph_depth: int = 0,
    embedding: list[float] | None = None,
    profiles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Full search pipeline: retrieve, rerank (optional), record access.

    Pipeline stages:
    1. _search_memories_raw: intent detection, retrieval, temporal/entity enrichment
    2. _maybe_rerank: optional FlashRank cross-encoder reranking (RERANK_ENABLED=true)
    3. Record access for retrieved memories
    """
    results = _search_memories_raw(
        query,
        profile,
        limit,
        tags,
        source,
        graph_depth,
        embedding,
        profiles,
    )

    if results:
        results = _maybe_rerank(query, results, limit)
        results = _reorder_for_attention(results)
        record_access([r["id"] for r in results])

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
) -> list[dict[str, Any]]:
    """Retrieve memories via intent-aware search paths. No reranking, no access recording."""
    if embedding is None:
        embedding = generate_embedding(query)

    # Elastic K: set-queries (ordering, summary, multi-session) get 2x limit
    # for broader coverage of scattered facts across the timeline.
    elastic_limit = limit * 2

    # Ordering queries: strided retrieval + entity threading + chronological sort
    if is_ordering_query(query):
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=elastic_limit * 5,
            tags=tags,
            source=source,
            profiles=profiles,
        )
        if results:
            strided = _strided_retrieval(results, elastic_limit * 2)
            threaded = _entity_thread(
                strided, query, embedding, profile, elastic_limit * 2, tags, source
            )
            for r in threaded:
                r["_sort_date"] = _extract_memory_date(r) or "9999"
            threaded.sort(key=lambda r: r["_sort_date"])
            results = threaded[:elastic_limit]
        return results

    # Multi-hop temporal: entity-centric bridge retrieval + threading
    if is_multi_hop_temporal(query):
        bridge_results = _bridge_retrieval(query, profile, elastic_limit, tags, source)
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

    # Broad summary queries: strided retrieval for timeline coverage
    if is_broad_summary_query(query):
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=elastic_limit * 5,
            tags=tags,
            source=source,
            profiles=profiles,
        )
        if results:
            results = _strided_retrieval(results, elastic_limit)
            results = _mmr_rerank(results, embedding, elastic_limit, lambda_param=0.5)
        return results

    # Standard search path
    fetch_limit = limit * 3 if has_temporal_intent(query) else limit

    if graph_depth > 0:
        from ogham.database import graph_augmented_search

        results = graph_augmented_search(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            graph_depth=graph_depth,
            tags=tags,
            source=source,
        )
    else:
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            tags=tags,
            source=source,
            profiles=profiles,
        )

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
        anchor_embedding = generate_embedding(anchor)
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
