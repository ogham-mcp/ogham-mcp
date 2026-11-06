# Ogham MCP — Search Deep Dive

The search system in `service.py` is the most complex part of Ogham. It implements multiple retrieval strategies depending on query classification.

## Query Classification

Four classifiers in `extraction.py` route queries to different retrieval paths. All use regex — no LLM calls.

| Classifier | Detects | Example |
|-----------|---------|---------|
| `is_ordering_query()` | Chronological ordering requests | "What is the order of my projects from earliest to latest?" |
| `is_multi_hop_temporal()` | Cross-event temporal reasoning | "How many months between X and Y?" |
| `is_broad_summary_query()` | Summary/overview requests | "Give me a comprehensive summary of how my project has progressed" |
| `has_temporal_intent()` | Any temporal keywords | "What happened last week?" |

## Retrieval Strategies

### 1. Standard Path (default)

```text
Query → Embed → hybrid_search_memories() → [optional graph traversal] → [optional temporal rerank] → results
```

- `hybrid_search_memories()` is a PostgreSQL function using Reciprocal Rank Fusion (RRF) of vector similarity + full-text search
- If `graph_depth > 0`, follows relationship edges from top-5 results via `graph_augmented_search()`
- If temporal intent detected, applies temporal re-ranking

### 2. Ordering Queries

```text
Query → Embed → hybrid_search(10x limit) → strided_retrieval → entity_thread → chronological sort → trim
```

- **Strided retrieval**: Prevents temporal clumping. Divides timeline into N equal buckets, takes top-1 by relevance from each, round-robin
- **Entity threading**: Two-pass. Pass 1: extract entities (capitalised phrases, technical terms, quoted strings) from top-3 results + query. Pass 2: search for ALL occurrences of those entities. Interleave results
- Final chronological sort ensures proper ordering

### 3. Multi-hop Temporal Queries

```text
Query → extract_query_anchors() → per-anchor searches → interleave → merge with standard search → entity_thread → results
```

- **Bridge retrieval**: Extracts entity anchors from patterns like "between X and Y", "X or Y", "before X did Y"
- Runs separate hybrid searches per anchor
- Results are interleaved round-robin across anchors
- Merged with standard search results; bridge results get logarithmic boost: `ccf + 0.3 * log1p(ccf)`
- Timestamp tiebreaker: `relevance += timestamp * 1e-15`

### 4. Broad Summary Queries

```text
Query → Embed → hybrid_search(10x limit) → strided_retrieval → MMR_rerank → results
```

- Strided retrieval for timeline coverage (same as ordering)
- **MMR (Maximal Marginal Relevance)**: Greedy selection balancing relevance vs diversity
  - `λ=0.5` (balanced)
  - Uses word-level Jaccard similarity as inter-document similarity proxy (no stored embeddings in results)

## Temporal Re-ranking (`_temporal_rerank`)

Applied to standard path when temporal intent is detected.

1. **Resolve date range**: `resolve_temporal_query()` tries parsedatetime first, then month name extraction, then optional LLM fallback
2. **Compute anchor**: midpoint of resolved range
3. **For each result with a date**:
   - **Directional penalty**: If query says "after" but memory is before anchor (or vice versa) → 0.1x multiplier ("temporal cliff")
   - **Gaussian decay**: `1.0 + 0.5 * exp(-delta²/2σ²)` where σ=3 days
   - **Same-day grace**: delta < 0.5 days → 1.5x boost
   - **Tiebreaker**: `timestamp * 1e-15` for equal scores
4. Re-sort by modified relevance

## Date Extraction Pipeline

Where memory dates come from (checked in order by `_extract_memory_date`):

1. `metadata.dates[0]` — extracted at store time by `extract_dates()`
2. Content prefix: `[Date: 2024-01-15]` pattern
3. `created_at` timestamp (fallback)

## Temporal Query Resolution (`resolve_temporal_query`)

Three-tier resolution:

1. **parsedatetime** (free, no deps): Handles "last Tuesday", "3 weeks ago", "between X and Y"
2. **Month reference**: "in January", "last March" → full month range
3. **LLM fallback** (optional): Uses `TEMPORAL_LLM_MODEL` setting with litellm. Disabled by default

## Elastic K

Ordering, summary, and multi-hop queries automatically expand the fetch limit to 2x the requested limit. This gives broader coverage of scattered facts across the timeline before trimming.
