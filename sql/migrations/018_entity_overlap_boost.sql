-- Migration 018: query-conditioned entity overlap boost in hybrid_search_memories
--
-- v0.9.2 Stage 1 (Part A) of the entity-graph spreading activation work.
-- Plan: docs/plans/2026-04-06-entity-graph-spreading-activation.md
--
-- WHAT THIS DOES
-- Adds a new optional parameter `query_entity_tags text[]` to hybrid_search_memories.
-- When non-null, the relevance formula gets a multiplicative boost between 1.0 and 1.3
-- based on how many of the query's entity tags appear in each memory's tags column.
--
-- WHY
-- BEAM 100K shows that retrieval is weak on multi_session_reasoning (R@10 = 0.645) and
-- knowledge_update (QA = 0.425). Both categories need to find memories that mention the
-- same entity as the query, even when the query phrasing differs from the original.
-- Vector search captures this partially via dense embedding similarity; this boost adds
-- an explicit, discrete entity-match signal on top.
--
-- HOW IT'S SAFE (this is NOT the failed graph_depth=1 attempt from 2026-04-05)
-- - It's a re-rank within the existing top-K, NOT a candidate-generation step
-- - Bounded multiplicatively: factor is between 1.0 and 1.3, can't catastrophically displace
-- - No-op when query has no entity tags (factor = 1.0)
-- - No-op when memory has no overlapping tags (factor = 1.0)
--
-- EXTRACTING QUERY ENTITY TAGS
-- The Python caller extracts entities from the query string using the existing
-- ogham.extraction.extract_entities() function and passes the result as text[].
-- Tags use the same prefix scheme as memory tags: person:John, event:wedding,
-- preference:prefer, location:Paris, etc.
--
-- Idempotent: CREATE OR REPLACE means re-running this migration is safe.
--
-- POSTGRES OVERLOADING GOTCHA
-- CREATE OR REPLACE FUNCTION only replaces a function with the EXACT same parameter list.
-- Adding a new parameter creates a new overload alongside the old one. To avoid having
-- multiple versions live in the database (which Python callers would hit non-deterministically),
-- we explicitly DROP the previous signatures first. Both the 9-arg (pre-multi-profile) and
-- the 10-arg (with filter_profiles) versions are removed.

DROP FUNCTION IF EXISTS hybrid_search_memories(
    text, vector, integer, text, text[], text, float, float, integer
);
DROP FUNCTION IF EXISTS hybrid_search_memories(
    text, vector, integer, text, text[], text, float, float, integer, text[]
);

CREATE OR REPLACE FUNCTION hybrid_search_memories(
    query_text text,
    query_embedding vector,
    match_count integer DEFAULT 10,
    filter_profile text DEFAULT 'default',
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL,
    full_text_weight float DEFAULT 0.3,
    semantic_weight float DEFAULT 0.7,
    rrf_k integer DEFAULT 10,
    filter_profiles text[] DEFAULT NULL,
    query_entity_tags text[] DEFAULT NULL
)
RETURNS TABLE(
    id uuid, content text, metadata jsonb, source text, profile text, tags text[],
    similarity float, keyword_rank float, relevance float,
    access_count integer, last_accessed_at timestamptz, confidence float,
    created_at timestamptz, updated_at timestamptz
)
LANGUAGE sql
SET search_path = public, extensions
AS $function$
with semantic as (
    select
        m.id,
        (1 - (m.embedding::halfvec(512) <=> query_embedding::halfvec(512)))::float as similarity,
        row_number() over (order by m.embedding::halfvec(512) <=> query_embedding::halfvec(512)) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by m.embedding::halfvec(512) <=> query_embedding::halfvec(512)
    limit match_count * 3
),
keyword as (
    select
        m.id,
        ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34)::float as keyword_rank,
        row_number() over (order by ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34) desc) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and m.fts @@ websearch_to_tsquery(query_text)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by keyword_rank desc
    limit match_count * 3
),
fused as (
    select
        coalesce(s.id, k.id) as id,
        coalesce(s.similarity, 0.0) as similarity,
        coalesce(k.keyword_rank, 0.0) as keyword_rank,
        -- Reciprocal Rank Fusion: position-based, score-agnostic
        (
            semantic_weight * (1.0 / (rrf_k + coalesce(s.rank_ix, match_count * 3)))
            + full_text_weight * (1.0 / (rrf_k + coalesce(k.rank_ix, match_count * 3)))
        ) as score
    from semantic s
    full outer join keyword k on s.id = k.id
)
select
    m.id, m.content, m.metadata, m.source, m.profile, m.tags,
    f.similarity, f.keyword_rank,
    (
        f.score
        * (1.0 + ln(m.access_count + 1.0) * 0.1)
        * m.confidence
        * (1.0 + g.graph_boost * 0.2)
        -- Entity overlap boost (v0.9.2): multiplicative factor in [1.0, 1.3]
        -- 1.0 when query has no entity tags or no overlap
        -- 1.0 + 0.3 * (overlap / total query tags) otherwise
        * case
            when query_entity_tags is null or cardinality(query_entity_tags) = 0 then 1.0
            when not (m.tags && query_entity_tags) then 1.0
            else 1.0 + 0.3 * (
                cardinality(array(
                    select unnest(m.tags) intersect select unnest(query_entity_tags)
                ))::float
                / cardinality(query_entity_tags)::float
            )
          end
    )::float as relevance,
    m.access_count, m.last_accessed_at, m.confidence, m.created_at, m.updated_at
from fused f
join memories m on m.id = f.id
left join lateral (
    select coalesce(sum(r.strength), 0.0) as graph_boost
    from memory_relationships r
    where r.target_id = m.id or r.source_id = m.id
) g on true
order by relevance desc
limit match_count;
$function$;
