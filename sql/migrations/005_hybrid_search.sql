-- Migration: Add hybrid search (tsvector full-text + pgvector semantic + RRF fusion)
-- Adds a generated tsvector column, GIN index, and hybrid_search_memories RPC.

-- ============================================================
-- Add generated tsvector column for full-text search
-- ============================================================
ALTER TABLE memories ADD COLUMN fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- ============================================================
-- GIN index on the tsvector column
-- ============================================================
CREATE INDEX memories_fts_idx ON memories USING gin (fts);

-- ============================================================
-- Hybrid search RPC: combines semantic (pgvector) and keyword
-- (tsvector) results using Reciprocal Rank Fusion (RRF)
-- ============================================================
CREATE OR REPLACE FUNCTION hybrid_search_memories(
    query_text text,
    query_embedding extensions.vector(768),
    match_count int DEFAULT 10,
    filter_profile text DEFAULT 'default',
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL,
    full_text_weight float DEFAULT 1.0,
    semantic_weight float DEFAULT 1.0,
    rrf_k int DEFAULT 60
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    profile text,
    tags text[],
    similarity float,
    keyword_rank float,
    relevance float,
    access_count integer,
    last_accessed_at timestamptz,
    confidence float,
    created_at timestamptz,
    updated_at timestamptz
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
WITH semantic AS (
    SELECT
        m.id,
        row_number() OVER (ORDER BY m.embedding <=> query_embedding) AS rank_ix,
        (1 - (m.embedding <=> query_embedding))::float AS similarity
    FROM memories m
    WHERE m.profile = filter_profile
      AND (filter_tags IS NULL OR m.tags && filter_tags)
      AND (filter_source IS NULL OR m.source = filter_source)
      AND (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count * 2
),
keyword AS (
    SELECT
        m.id,
        row_number() OVER (ORDER BY ts_rank_cd(m.fts, websearch_to_tsquery(query_text)) DESC) AS rank_ix,
        ts_rank_cd(m.fts, websearch_to_tsquery(query_text))::float AS keyword_rank
    FROM memories m
    WHERE m.profile = filter_profile
      AND m.fts @@ websearch_to_tsquery(query_text)
      AND (filter_tags IS NULL OR m.tags && filter_tags)
      AND (filter_source IS NULL OR m.source = filter_source)
      AND (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY keyword_rank DESC
    LIMIT match_count * 2
),
combined AS (
    SELECT id, rank_ix, similarity, 0.0::float AS keyword_rank, 'semantic' AS src
    FROM semantic
    UNION ALL
    SELECT id, rank_ix, 0.0::float AS similarity, keyword_rank, 'keyword' AS src
    FROM keyword
),
fused AS (
    SELECT
        c.id,
        MAX(c.similarity) AS similarity,
        MAX(c.keyword_rank) AS keyword_rank,
        SUM(
            CASE WHEN c.src = 'semantic' THEN semantic_weight / (rrf_k + c.rank_ix)
                 WHEN c.src = 'keyword'  THEN full_text_weight / (rrf_k + c.rank_ix)
                 ELSE 0.0 END
        ) AS rrf_score
    FROM combined c
    GROUP BY c.id
)
SELECT
    m.id,
    m.content,
    m.metadata,
    m.source,
    m.profile,
    m.tags,
    f.similarity,
    f.keyword_rank,
    (
        f.rrf_score
        * ln(1.0 + exp(
            ln(m.access_count + 1.0) -
            0.5 * ln(
                greatest(
                    extract(epoch from now() - coalesce(m.last_accessed_at, m.created_at)) / 86400.0,
                    0.01
                ) / (m.access_count + 1.0)
            )
        ))
        * m.confidence
    )::float AS relevance,
    m.access_count,
    m.last_accessed_at,
    m.confidence,
    m.created_at,
    m.updated_at
FROM fused f
JOIN memories m ON m.id = f.id
ORDER BY relevance DESC
LIMIT match_count;
$$;
