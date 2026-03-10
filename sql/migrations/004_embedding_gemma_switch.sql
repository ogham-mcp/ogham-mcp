-- Migration: Switch from mxbai-embed-large (1024) to EmbeddingGemma (768)
-- Run re_embed_all() for each profile immediately after applying this migration.

-- ============================================================
-- Drop HNSW index (can't ALTER vector dimension with index present)
-- ============================================================
DROP INDEX IF EXISTS memories_embedding_idx;

-- ============================================================
-- Clear existing 1024-dim embeddings (incompatible with new dimension)
-- ============================================================
UPDATE memories SET embedding = NULL;

-- ============================================================
-- Change embedding column from 1024 to 768 dimensions
-- ============================================================
ALTER TABLE memories ALTER COLUMN embedding TYPE extensions.vector(768);

-- ============================================================
-- Recreate HNSW index for 768 dimensions
-- ============================================================
CREATE INDEX memories_embedding_idx
    ON memories USING hnsw (embedding extensions.vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Recreate match_memories with vector(768) parameter
-- Must drop first: parameter type changed
-- ============================================================
DROP FUNCTION IF EXISTS match_memories(extensions.vector, float, int, text[], text, text);

CREATE OR REPLACE FUNCTION match_memories(
    query_embedding extensions.vector(768),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL,
    filter_profile text DEFAULT 'default'
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    profile text,
    tags text[],
    similarity float,
    relevance float,
    access_count integer,
    last_accessed_at timestamptz,
    confidence float,
    created_at timestamptz,
    updated_at timestamptz
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.metadata,
        m.source,
        m.profile,
        m.tags,
        (1 - (m.embedding <=> query_embedding))::float AS similarity,
        (
            (1 - (m.embedding <=> query_embedding)) *
            ln(1.0 + exp(
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
    FROM public.memories m
    WHERE
        1 - (m.embedding <=> query_embedding) > match_threshold
        AND (filter_tags IS NULL OR m.tags && filter_tags)
        AND (filter_source IS NULL OR m.source = filter_source)
        AND m.profile = filter_profile
        AND (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY relevance DESC
    LIMIT match_count;
END;
$$;

-- ============================================================
-- Recreate batch_update_embeddings with vector(768) parameter
-- ============================================================
CREATE OR REPLACE FUNCTION batch_update_embeddings(
    memory_ids uuid[],
    new_embeddings extensions.vector(768)[]
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
    updated_count integer := 0;
    i integer;
BEGIN
    FOR i IN 1..array_length(memory_ids, 1) LOOP
        UPDATE memories
        SET embedding = new_embeddings[i], updated_at = now()
        WHERE id = memory_ids[i];
        updated_count := updated_count + 1;
    END LOOP;
    RETURN updated_count;
END;
$$;
