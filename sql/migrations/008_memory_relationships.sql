-- Migration: Add memory_relationships table for knowledge graph edges
-- Run in Supabase SQL Editor

-- Relationship type enum
CREATE TYPE relationship_type AS ENUM (
    'similar',
    'related',
    'contradicts',
    'supports',
    'follows',
    'derived_from'
);

-- Edge table
CREATE TABLE memory_relationships (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship relationship_type NOT NULL,
    strength float NOT NULL DEFAULT 1.0 CHECK (strength >= 0.0 AND strength <= 1.0),
    metadata jsonb DEFAULT '{}'::jsonb,
    created_by text NOT NULL DEFAULT 'auto',
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT unique_relationship UNIQUE (source_id, target_id, relationship)
);

-- RLS
ALTER TABLE memory_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_relationships FORCE ROW LEVEL SECURITY;

CREATE POLICY "Deny anon access" ON memory_relationships
    FOR ALL TO anon USING (false) WITH CHECK (false);

-- Indexes: FK columns with composite for common query patterns
CREATE INDEX idx_relationships_source
    ON memory_relationships (source_id, relationship);

CREATE INDEX idx_relationships_target
    ON memory_relationships (target_id, relationship);

-- Partial index: auto-linked edges for maintenance queries
CREATE INDEX idx_relationships_auto
    ON memory_relationships (created_at)
    WHERE created_by = 'auto';

-- RPC: auto-link a new memory to similar existing memories
CREATE OR REPLACE FUNCTION auto_link_memory(
    new_memory_id uuid,
    new_embedding extensions.vector(768),
    link_threshold float DEFAULT 0.85,
    max_links int DEFAULT 5,
    filter_profile text DEFAULT 'default'
)
RETURNS integer
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
    WITH candidates AS (
        SELECT m.id, (1 - (m.embedding <=> new_embedding))::float AS similarity
        FROM memories m
        WHERE m.id != new_memory_id
          AND m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND 1 - (m.embedding <=> new_embedding) > link_threshold
        ORDER BY m.embedding <=> new_embedding
        LIMIT max_links
    ),
    inserted AS (
        INSERT INTO memory_relationships (source_id, target_id, relationship, strength, created_by)
        SELECT new_memory_id, c.id, 'similar', c.similarity, 'auto'
        FROM candidates c
        ON CONFLICT (source_id, target_id, relationship) DO NOTHING
        RETURNING 1
    )
    SELECT count(*)::integer FROM inserted;
$$;

-- RPC: bulk backfill auto-links for memories that have no outgoing auto edges
CREATE OR REPLACE FUNCTION link_unlinked_memories(
    filter_profile text DEFAULT 'default',
    link_threshold float DEFAULT 0.85,
    max_links int DEFAULT 5,
    batch_size int DEFAULT 100
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
    processed integer := 0;
    links integer;
    mem record;
BEGIN
    FOR mem IN
        SELECT m.id, m.embedding
        FROM memories m
        WHERE m.profile = filter_profile
          AND m.embedding IS NOT NULL
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND NOT EXISTS (
              SELECT 1 FROM memory_relationships mr
              WHERE mr.source_id = m.id AND mr.created_by = 'auto'
          )
        LIMIT batch_size
    LOOP
        SELECT auto_link_memory(mem.id, mem.embedding, link_threshold, max_links, filter_profile) INTO links;
        IF links > 0 THEN
            processed := processed + 1;
        END IF;
    END LOOP;
    RETURN processed;
END;
$$;
