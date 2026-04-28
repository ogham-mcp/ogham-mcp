-- Migration 020: Entity graph activation support
-- Adds temporal_span for edge weighting + spreading activation function
-- Prerequisite: entities + memory_entities tables from migration 018

-- 1. Temporal span columns
ALTER TABLE entities ADD COLUMN IF NOT EXISTS temporal_span float NOT NULL DEFAULT 1.0;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS session_count integer NOT NULL DEFAULT 1;

-- 2. Index for entity tag lookup (resolving query tags to entity IDs)
CREATE INDEX IF NOT EXISTS idx_entities_canonical_type
    ON entities (canonical_name, entity_type);

-- 3. Composite index for the walk (should already exist from 018)
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_profile
    ON memory_entities (entity_id, profile);

-- 4. Refresh temporal span for a single entity
CREATE OR REPLACE FUNCTION refresh_entity_temporal_span(target_entity_id bigint)
RETURNS void LANGUAGE sql AS $$
    UPDATE entities SET
        session_count = sub.cnt,
        temporal_span = ln(1.0 + sub.cnt)
    FROM (
        SELECT COUNT(DISTINCT DATE_TRUNC('day', m.created_at)) AS cnt
        FROM memory_entities me
        JOIN memories m ON m.id = me.memory_id
        WHERE me.entity_id = target_entity_id
    ) sub
    WHERE id = target_entity_id;
$$;

-- 5. Spreading activation over the bipartite entity/memory graph.
--    Walks entity -> memory_entities -> memory -> memory_entities -> entity
--    weighted by temporal_span (bridge entities amplified, noise suppressed).
--    Returns (memory_id, activation) for the top-N activated memories.
CREATE OR REPLACE FUNCTION spread_entity_activation_memories(
    seed_entity_tags text[],
    filter_profile text,
    max_depth int DEFAULT 2,
    decay float DEFAULT 0.65,
    min_activation float DEFAULT 0.1,
    max_results int DEFAULT 50
) RETURNS TABLE (memory_id uuid, activation float)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE
    -- Resolve seed tags (format "type:name") to entity IDs
    seeds AS (
        SELECT DISTINCT e.id, e.temporal_span
        FROM entities e
        JOIN LATERAL unnest(seed_entity_tags) AS t ON true
        WHERE e.canonical_name = split_part(t, ':', 2)
          AND e.entity_type = split_part(t, ':', 1)
        LIMIT 6  -- seed count cap
    ),
    walk AS (
        -- Depth 0: seed entities with activation 1.0
        SELECT s.id AS entity_id,
               1.0::float AS activation,
               0 AS depth
        FROM seeds s

        UNION ALL

        -- Walk: entity -> memory_entities -> memory -> memory_entities -> entity
        -- temporal_span used directly as relative boost (0.69 single-session, 2.4+ bridge)
        -- Super-node damping: 1/ln(1+mention_count) suppresses noise entities
        SELECT e2.id AS entity_id,
               LEAST(1.0,
                 w.activation * decay
                 * LEAST(e2.temporal_span, 3.0)
                 * (1.0 / ln(1.0 + GREATEST(e2.mention_count, 1)))
               )::float AS activation,
               w.depth + 1 AS depth
        FROM walk w
        JOIN memory_entities me1 ON me1.entity_id = w.entity_id
                                AND me1.profile = filter_profile
        JOIN memory_entities me2 ON me2.memory_id = me1.memory_id
                                AND me2.entity_id != w.entity_id
                                AND me2.profile = filter_profile
        JOIN entities e2 ON e2.id = me2.entity_id
        WHERE w.depth < max_depth
          AND w.activation * decay
              * LEAST(e2.temporal_span, 3.0)
              * (1.0 / ln(1.0 + GREATEST(e2.mention_count, 1)))
              > min_activation
    ),
    -- Aggregate per entity: MAX activation (not SUM, prevents triangular over-boost)
    activated_entities AS (
        SELECT w2.entity_id, max(w2.activation) AS activation
        FROM walk w2
        GROUP BY w2.entity_id
    ),
    -- Map activated entities back to memories
    activated_memories AS (
        SELECT me.memory_id,
               max(ae.activation) AS activation
        FROM activated_entities ae
        JOIN memory_entities me ON me.entity_id = ae.entity_id
                               AND me.profile = filter_profile
        GROUP BY me.memory_id
    )
    SELECT am.memory_id, am.activation
    FROM activated_memories am
    ORDER BY am.activation DESC
    LIMIT max_results;
END;
$$;
