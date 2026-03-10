-- RPC: traverse relationship graph from a known memory ID
CREATE OR REPLACE FUNCTION get_related_memories(
    start_id uuid,
    max_depth int DEFAULT 1,
    min_strength float DEFAULT 0.5,
    filter_types relationship_type[] DEFAULT NULL,
    result_limit int DEFAULT 20
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    tags text[],
    confidence float,
    depth int,
    relationship text,
    edge_strength float,
    connected_from uuid
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE graph AS (
        SELECT start_id AS id, 0 AS depth,
               NULL::relationship_type AS rel,
               NULL::float AS edge_strength,
               NULL::uuid AS connected_from
        UNION ALL
        SELECT
            CASE WHEN mr.source_id = g.id THEN mr.target_id ELSE mr.source_id END,
            g.depth + 1,
            mr.relationship,
            mr.strength,
            g.id
        FROM graph g
        JOIN memory_relationships mr
            ON (mr.source_id = g.id OR mr.target_id = g.id)
        WHERE g.depth < max_depth
          AND mr.strength >= min_strength
          AND (filter_types IS NULL OR mr.relationship = ANY(filter_types))
    ),
    deduped AS (
        SELECT DISTINCT ON (g.id) g.*
        FROM graph g
        WHERE g.id != start_id
        ORDER BY g.id, g.depth ASC, g.edge_strength DESC NULLS LAST
    )
    SELECT
        m.id, m.content, m.metadata, m.source, m.tags, m.confidence,
        d.depth, d.rel::text, d.edge_strength, d.connected_from
    FROM deduped d
    JOIN memories m ON m.id = d.id
    WHERE (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY d.depth ASC, d.edge_strength DESC
    LIMIT result_limit;
END;
$$;
