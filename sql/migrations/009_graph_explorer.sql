-- RPC: explore knowledge graph — hybrid search seeds + relationship traversal
CREATE OR REPLACE FUNCTION explore_memory_graph(
    query_text text,
    query_embedding extensions.vector(768),
    filter_profile text DEFAULT 'default',
    match_count int DEFAULT 5,
    traversal_depth int DEFAULT 1,
    min_strength float DEFAULT 0.5,
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    tags text[],
    relevance float,
    depth int,
    relationship text,
    edge_strength float,
    connected_from uuid
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE
    seeds AS (
        SELECT h.id, h.relevance
        FROM hybrid_search_memories(
            query_text, query_embedding, match_count,
            filter_profile, filter_tags, filter_source
        ) h
    ),
    graph AS (
        SELECT s.id, 0 AS depth, NULL::relationship_type AS rel,
               NULL::float AS edge_strength, NULL::uuid AS connected_from,
               s.relevance
        FROM seeds s
        UNION ALL
        SELECT
            CASE WHEN mr.source_id = g.id THEN mr.target_id ELSE mr.source_id END,
            g.depth + 1,
            mr.relationship,
            mr.strength,
            g.id,
            (g.relevance * mr.strength)::float
        FROM graph g
        JOIN memory_relationships mr
            ON (mr.source_id = g.id OR mr.target_id = g.id)
        WHERE g.depth < traversal_depth
          AND mr.strength >= min_strength
    ),
    deduped AS (
        SELECT DISTINCT ON (g.id)
            g.id, g.depth, g.rel, g.edge_strength, g.connected_from, g.relevance
        FROM graph g
        ORDER BY g.id, g.relevance DESC
    )
    SELECT
        m.id, m.content, m.metadata, m.source, m.tags,
        d.relevance::float,
        d.depth,
        d.rel::text AS relationship,
        d.edge_strength,
        d.connected_from
    FROM deduped d
    JOIN memories m ON m.id = d.id
    WHERE (m.expires_at IS NULL OR m.expires_at > now())
    ORDER BY d.depth ASC, d.relevance DESC;
END;
$$;
