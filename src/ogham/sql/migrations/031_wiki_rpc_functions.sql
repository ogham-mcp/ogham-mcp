-- Migration 031: Wiki Tier 1 RPC functions for SupabaseBackend support.
--
-- Background
-- ----------
-- The Wiki Tier 1 layer (compile_wiki, query_topic_summary, walk_knowledge,
-- lint_wiki, plus the search_summaries preamble injection) was built around
-- backend._execute() which only PostgresBackend implements. SupabaseBackend
-- routes through PostgREST and only supports rpc() calls into pre-registered
-- functions. So the entire wiki layer broke on the most common self-hoster
-- backend until 031.
--
-- This migration creates the SQL functions PostgREST needs. Each function
-- mirrors a single SQL operation the wiki Python layer used to issue inline
-- via _execute. SECURITY INVOKER means callers run as themselves; no
-- privilege escalation. SET search_path = public, extensions, pg_catalog locks the
-- function lookup path (matches migration 029).
--
-- Layer mapping:
--   topic_summaries.search_summaries          -> wiki_topic_search
--   topic_summaries.upsert_summary            -> wiki_topic_upsert
--   topic_summaries.get_summary_by_topic      -> wiki_topic_get_by_key
--   topic_summaries.get_affected_summaries... -> wiki_topic_get_affected
--   topic_summaries.mark_stale                -> wiki_topic_mark_stale
--   topic_summaries.sweep_stale_summaries     -> wiki_topic_sweep_stale
--   topic_summaries.list_stale                -> wiki_topic_list_stale
--   recompute (source-id fetch)               -> wiki_recompute_get_source_ids
--   recompute (source-content fetch)          -> wiki_recompute_get_source_content
--   database.walk_memory_graph                -> wiki_walk_graph
--   wiki_lint.find_contradictions             -> wiki_lint_contradictions
--   wiki_lint.find_orphans                    -> wiki_lint_orphans
--   wiki_lint.find_stale_lifecycle            -> wiki_lint_stale_lifecycle
--   wiki_lint.find_summary_drift (helper)     -> wiki_topic_list_fresh_for_drift
--
-- Idempotent: every function uses CREATE OR REPLACE.

BEGIN;

-- Make `vector` etc. resolvable at function-creation time. Each function
-- already pins its execution-time search_path, but the parser needs the
-- types in scope when it compiles the bodies. On Supabase pgvector lives
-- in `extensions`, not `public`; on vanilla self-hosters it's typically
-- in `public`. Listing both keeps creation portable.
SET LOCAL search_path = public, extensions, pg_catalog;

-- 1. Vector search over fresh topic summaries.
CREATE OR REPLACE FUNCTION wiki_topic_search(
    p_profile text,
    p_query_embedding vector,
    p_top_k integer DEFAULT 3,
    p_min_similarity float DEFAULT 0.0
)
RETURNS TABLE (
    id uuid,
    topic_key text,
    profile_id text,
    content text,
    source_count integer,
    source_cursor uuid,
    source_hash bytea,
    model_used text,
    version integer,
    status text,
    updated_at timestamptz,
    similarity float
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    -- HNSW + threshold trap: combining `WHERE similarity >= threshold`
    -- with `ORDER BY <=> ... LIMIT k` defeats the index when the
    -- threshold filters out top-k. Postgres falls back to scanning the
    -- HNSW tail row-by-row. Wrap the index-driven top-k in a CTE,
    -- apply the threshold AFTER. The index path then runs unfiltered
    -- and the threshold trims the (already small) output.
    WITH top_k AS (
        SELECT id, topic_key, profile_id, content, source_count,
               source_cursor, source_hash, model_used, version, status,
               updated_at,
               1 - (embedding <=> p_query_embedding) AS similarity
          FROM topic_summaries
         WHERE profile_id = p_profile
           AND status = 'fresh'
           AND embedding IS NOT NULL
         ORDER BY embedding <=> p_query_embedding
         LIMIT p_top_k
    )
    SELECT * FROM top_k WHERE similarity >= p_min_similarity;
$$;

-- 2. Atomic upsert of a topic summary + its source-junction rows.
--    Mirrors the CTE pattern from topic_summaries.upsert_summary.
CREATE OR REPLACE FUNCTION wiki_topic_upsert(
    p_profile text,
    p_topic_key text,
    p_content text,
    p_embedding vector,
    p_source_memory_ids uuid[],
    p_model_used text,
    p_source_cursor uuid,
    p_source_hash bytea,
    p_token_count integer DEFAULT NULL,
    p_importance float DEFAULT 0.5
)
RETURNS topic_summaries
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
DECLARE
    upserted topic_summaries;
BEGIN
    -- cardinality() returns 0 for empty arrays; array_length() returns NULL,
    -- which would crash the NOT NULL source_count constraint.
    INSERT INTO topic_summaries (
        topic_key, profile_id, content, embedding,
        source_count, source_cursor, source_hash,
        token_count, importance, model_used
    )
    VALUES (
        p_topic_key, p_profile, p_content, p_embedding,
        cardinality(p_source_memory_ids), p_source_cursor, p_source_hash,
        p_token_count, p_importance, p_model_used
    )
    ON CONFLICT (profile_id, topic_key) DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        source_count = EXCLUDED.source_count,
        source_cursor = EXCLUDED.source_cursor,
        source_hash = EXCLUDED.source_hash,
        token_count = EXCLUDED.token_count,
        importance = EXCLUDED.importance,
        model_used = EXCLUDED.model_used,
        version = topic_summaries.version + 1,
        status = 'fresh',
        stale_reason = NULL
    RETURNING * INTO upserted;

    -- Concurrent-delete safety: if another transaction deleted the topic
    -- between our row-lock release and the RETURNING, INSERT...DO UPDATE
    -- can yield zero rows. Bail rather than crash on the FK insert.
    IF upserted.id IS NULL THEN
        RETURN NULL;
    END IF;

    DELETE FROM topic_summary_sources WHERE summary_id = upserted.id;

    -- JOIN against memories so concurrently-deleted memory ids drop
    -- silently instead of throwing a FK violation. Wiki content is a
    -- best-effort snapshot; missing one source is preferable to
    -- failing the whole upsert.
    INSERT INTO topic_summary_sources (summary_id, memory_id)
    SELECT upserted.id, m.id
      FROM unnest(p_source_memory_ids) AS t(id)
      JOIN memories m ON m.id = t.id
    ON CONFLICT DO NOTHING;

    RETURN upserted;
END;
$$;

-- 3. Fetch live summary for a (profile, topic_key) pair.
CREATE OR REPLACE FUNCTION wiki_topic_get_by_key(
    p_profile text,
    p_topic_key text
)
RETURNS SETOF topic_summaries
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT * FROM topic_summaries
     WHERE profile_id = p_profile AND topic_key = p_topic_key
     LIMIT 1;
$$;

-- 4. Reverse cascade: every summary that cites a given memory id.
--    Hot path for the Phase 6 hooks on memory mutations.
CREATE OR REPLACE FUNCTION wiki_topic_get_affected(
    p_memory_id uuid
)
RETURNS TABLE (
    id uuid,
    profile_id text,
    topic_key text,
    status text,
    version integer
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT ts.id, ts.profile_id, ts.topic_key, ts.status, ts.version
      FROM topic_summary_sources tss
      JOIN topic_summaries ts ON ts.id = tss.summary_id
     WHERE tss.memory_id = p_memory_id;
$$;

-- 5. Mark a summary stale (with optional reason).
CREATE OR REPLACE FUNCTION wiki_topic_mark_stale(
    p_summary_id uuid,
    p_reason text DEFAULT NULL
)
RETURNS void
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    UPDATE topic_summaries
       SET status = 'stale', stale_reason = p_reason
     WHERE id = p_summary_id;
$$;

-- 6. Sweep fresh summaries idle past N days into 'stale'.
CREATE OR REPLACE FUNCTION wiki_topic_sweep_stale(
    p_profile text,
    p_older_than_days integer DEFAULT 30
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
DECLARE
    n integer;
BEGIN
    WITH updated AS (
        UPDATE topic_summaries
           SET status = 'stale',
               stale_reason = 'nightly sweep: idle past threshold'
         WHERE profile_id = p_profile
           AND status = 'fresh'
           AND updated_at < now() - make_interval(days => p_older_than_days)
         RETURNING id
    )
    SELECT count(*) INTO n FROM updated;
    RETURN n;
END;
$$;

-- 7. List stale summaries (optionally scoped by profile + age).
CREATE OR REPLACE FUNCTION wiki_topic_list_stale(
    p_profile text DEFAULT NULL,
    p_older_than_days integer DEFAULT NULL
)
RETURNS SETOF topic_summaries
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT * FROM topic_summaries
     WHERE status = 'stale'
       AND (p_profile IS NULL OR profile_id = p_profile)
       AND (p_older_than_days IS NULL
            OR updated_at < now() - make_interval(days => p_older_than_days));
$$;

-- 8. List fresh summaries with their stored source_hash (drift candidates).
CREATE OR REPLACE FUNCTION wiki_topic_list_fresh_for_drift(
    p_profile text
)
RETURNS TABLE (id uuid, topic_key text, source_hash bytea)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT id, topic_key, source_hash
      FROM topic_summaries
     WHERE profile_id = p_profile
       AND status = 'fresh';
$$;

-- 9. Recompute helper: ordered list of memory ids carrying a tag.
CREATE OR REPLACE FUNCTION wiki_recompute_get_source_ids(
    p_profile text,
    p_tag text
)
RETURNS TABLE (id text)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT id::text
      FROM memories
     WHERE profile = p_profile
       AND p_tag = ANY(tags)
       AND (expires_at IS NULL OR expires_at > now())
     ORDER BY id;
$$;

-- 10. Recompute helper: id + content for a list of memories (prompt build).
CREATE OR REPLACE FUNCTION wiki_recompute_get_source_content(
    p_memory_ids uuid[]
)
RETURNS TABLE (id text, content text)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    SELECT id::text, content
      FROM memories
     WHERE id = ANY(p_memory_ids)
     ORDER BY id;
$$;

-- 11. Direction-aware graph walk over memory_relationships (recursive CTE).
--     Mirrors database.walk_memory_graph with all five knobs as parameters.
CREATE OR REPLACE FUNCTION wiki_walk_graph(
    p_start_id uuid,
    p_max_depth integer DEFAULT 1,
    p_direction text DEFAULT 'both',
    p_min_strength float DEFAULT 0.0,
    p_relationship_types text[] DEFAULT NULL,
    p_result_limit integer DEFAULT 50
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    tags text[],
    confidence float,
    depth integer,
    relationship text,
    edge_strength float,
    connected_from uuid,
    direction_used text
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
BEGIN
    IF p_direction NOT IN ('outgoing', 'incoming', 'both') THEN
        RAISE EXCEPTION 'direction must be outgoing/incoming/both, got %', p_direction;
    END IF;
    IF p_max_depth < 0 OR p_max_depth > 5 THEN
        RAISE EXCEPTION 'depth must be 0..5, got %', p_max_depth;
    END IF;

    RETURN QUERY
    -- Track the path so cycles (A->B->A) and diamond patterns
    -- (A->B->C, A->D->C) don't blow the recursion size. Without
    -- this, dense graphs at depth=5 generate orders of magnitude
    -- more rows than DISTINCT ON ultimately keeps.
    WITH RECURSIVE graph AS (
        SELECT p_start_id AS id, 0 AS depth,
               NULL::relationship_type AS rel,
               NULL::float AS edge_strength,
               NULL::uuid AS connected_from,
               NULL::text AS direction_used,
               ARRAY[p_start_id] AS visited
        UNION ALL
        SELECT
            next_id.v,
            g.depth + 1,
            mr.relationship,
            mr.strength,
            g.id,
            CASE
                WHEN mr.source_id = g.id THEN 'outgoing'
                ELSE 'incoming'
            END,
            g.visited || next_id.v
        FROM graph g
        JOIN memory_relationships mr
          ON CASE
                WHEN p_direction = 'outgoing' THEN mr.source_id = g.id
                WHEN p_direction = 'incoming' THEN mr.target_id = g.id
                ELSE (mr.source_id = g.id OR mr.target_id = g.id)
             END
        CROSS JOIN LATERAL (
            -- Materialise the next id once so the cycle filter and the
            -- SELECT projection see the same value without restating the
            -- direction CASE three times.
            SELECT CASE
                WHEN p_direction = 'outgoing' THEN mr.target_id
                WHEN p_direction = 'incoming' THEN mr.source_id
                WHEN mr.source_id = g.id THEN mr.target_id
                ELSE mr.source_id
            END AS v
        ) next_id
        WHERE g.depth < p_max_depth
          AND mr.strength >= p_min_strength
          AND (p_relationship_types IS NULL
               OR mr.relationship::text = ANY(p_relationship_types))
          AND NOT (next_id.v = ANY(g.visited))
    )
    SELECT
        m.id, m.content, m.metadata, m.source, m.tags, m.confidence,
        deduped.depth, deduped.rel::text, deduped.edge_strength,
        deduped.connected_from, deduped.direction_used
    FROM (
        -- Alias the CTE to `g` and qualify every column with the alias.
        -- Two reasons: (1) the function's RETURNS TABLE(id, depth, ...)
        -- declares OUT parameters with the same names, and PG17 raises
        -- AmbiguousColumn when bare `id` is used inside the body;
        -- (2) qualified `graph.id` references work in scratch PG17 but
        -- the Supabase PG build rejects "relation graph does not exist"
        -- on parse, so use an alias instead of the CTE name directly.
        SELECT DISTINCT ON (g.id)
               g.id, g.depth, g.rel,
               g.edge_strength, g.connected_from, g.direction_used
          FROM graph g
         WHERE g.depth > 0
         ORDER BY g.id, g.depth ASC
    ) deduped
    JOIN memories m ON m.id = deduped.id
    ORDER BY deduped.depth ASC, deduped.edge_strength DESC NULLS LAST
    LIMIT p_result_limit;
END;
$$;

-- 12. Lint: count + sample of contradicts edges.
CREATE OR REPLACE FUNCTION wiki_lint_contradictions(
    p_profile text,
    p_sample_size integer DEFAULT 10
)
RETURNS TABLE (
    source_id text,
    target_id text,
    strength float,
    created_at timestamptz,
    total_count bigint
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    WITH all_pairs AS (
        SELECT mr.source_id, mr.target_id, mr.strength, mr.created_at
          FROM memory_relationships mr
          JOIN memories m ON m.id = mr.source_id
         WHERE mr.relationship = 'contradicts'
           AND m.profile = p_profile
    )
    SELECT mr.source_id::text, mr.target_id::text, mr.strength, mr.created_at,
           (SELECT count(*) FROM all_pairs)
      FROM all_pairs mr
     ORDER BY mr.created_at DESC
     LIMIT p_sample_size;
$$;

-- 13. Lint: count + sample of orphan memories (no edges, past grace window).
CREATE OR REPLACE FUNCTION wiki_lint_orphans(
    p_profile text,
    p_sample_size integer DEFAULT 10,
    p_grace_minutes integer DEFAULT 5
)
RETURNS TABLE (
    id text,
    content text,
    tags text[],
    created_at timestamptz,
    total_count bigint
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    -- LEFT JOIN ... ON (source_id = m.id OR target_id = m.id) defeats the
    -- per-column indexes on memory_relationships and forces a sequential
    -- scan of the edge table. Two NOT EXISTS subqueries each use an index
    -- cleanly. Critical for profiles with thousands of memories.
    WITH orphans AS (
        SELECT m.id, m.content, m.tags, m.created_at
          FROM memories m
         WHERE m.profile = p_profile
           AND m.created_at < now() - make_interval(mins => p_grace_minutes)
           AND (m.expires_at IS NULL OR m.expires_at > now())
           AND NOT EXISTS (
               SELECT 1 FROM memory_relationships mr
                WHERE mr.source_id = m.id
           )
           AND NOT EXISTS (
               SELECT 1 FROM memory_relationships mr
                WHERE mr.target_id = m.id
           )
    )
    SELECT id::text, content, tags, created_at,
           (SELECT count(*) FROM orphans)
      FROM orphans
     ORDER BY created_at DESC
     LIMIT p_sample_size;
$$;

-- 14. Lint: count + sample of memories stuck in 'stable' lifecycle past N days.
CREATE OR REPLACE FUNCTION wiki_lint_stale_lifecycle(
    p_profile text,
    p_older_than_days integer DEFAULT 90,
    p_sample_size integer DEFAULT 10
)
RETURNS TABLE (
    id text,
    stage text,
    stage_entered_at timestamptz,
    content text,
    total_count bigint
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = public, extensions, pg_catalog
AS $$
    WITH stale AS (
        SELECT ml.memory_id, ml.stage, ml.stage_entered_at, m.content
          FROM memory_lifecycle ml
          JOIN memories m ON m.id = ml.memory_id
         WHERE ml.profile = p_profile
           AND ml.stage = 'stable'
           AND ml.stage_entered_at < now() - make_interval(days => p_older_than_days)
    )
    SELECT memory_id::text, stage, stage_entered_at, content,
           (SELECT count(*) FROM stale)
      FROM stale
     ORDER BY stage_entered_at ASC
     LIMIT p_sample_size;
$$;

COMMIT;
