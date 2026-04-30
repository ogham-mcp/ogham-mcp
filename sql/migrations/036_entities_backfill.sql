-- Migration 036: entities + memory_entities backfill (v0.14)
--
-- WHY THIS MIGRATION EXISTS
-- The entities + memory_entities tables landed in v0.10's schema.sql but
-- were never delivered as a standalone migration. Older Supabase
-- deployments that took the migration path (rather than fresh-installing
-- from schema.sql at v0.10+) are missing these tables. Surfaced by Hotfix A
-- (v0.13.1) which added to_regclass() guards on density + suggest_connections
-- so the missing tables degrade gracefully -- this migration retrofits
-- them so older self-hosters get full entity-graph functionality.
--
-- WHAT IT DOES
-- 1. Create entities + memory_entities tables (IF NOT EXISTS -- no-op
--    on fresh installs that already have them from schema.sql).
-- 2. Add the b-tree indexes needed for entity lookup and graph walks.
-- 3. Apply RLS policies denying anon access (DROP IF EXISTS first since
--    CREATE POLICY has no IF NOT EXISTS form).
-- 4. Create the supporting RPC functions:
--      - refresh_entity_temporal_span(bigint)
--      - spread_entity_activation_memories(text[], text, int, float, float, int)
--    Both use CREATE OR REPLACE so re-running on an up-to-date install
--    just refreshes the function bodies.
--
-- WHAT IT DOES NOT DO
-- This migration is schema-only. After applying it, run the data backfill
-- script (`ogham backfill entities`) to populate memory_entities for
-- existing memory rows. The migration leaves the tables empty -- safe to
-- apply during traffic since reads short-circuit on empty memory_entities.

-- 1. Tables (idempotent)

CREATE TABLE IF NOT EXISTS entities (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    mention_count integer NOT NULL DEFAULT 0,
    temporal_span float NOT NULL DEFAULT 1.0,
    session_count integer NOT NULL DEFAULT 1,
    UNIQUE (canonical_name, entity_type)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    profile text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, entity_id)
);

-- 2. Indexes (idempotent)

CREATE INDEX IF NOT EXISTS idx_entities_type_name
    ON entities (entity_type, canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_type
    ON entities (canonical_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
    ON memory_entities (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_profile
    ON memory_entities (entity_id, profile);

-- 3. RLS policies (CREATE POLICY has no IF NOT EXISTS so DROP first)

ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Deny anon access" ON entities;
CREATE POLICY "Deny anon access" ON entities
    FOR ALL TO anon USING (false) WITH CHECK (false);

ALTER TABLE memory_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_entities FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Deny anon access" ON memory_entities;
CREATE POLICY "Deny anon access" ON memory_entities
    FOR ALL TO anon USING (false) WITH CHECK (false);

-- 4. Supporting RPC functions (CREATE OR REPLACE -- safe to re-apply)

-- link_memory_entities: upsert entities + link in memory_entities for one
-- memory. Both live writes (service.store_memory) and the backfill loop
-- call this so the two paths produce identical state. Idempotent on
-- memory_entities via ON CONFLICT DO NOTHING. Mention counts accumulate
-- on entities so the temporal-span refresher has data to work with.
-- Returns the number of (memory, entity) edges newly inserted.
CREATE OR REPLACE FUNCTION link_memory_entities(
    p_memory_id uuid,
    p_profile text,
    p_entity_tags text[]
) RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    inserted_count integer := 0;
BEGIN
    IF p_entity_tags IS NULL OR array_length(p_entity_tags, 1) IS NULL THEN
        RETURN 0;
    END IF;

    WITH parsed AS (
        SELECT split_part(t, ':', 1) AS et,
               split_part(t, ':', 2) AS cn
        FROM unnest(p_entity_tags) AS t
        WHERE t LIKE '%:%' AND length(split_part(t, ':', 2)) > 0
    ),
    entity_upsert AS (
        INSERT INTO entities (canonical_name, entity_type, mention_count)
        SELECT cn, et, 1 FROM parsed
        ON CONFLICT (canonical_name, entity_type) DO UPDATE
            SET mention_count = entities.mention_count + 1
        RETURNING id
    ),
    edge_insert AS (
        INSERT INTO memory_entities (memory_id, entity_id, profile)
        SELECT p_memory_id, eu.id, p_profile
        FROM entity_upsert eu
        ON CONFLICT (memory_id, entity_id) DO NOTHING
        RETURNING memory_id
    )
    SELECT count(*) INTO inserted_count FROM edge_insert;

    RETURN inserted_count;
END;
$$;

CREATE OR REPLACE FUNCTION refresh_entity_temporal_span(target_entity_id bigint)
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
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

CREATE OR REPLACE FUNCTION spread_entity_activation_memories(
    seed_entity_tags text[],
    filter_profile text,
    max_depth int DEFAULT 2,
    decay float DEFAULT 0.65,
    min_activation float DEFAULT 0.1,
    max_results int DEFAULT 50
) RETURNS TABLE (memory_id uuid, activation float)
LANGUAGE plpgsql STABLE
SECURITY DEFINER
SET search_path = public, extensions
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE
    seeds AS (
        SELECT DISTINCT e.id, e.temporal_span
        FROM entities e
        JOIN LATERAL unnest(seed_entity_tags) AS t ON true
        WHERE e.canonical_name = split_part(t, ':', 2)
          AND e.entity_type = split_part(t, ':', 1)
        LIMIT 6
    ),
    walk AS (
        SELECT s.id AS entity_id, 1.0::float AS activation, 0 AS depth
        FROM seeds s
        UNION ALL
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
    activated_entities AS (
        SELECT w2.entity_id, max(w2.activation) AS activation
        FROM walk w2
        GROUP BY w2.entity_id
    ),
    activated_memories AS (
        SELECT me.memory_id, max(ae.activation) AS activation
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
