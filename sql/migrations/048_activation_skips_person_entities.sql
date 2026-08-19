-- Spreading activation must not traverse person entities (TBU-251).
--
-- `9277a0f` stopped `person:` tags reaching the entities table on NEW writes.
-- It did nothing about rows already stored, and this function had no type
-- filter, so on every store ingested before that commit the graph walk still
-- expands through junk.
--
-- The SEEDS were already safe: `e.entity_type = split_part(t, ':', 1)` cannot
-- match a person seed, because `service.py` filters `query_entity_tags`. The
-- `walk` CTE was not. A clean seed reaches a memory and then hops through that
-- memory's person entities via `JOIN entities e2` -- unfiltered, to depth 2.
--
-- Measured on the BEAM corpus (2,866 memories) 2026-08-18:
--
--     entities table              17,776 rows, 16,651 person = 93.7%
--     entities per memory         12.2 avg, of which 6.6 are person
--     person entities shared
--       across 2+ memories        2,499 entities / 13,189 links
--
-- THIS CHANGE IS JUSTIFIED ON PRINCIPLE, NOT ON A MEASURED RETRIEVAL GAIN.
-- Read the next paragraph before quoting any benchmark number at it.
--
-- A three-arm BEAM A/B was run (graph off / unfiltered / filtered) and its
-- result is WITHDRAWN. BEAM with OGHAM_BENCH_MODE unset is not idempotent:
-- `hybrid_search_memories` multiplies relevance by
-- `(1.0 + ln(access_count + 1.0) * 0.1)`, and the PYTHON CLIENT then calls
-- `record_access()` on every returned row (`service.py:605`), which is where
-- the increment happens -- `record_access` is a separate function, NOT part of
-- the search RPC. Consecutive runs therefore reinforce whatever the previous
-- run retrieved. (An earlier revision of this header said the search function
-- itself incremented the counter. It does not; corrected after the ogham-cli
-- session checked the Go side and found its client performs no such write.) Three repeats under an IDENTICAL configuration gave
-- MRR 0.5439 / 0.5484 / 0.5511 -- a spread of 0.0072, monotonically increasing
-- with run order. The three arms had been run sequentially and produced
-- 0.5236 / 0.5338 / 0.5351, also monotonic in run order and inside that
-- spread. The apparent effect was the instrument, not the change. See TBU-252.
--
-- What stands without a benchmark: a graph walk whose traversable set is 93.7%
-- entities that were never people, produced by a classifier measured at ~0%
-- precision, is wrong on its face. Filtering them cannot cost recall -- the
-- seeds could never match a person entity in the first place, so nothing that
-- was reachable via a legitimate path becomes unreachable. It strictly removes
-- junk from an expansion.
--
-- Deliberately NOT done here: deleting the 16,651 person entity rows. Once the
-- walk ignores them they are inert, so that is a storage and traversal-cost
-- decision, not a correctness one.
--
-- Idempotent: CREATE OR REPLACE, no data touched, safe to re-run.

CREATE OR REPLACE FUNCTION spread_entity_activation_memories(
    seed_entity_tags text[],
    filter_profile text,
    max_depth int DEFAULT 2,
    decay float DEFAULT 0.65,
    min_activation float DEFAULT 0.1,
    max_results int DEFAULT 50
) RETURNS TABLE (memory_id uuid, activation float)
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path TO 'public', 'extensions' AS $$
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
                          AND e2.entity_type <> 'person'
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
