-- In-result contradiction pairs for supersession ranking (TBU-207).
--
-- `gap_contradictions_for_ids` (migration 039) deliberately EXCLUDES pairs
-- where both endpoints are in the supplied result set:
--
--     AND NOT (mr.source_id = ANY(...) AND mr.target_id = ANY(...))
--
-- That is correct for gap analysis, whose question is "what is missing from
-- these results". It is wrong for ranking, whose question is "which of these
-- results is stale" -- and the most common shape of a correction is that both
-- the correction and the memory it supersedes match the same query, so both
-- come back together and the pair is invisible.
--
-- Measured on the live store 2026-07-30: of 11 superseded rows returned across
-- 8 queries, the 2 that came back unmarked were exactly this case. They ranked
-- acceptably by luck rather than because anything enforced it.
--
-- This returns the complement: contradiction pairs with BOTH endpoints inside
-- the result set, oriented so the caller does not have to guess which side is
-- stale. Orientation is by recency, not edge direction -- a correction is
-- written after what it corrects, and all 148 contradicts edges in the live
-- store have the source strictly newer than the target.
--
-- Additive and idempotent: CREATE OR REPLACE only, no DDL on any table.

CREATE OR REPLACE FUNCTION in_result_contradictions(
    p_profile text,
    p_memory_ids uuid[]
)
RETURNS TABLE (
    stale_id text,
    newer_id text,
    strength float
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        CASE WHEN a.created_at >= b.created_at THEN b.id ELSE a.id END::text AS stale_id,
        CASE WHEN a.created_at >= b.created_at THEN a.id ELSE b.id END::text AS newer_id,
        mr.strength
    FROM memory_relationships mr
    JOIN memories a ON a.id = mr.source_id AND a.profile = p_profile
    JOIN memories b ON b.id = mr.target_id AND b.profile = p_profile
    WHERE mr.relationship = 'contradicts'
      AND mr.source_id = ANY(p_memory_ids)
      AND mr.target_id = ANY(p_memory_ids)
      -- Equal timestamps mean two peers written together, not a correction.
      -- Ranking one above the other would be a guess, so emit nothing.
      AND a.created_at <> b.created_at;
$$;

COMMENT ON FUNCTION in_result_contradictions(text, uuid[]) IS
    'Contradiction pairs with both endpoints inside a result set, oriented '
    'stale -> newer by created_at. Complements gap_contradictions_for_ids, '
    'which covers only pairs reaching outside the set. See TBU-207.';
