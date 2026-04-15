-- Migration 022: extend get_memory_stats_sql() with profile health counters.
--
-- Adds additive JSON fields for relationship, tagging, and decay health so
-- existing installs get the same stats payload as fresh installs from the
-- current schema snapshots.

create or replace function get_memory_stats_sql(filter_profile text default 'default')
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
    result jsonb;
begin
    WITH active_memories AS (
        SELECT id, source, tags, importance, last_accessed_at
        FROM memories
        WHERE profile = filter_profile
          AND (expires_at IS NULL OR expires_at > now())
    ),
    related_active_memories AS (
        SELECT mr.source_id AS memory_id
        FROM memory_relationships mr
        JOIN active_memories source_mem ON source_mem.id = mr.source_id
        JOIN active_memories target_mem ON target_mem.id = mr.target_id
        WHERE mr.source_id <> mr.target_id
        UNION
        SELECT mr.target_id AS memory_id
        FROM memory_relationships mr
        JOIN active_memories source_mem ON source_mem.id = mr.source_id
        JOIN active_memories target_mem ON target_mem.id = mr.target_id
        WHERE mr.source_id <> mr.target_id
    ),
    tag_rows AS (
        SELECT unnest(tags) AS tag
        FROM active_memories
        WHERE tags IS NOT NULL AND cardinality(tags) > 0
    )
    SELECT jsonb_build_object(
        'profile', filter_profile,
        'total', (SELECT count(*) FROM active_memories),
        'sources', COALESCE((
            SELECT jsonb_object_agg(source, cnt)
            FROM (
                SELECT coalesce(source, 'unknown') AS source, count(*) AS cnt
                FROM active_memories
                GROUP BY coalesce(source, 'unknown')
            ) s
        ), '{}'::jsonb),
        'top_tags', COALESCE((
            SELECT jsonb_agg(jsonb_build_object('tag', tag, 'count', cnt))
            FROM (
                SELECT tag, count(*) AS cnt
                FROM tag_rows
                GROUP BY tag
                ORDER BY cnt DESC, tag
                LIMIT 20
            ) t
        ), '[]'::jsonb),
        'relationships', jsonb_build_object(
            'orphan_count', (
                SELECT count(*)
                FROM active_memories m
                LEFT JOIN related_active_memories ram ON ram.memory_id = m.id
                WHERE ram.memory_id IS NULL
            )
        ),
        'tagging', jsonb_build_object(
            'untagged_count', (
                SELECT count(*)
                FROM active_memories
                WHERE tags IS NULL OR cardinality(tags) = 0
            ),
            'distinct_tag_count', (SELECT count(DISTINCT tag) FROM tag_rows)
        ),
        'decay', jsonb_build_object(
            'eligible_count', (
                SELECT count(*)
                FROM active_memories
                WHERE importance > 0.05
                  AND (
                      last_accessed_at IS NULL
                      OR last_accessed_at < now() - interval '7 days'
                  )
            ),
            'floor_count', (
                SELECT count(*)
                FROM active_memories
                WHERE importance <= 0.05
            )
        )
    ) INTO result;

    return result;
end;
$$;
