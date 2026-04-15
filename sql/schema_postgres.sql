-- Ogham MCP Schema (vanilla PostgreSQL / Neon)
-- Use this for Neon serverless, vanilla PostgreSQL, or any Postgres without Supabase.
-- No RLS, no anon role — connect directly via psycopg or connection pooler.

-- Enable pgvector extension
create extension if not exists vector with schema public;

-- Memories table
create table if not exists memories (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    embedding vector(512),
    metadata jsonb default '{}'::jsonb,
    source text,
    profile text not null default 'default',
    tags text[] default '{}',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    expires_at timestamptz,
    access_count integer not null default 0,
    last_accessed_at timestamptz,
    confidence float not null default 0.5,
    importance float not null default 0.5,
    surprise float not null default 0.5,
    compression_level integer not null default 0,
    original_content text,
    occurrence_period tstzrange,
    recurrence_days int[],
    fts tsvector generated always as (to_tsvector('english', content)) stored
);

-- lz4 TOAST compression for text columns (faster decompress than default pglz)
alter table memories alter column content set compression lz4;
alter table memories alter column original_content set compression lz4;
alter table memories alter column metadata set compression lz4;

-- HNSW index for fast cosine similarity search
create index if not exists memories_embedding_idx
    on memories using hnsw ((embedding::halfvec(512)) halfvec_cosine_ops)
    with (m = 16, ef_construction = 64);

-- GIN indexes for filtering
create index if not exists memories_metadata_idx on memories using gin (metadata jsonb_path_ops);
create index if not exists memories_tags_idx on memories using gin (tags);

-- GIN index for full-text search
create index if not exists memories_fts_idx on memories using gin (fts);

-- B-tree indexes
create index if not exists memories_profile_created_at_idx on memories (profile, created_at desc);
create index if not exists memories_source_idx on memories (source);

-- Partial index for expiration queries
create index if not exists memories_expires_at_idx on memories (expires_at)
    where expires_at is not null;

-- Temporal indexes (calendar/timeline)
create index if not exists idx_memories_occurrence on memories using gist (occurrence_period)
    where occurrence_period is not null;
create index if not exists idx_memories_recurrence on memories using gin (recurrence_days)
    where recurrence_days is not null;

-- Profile settings table for TTL configuration
create table if not exists profile_settings (
    profile text primary key,
    ttl_days integer check (ttl_days is null or ttl_days >= 1),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Relationship type enum
CREATE TYPE relationship_type AS ENUM (
    'similar',
    'related',
    'contradicts',
    'supports',
    'follows',
    'derived_from'
);

-- Edge table for memory relationships (knowledge graph)
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
    new_embedding vector(512),
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
        SELECT m.id, (1 - (m.embedding::halfvec(512) <=> new_embedding::halfvec(512)))::float AS similarity
        FROM memories m
        WHERE m.id != new_memory_id
          AND m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND 1 - (m.embedding::halfvec(512) <=> new_embedding::halfvec(512)) > link_threshold
        ORDER BY m.embedding::halfvec(512) <=> new_embedding::halfvec(512)
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

-- Auto-update updated_at trigger
create or replace function update_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists memories_updated_at on memories;
create trigger memories_updated_at
    before update on memories
    for each row
    execute function update_updated_at();

drop trigger if exists profile_settings_updated_at on profile_settings;
create trigger profile_settings_updated_at
    before update on profile_settings
    for each row
    execute function update_updated_at();

-- RPC: clean up expired memories for a profile
create or replace function cleanup_expired_memories(target_profile text)
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
    deleted_count integer;
begin
    delete from memories
    where profile = target_profile
      and expires_at is not null
      and expires_at < now();
    get diagnostics deleted_count = row_count;
    return deleted_count;
end;
$$;

-- RPC: count expired memories for a profile (preview before cleanup)
create or replace function count_expired_memories(target_profile text)
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
    expired_count integer;
begin
    select count(*)::integer into expired_count
    from memories
    where profile = target_profile
      and expires_at is not null
      and expires_at < now();
    return expired_count;
end;
$$;

-- RPC function for cosine similarity search with ACT-R temporal scoring
create or replace function match_memories(
    query_embedding vector(512),
    match_threshold float default 0.7,
    match_count int default 10,
    filter_tags text[] default null,
    filter_source text default null,
    filter_profile text default 'default'
)
returns table (
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
language plpgsql
security invoker
set search_path = public, extensions
as $$
begin
    return query
    select
        m.id,
        m.content,
        m.metadata,
        m.source,
        m.profile,
        m.tags,
        (1 - (m.embedding::halfvec(512) <=> query_embedding::halfvec(512)))::float as similarity,
        -- Relevance = similarity * softplus(ACT-R) * confidence * graph_boost
        -- ACT-R: B(M) = ln(n+1) - 0.5 * ln(ageDays / (n+1))
        -- softplus: ln(1 + exp(B)) keeps score positive
        -- graph_boost: (1 + sum(relationship_strength) * 0.2)
        (
            (1 - (m.embedding::halfvec(512) <=> query_embedding::halfvec(512))) *
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
        * (1.0 + g.graph_boost * 0.2)
        )::float as relevance,
        m.access_count,
        m.last_accessed_at,
        m.confidence,
        m.created_at,
        m.updated_at
    from public.memories m
    left join lateral (
        select coalesce(sum(r.strength), 0.0) as graph_boost
        from memory_relationships r
        where r.target_id = m.id or r.source_id = m.id
    ) g on true
    where
        1 - (m.embedding::halfvec(512) <=> query_embedding::halfvec(512)) > match_threshold
        and (filter_tags is null or m.tags && filter_tags)
        and (filter_source is null or m.source = filter_source)
        and m.profile = filter_profile
        and (m.expires_at is null or m.expires_at > now())
    order by relevance desc
    limit match_count;
end;
$$;

-- RPC: hybrid search combining semantic (pgvector) and keyword (tsvector) via RRF
CREATE OR REPLACE FUNCTION hybrid_search_memories(
    query_text text,
    query_embedding vector,
    match_count integer DEFAULT 10,
    filter_profile text DEFAULT 'default',
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL,
    full_text_weight float DEFAULT 0.3,
    semantic_weight float DEFAULT 0.7,
    rrf_k integer DEFAULT 10,
    filter_profiles text[] DEFAULT NULL,
    query_entity_tags text[] DEFAULT NULL,
    recency_decay float DEFAULT 0.0
)
RETURNS TABLE(
    id uuid, content text, metadata jsonb, source text, profile text, tags text[],
    similarity float, keyword_rank float, relevance float,
    access_count integer, last_accessed_at timestamptz, confidence float,
    created_at timestamptz, updated_at timestamptz
)
LANGUAGE sql
SET search_path = public, extensions
AS $function$
with semantic as (
    select
        m.id,
        (1 - (m.embedding::halfvec(512) <=> query_embedding::halfvec(512)))::float as similarity,
        row_number() over (order by m.embedding::halfvec(512) <=> query_embedding::halfvec(512)) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by m.embedding::halfvec(512) <=> query_embedding::halfvec(512)
    limit match_count * 3
),
keyword as (
    select
        m.id,
        ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34)::float as keyword_rank,
        row_number() over (order by ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34) desc) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and m.fts @@ websearch_to_tsquery(query_text)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by keyword_rank desc
    limit match_count * 3
),
fused as (
    select
        coalesce(s.id, k.id) as id,
        coalesce(s.similarity, 0.0) as similarity,
        coalesce(k.keyword_rank, 0.0) as keyword_rank,
        -- Reciprocal Rank Fusion: position-based, score-agnostic
        (
            semantic_weight * (1.0 / (rrf_k + coalesce(s.rank_ix, match_count * 3)))
            + full_text_weight * (1.0 / (rrf_k + coalesce(k.rank_ix, match_count * 3)))
        ) as score
    from semantic s
    full outer join keyword k on s.id = k.id
)
select
    m.id, m.content, m.metadata, m.source, m.profile, m.tags,
    f.similarity, f.keyword_rank,
    (
        f.score
        * m.importance
        * (1.0 + ln(m.access_count + 1.0) * 0.1)
        * m.confidence
        * (1.0 + g.graph_boost * 0.2)
        -- Entity overlap boost: query-conditioned re-ranking via tag intersection.
        -- When the query has extractable entities, memories sharing those entities
        -- get up to 1.3x boost. Bounded, safe, uses existing GIN-indexed tags.
        -- See docs/plans/2026-04-06-entity-graph-spreading-activation.md Stage 1.
        * (1.0 + case
            when query_entity_tags is null or cardinality(query_entity_tags) = 0 then 0.0
            else (select count(*)::float from unnest(query_entity_tags) qt
                  where qt = any(m.tags))
                 / cardinality(query_entity_tags) * 0.4
          end)
        -- Gated recency decay: exp(-decay * age_days). Only fires when
        -- recency_decay > 0 (caller gates by query type). Default 0 = no effect.
        * exp(-recency_decay * extract(epoch from (now() - m.created_at)) / 86400.0)
    )::float as relevance,
    m.access_count, m.last_accessed_at, m.confidence, m.created_at, m.updated_at
from fused f
join memories m on m.id = f.id
left join lateral (
    select coalesce(sum(r.strength), 0.0) as graph_boost
    from memory_relationships r
    where r.target_id = m.id or r.source_id = m.id
) g on true
order by relevance desc
limit match_count;
$function$;

-- RPC: record access for memories returned by search
create or replace function record_access(memory_ids uuid[])
returns void
language plpgsql
security invoker
set search_path = public
as $$
begin
    update memories
    set access_count = access_count + 1,
        last_accessed_at = now()
    where id = any(memory_ids);
end;
$$;

-- RPC: Bayesian confidence update
-- signal: 0.85 = reinforce, 0.15 = contradict, 0.5 = neutral
create or replace function update_confidence(
    memory_id uuid,
    signal float,
    memory_profile text
)
returns float
language plpgsql
security invoker
set search_path = public
as $$
declare
    current_conf float;
    posterior float;
    new_conf float;
begin
    select confidence into current_conf
    from memories
    where id = memory_id and profile = memory_profile;

    if not found then
        raise exception 'Memory % not found in profile %', memory_id, memory_profile;
    end if;

    posterior := (current_conf * signal) /
                (current_conf * signal + (1.0 - current_conf) * (1.0 - signal));
    new_conf := 0.95 * posterior + 0.025;

    update memories
    set confidence = new_conf
    where id = memory_id and profile = memory_profile;

    return new_conf;
end;
$$;

-- RPC: batch update embeddings (reduces N+1 round trips in re_embed_all)
create or replace function batch_update_embeddings(
    memory_ids uuid[],
    new_embeddings vector(512)[]
)
returns integer
language plpgsql
security invoker
set search_path = public, extensions
as $$
declare
    updated_count integer;
begin
    update memories m
    set embedding = u.emb,
        updated_at = now()
    from unnest(memory_ids, new_embeddings) as u(id, emb)
    where m.id = u.id;

    get diagnostics updated_count = row_count;
    return updated_count;
end;
$$;

-- Batch duplicate checking for import dedup.
-- Accepts an array of embeddings and returns a boolean array
-- indicating whether each embedding has a match above threshold.
-- Uses a simple cosine similarity check (no ACT-R scoring needed for dedup).
create or replace function batch_check_duplicates(
    query_embeddings vector(512)[],
    match_threshold float default 0.8,
    filter_profile text default 'default'
)
returns boolean[]
language plpgsql
security invoker
set search_path = public, extensions
as $$
declare
    results boolean[];
    i integer;
    found boolean;
begin
    -- Lower ef_search for dedup: we only need "is there a match?", not high recall
    perform set_config('hnsw.ef_search', '40', true);
    results := array[]::boolean[];
    for i in 1..array_length(query_embeddings, 1) loop
        select exists(
            select 1 from memories m
            where m.profile = filter_profile
              and (m.expires_at is null or m.expires_at > now())
              and 1 - (m.embedding::halfvec(512) <=> query_embeddings[i]::halfvec(512)) > match_threshold
            limit 1
        ) into found;
        results := array_append(results, found);
    end loop;
    return results;
end;
$$;

-- RPC function for profile counts (replaces Python-side counting)
create or replace function get_profile_counts()
returns table (profile text, count bigint)
language plpgsql
security invoker
set search_path = public
as $$
begin
    return query
    select m.profile, count(*) as count
    from public.memories m
    where m.expires_at is null or m.expires_at > now()
    group by m.profile
    order by m.profile;
end;
$$;

-- RPC function for memory stats (replaces 3 queries + Python counting)
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

-- RPC: explore knowledge graph — hybrid search seeds + relationship traversal
CREATE OR REPLACE FUNCTION explore_memory_graph(
    query_text text,
    query_embedding vector(512),
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

-- ── Hebbian decay + potentiation ───────────────────────────────────────
-- Memories not accessed within 7 days decay in importance.
-- Potentiated memories (access_count >= 10) decay much slower (LTP).
-- Floor at 0.05: keeps memories in the "unconscious" but retrievable.
-- Run as a batch job (cron, CLI, or MCP tool) -- not on-access.

CREATE OR REPLACE FUNCTION apply_hebbian_decay(
    target_profile text,
    batch_size int DEFAULT 1000
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $$
DECLARE
    rows_affected integer;
BEGIN
    WITH decay_targets AS (
        SELECT id FROM memories
        WHERE profile = target_profile
          AND importance > 0.05
          AND (expires_at IS NULL OR expires_at > now())
          AND (last_accessed_at IS NULL
               OR last_accessed_at < now() - interval '7 days')
        LIMIT batch_size
    )
    UPDATE memories
    SET importance = greatest(0.05,
        importance * power(
            CASE WHEN access_count >= 10 THEN 0.99 ELSE 0.95 END,
            extract(epoch from (now() - coalesce(last_accessed_at, created_at))) / 2592000.0
        )
    )
    WHERE id IN (SELECT id FROM decay_targets);

    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    RETURN rows_affected;
END;
$$;

-- ── Audit log (append-only event trail) ────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time timestamptz NOT NULL DEFAULT now(),
    profile text NOT NULL,
    operation text NOT NULL,              -- store | search | delete | update | re_embed
    resource_id uuid,                     -- memories.id if applicable
    outcome text NOT NULL DEFAULT 'success',
    source text,                          -- claude-code | cursor | gateway | cli
    embedding_model text,
    tokens_used integer,
    cost_usd numeric(10,6),
    result_ids uuid[],                    -- memory IDs returned by search
    result_count integer,
    query_hash text,                      -- sha256 of query text (not the query itself)
    metadata jsonb DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_log_profile_time
    ON audit_log (profile, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource
    ON audit_log (resource_id) WHERE resource_id IS NOT NULL;

-- ── Entity graph (spreading activation substrate) ─────────────────────

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

CREATE INDEX IF NOT EXISTS idx_entities_type_name
    ON entities (entity_type, canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_type
    ON entities (canonical_name, entity_type);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    profile text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
    ON memory_entities (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_profile
    ON memory_entities (entity_id, profile);

-- Refresh temporal span for a single entity
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

-- Spreading activation over the bipartite entity/memory graph
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
