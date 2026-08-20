-- Ogham MCP Schema
-- Run this in the Supabase SQL Editor
--
-- memory_lifecycle + triggers + decay params incorporate migrations 025 + 026.
-- Fresh installs land at post-026 state; upgraders from v0.10.x run ./sql/upgrade.sh.
--
-- TBU-159: vector/halfvec columns use a `:embedding_dim` placeholder instead
-- of a hardcoded dimension. Substitute it with your embedding provider's
-- output dimension (see EMBEDDING_DIM in src/ogham/config.py) BEFORE pasting
-- into the SQL Editor -- the raw file is not paste-ready as-is. Preprocess
-- as text, e.g.:
--   sed "s/:embedding_dim/$EMBEDDING_DIM/g" sql/schema.sql > /tmp/schema_applied.sql
-- then paste /tmp/schema_applied.sql's contents. `ogham init` does this for
-- you automatically (writes schema_<dim>d.sql next to this file).

-- Enable pgvector extension
create extension if not exists vector with schema extensions;

-- Memories table
create table if not exists memories (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    embedding extensions.vector(:embedding_dim),
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
    -- Multi-tenant: NULL for self-hosted single-tenant deployments,
    -- set by the gateway via SET LOCAL app.tenant_id for cloud requests.
    tenant_id uuid,
    fts tsvector generated always as (to_tsvector('english', content)) stored
);

-- lz4 TOAST compression for text columns (faster decompress than default pglz)
alter table memories alter column content set compression lz4;
alter table memories alter column original_content set compression lz4;
alter table memories alter column metadata set compression lz4;

-- Enable RLS (service role key bypasses; ready for multi-user later)
alter table memories enable row level security;
alter table memories force row level security;

-- Deny access via anon key by default
create policy "Deny anon access" on memories
    for all to anon using (false) with check (false);

-- Helper: returns the current tenant_id from session config, or NULL.
-- Wrapping in a STABLE function with EXCEPTION handling avoids the
-- "invalid input syntax for type uuid" error that happens when Postgres
-- evaluates an OR clause's cast on an empty session variable instead of
-- short-circuiting on the IS NULL / = '' check. STABLE means Postgres
-- evaluates it once per query, not once per row -- critical for query
-- performance under RLS.
--
-- SET search_path = pg_catalog is the Postgres-level mitigation for the
-- search-path-shadowing class of vulnerability (Supabase linter rule
-- 0011 function_search_path_mutable). Without this, a hostile user
-- with CREATE on a schema in their search_path could shadow
-- current_setting / nullif / the uuid cast and intercept the function
-- on every RLS check. All three primitives live in pg_catalog so we
-- pin to that only -- nothing from public is needed.
create or replace function current_tenant_id() returns uuid
    language plpgsql stable
    set search_path = pg_catalog
    as $$
begin
    return nullif(current_setting('app.tenant_id', true), '')::uuid;
exception
    when others then
        return null;
end;
$$;

-- Tenant-scoped access (gateway / multi-tenant):
-- - When current_tenant_id() is NULL (self-hosted, CLI, migrations,
--   no contextvar set) the policy returns all rows -- single-tenant fallback.
-- - When set (gateway requests) the policy filters to that tenant_id.
-- Variable name app.tenant_id is shared with the embedding_cache RLS
-- introduced in Phase 1 (2026-03-16). Service role bypasses both.
create policy "Tenant scoped access" on memories
    for all
    using (
        current_tenant_id() is null
        or tenant_id = current_tenant_id()
    )
    with check (
        current_tenant_id() is null
        or tenant_id = current_tenant_id()
    );

-- HNSW index for fast cosine similarity search
create index if not exists memories_embedding_idx
    on memories using hnsw ((embedding::halfvec(:embedding_dim)) extensions.halfvec_cosine_ops)
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
    decay_lambda double precision not null default 0.1,
    decay_beta double precision not null default 0.4,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- RLS for profile_settings
alter table profile_settings enable row level security;
alter table profile_settings force row level security;

create policy "Deny anon access" on profile_settings
    for all to anon using (false) with check (false);

-- No authenticated policy: service_role bypasses RLS.
-- Add scoped policies here when building multi-user support.

-- ── Memory lifecycle (FRESH / STABLE / EDITING) ───────────────────────
-- Lifecycle state lives in its own table so writes don't trigger HNSW
-- tuple rewrites on memories.embedding. See migrations 025 + 026 for the
-- history; fresh installs land directly at the post-026 shape.
create table if not exists memory_lifecycle (
    memory_id uuid primary key references memories(id) on delete cascade,
    profile text not null,
    stage text not null default 'fresh',
    stage_entered_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint memory_lifecycle_stage_valid check (stage in ('fresh', 'stable', 'editing'))
);

-- Partial index for sweeps (advance_stages, close_editing_windows).
create index if not exists memory_lifecycle_transitioning_idx
    on memory_lifecycle (profile, stage_entered_at)
    where stage in ('fresh', 'editing');

-- Full index for lifecycle_pipeline_counts which groups across all stages.
create index if not exists memory_lifecycle_profile_stage_idx
    on memory_lifecycle (profile, stage);

-- RLS for memory_lifecycle: mirror the "Deny anon access" pattern.
-- service_role bypasses RLS. Tenant scoping is enforced via joins to
-- memories (lifecycle rows have no tenant_id column of their own --
-- they're 1:1 with memories and follow that table's tenancy).
alter table memory_lifecycle enable row level security;
alter table memory_lifecycle force row level security;

create policy "Deny anon access" on memory_lifecycle
    for all to anon using (false) with check (false);

-- Trigger: auto-init a lifecycle row when a new memory is inserted.
create or replace function init_memory_lifecycle() returns trigger as $$
begin
    insert into memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
    values (new.id, new.profile, 'fresh', new.created_at, new.created_at)
    on conflict (memory_id) do nothing;
    return new;
end;
$$ language plpgsql set search_path to 'public', 'pg_catalog';

drop trigger if exists memories_init_lifecycle on memories;
create trigger memories_init_lifecycle
    after insert on memories
    for each row
    execute function init_memory_lifecycle();

-- Trigger: keep memory_lifecycle.profile in sync when a memory is moved
-- between profiles (rare but possible).
create or replace function sync_memory_lifecycle_profile() returns trigger as $$
begin
    if new.profile is distinct from old.profile then
        update memory_lifecycle
           set profile = new.profile, updated_at = now()
         where memory_id = new.id;
    end if;
    return new;
end;
$$ language plpgsql set search_path to 'public', 'pg_catalog';

drop trigger if exists memories_sync_lifecycle_profile on memories;
create trigger memories_sync_lifecycle_profile
    after update of profile on memories
    for each row
    execute function sync_memory_lifecycle_profile();

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
    -- Multi-tenant: denormalised from memories.tenant_id for query
    -- performance (RLS policy can filter without joining to memories).
    tenant_id uuid,
    CONSTRAINT unique_relationship UNIQUE (source_id, target_id, relationship)
);

-- RLS for memory_relationships
ALTER TABLE memory_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_relationships FORCE ROW LEVEL SECURITY;

CREATE POLICY "Deny anon access" ON memory_relationships
    FOR ALL TO anon USING (false) WITH CHECK (false);

-- Tenant-scoped access (uses the same current_tenant_id() helper as memories)
CREATE POLICY "Tenant scoped access" ON memory_relationships
    FOR ALL
    USING (
        current_tenant_id() IS NULL
        OR tenant_id = current_tenant_id()
    )
    WITH CHECK (
        current_tenant_id() IS NULL
        OR tenant_id = current_tenant_id()
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
    new_embedding extensions.vector(:embedding_dim),
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
        SELECT m.id, (1 - (m.embedding::halfvec(:embedding_dim) <=> new_embedding::halfvec(:embedding_dim)))::float AS similarity
        FROM memories m
        WHERE m.id != new_memory_id
          AND m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND 1 - (m.embedding::halfvec(:embedding_dim) <=> new_embedding::halfvec(:embedding_dim)) > link_threshold
        ORDER BY m.embedding::halfvec(:embedding_dim) <=> new_embedding::halfvec(:embedding_dim)
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
    query_embedding extensions.vector(:embedding_dim),
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
        (1 - (m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim)))::float as similarity,
        -- Relevance = similarity * softplus(ACT-R) * confidence * graph_boost
        -- ACT-R: B(M) = ln(n+1) - 0.5 * ln(ageDays / (n+1))
        -- softplus: ln(1 + exp(B)) keeps score positive
        -- graph_boost: (1 + sum(relationship_strength) * 0.2)
        (
            (1 - (m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim))) *
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
        1 - (m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim)) > match_threshold
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
        (1 - (m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim)))::float as similarity,
        row_number() over (order by m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim)) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by m.embedding::halfvec(:embedding_dim) <=> query_embedding::halfvec(:embedding_dim)
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
        * (1.0 + case
            when query_entity_tags is null or cardinality(query_entity_tags) = 0 then 0.0
            else (select count(*)::float from unnest(query_entity_tags) qt
                  where qt = any(m.tags))
                 / cardinality(query_entity_tags) * 0.4
          end)
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
    new_embeddings extensions.vector(:embedding_dim)[]
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
    query_embeddings extensions.vector(:embedding_dim)[],
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
              and 1 - (m.embedding::halfvec(:embedding_dim) <=> query_embeddings[i]::halfvec(:embedding_dim)) > match_threshold
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
    query_embedding extensions.vector(:embedding_dim),
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

-- ── Audit log (append-only event trail) ────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time timestamptz NOT NULL DEFAULT now(),
    profile text NOT NULL,
    operation text NOT NULL,
    resource_id uuid,
    outcome text NOT NULL DEFAULT 'success',
    source text,
    embedding_model text,
    tokens_used integer,
    cost_usd numeric(10,6),
    result_ids uuid[],
    result_count integer,
    query_hash text,
    metadata jsonb DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_log_profile_time
    ON audit_log (profile, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource
    ON audit_log (resource_id) WHERE resource_id IS NOT NULL;

-- RLS for audit_log: mirror the "Deny anon access" pattern (migration 027).
-- audit_log is the table recording who did what, so a fresh install must not
-- be the one place anon can read. The Python server writes via service_role,
-- which bypasses RLS, so functional behaviour is unchanged. TBU-247.
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON audit_log
    FOR ALL TO anon USING (false) WITH CHECK (false);

-- ── Entity graph (spreading activation substrate) ─────────────────────

CREATE TABLE IF NOT EXISTS entities (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    mention_count integer NOT NULL DEFAULT 0,
    temporal_span float NOT NULL DEFAULT 1.0,
    session_count integer NOT NULL DEFAULT 1,
    -- How this entity was established (TBU-261, migration 049). NOT a
    -- confidence score -- a fixed class per rule, nothing estimated.
    evidence_class text NOT NULL DEFAULT 'inferred'
        CHECK (evidence_class IN ('structured', 'syntactic', 'inferred')),
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

-- RLS for entity tables
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities FORCE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON entities
    FOR ALL TO anon USING (false) WITH CHECK (false);

ALTER TABLE memory_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_entities FORCE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON memory_entities
    FOR ALL TO anon USING (false) WITH CHECK (false);

CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
    ON memory_entities (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity_profile
    ON memory_entities (entity_id, profile);

-- link_memory_entities: upsert entities + link in memory_entities for one
-- memory. Both live writes (service.store_memory) and the backfill loop call
-- this, so the two paths produce identical state. Idempotent on
-- memory_entities via ON CONFLICT DO NOTHING. Mention counts accumulate on
-- entities so the temporal-span refresher has data to work with.
-- Returns the number of (memory, entity) edges newly inserted.
--
-- Backported from migration 036 (TBU-221). Its two sibling functions were
-- copied into the schema files when 036 landed and this one was missed, so a
-- FRESH install had the entity tables but no way to populate them: every
-- store_memory raised UndefinedFunction, service.py swallowed it at debug
-- level, memory_entities stayed empty, and the OKF export's MENTIONS bridge
-- was silently always empty. A fresh install applies a schema file and nothing
-- else -- there is no migration pass -- so a function that lives only in a
-- migration never arrives. Held by
-- tests/test_schema_parity.py::test_migration_functions_are_backported_into_every_schema
-- Evidence class mapping (TBU-261, migration 049). Kept as functions so the
-- write path and any backfill share one definition.
CREATE OR REPLACE FUNCTION entity_evidence_class(p_entity_type text)
RETURNS text
LANGUAGE sql IMMUTABLE
SET search_path = public, extensions
AS $$
    SELECT CASE p_entity_type
        WHEN 'entity'   THEN 'syntactic'
        WHEN 'file'     THEN 'syntactic'
        WHEN 'error'    THEN 'syntactic'
        WHEN 'quantity' THEN 'syntactic'
        ELSE 'inferred'
    END;
$$;

COMMENT ON COLUMN entities.evidence_class IS
    'How this entity was established: structured (adapter provenance), '
    'syntactic (unambiguous marker in the text), inferred (dictionary or '
    'keyword lookup). NOT a confidence score -- see TBU-261 and migration 049.';

COMMENT ON FUNCTION entity_evidence_class(text) IS
    'Maps an entity_type to its evidence class (TBU-261). Single source of '
    'truth -- link_memory_entities and the 049 backfill both call this.';

CREATE OR REPLACE FUNCTION entity_evidence_rank(p_class text)
RETURNS integer
LANGUAGE sql IMMUTABLE
SET search_path = public, extensions
AS $$
    SELECT CASE p_class
        WHEN 'structured' THEN 3
        WHEN 'syntactic'  THEN 2
        ELSE 1
    END;
$$;

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
        INSERT INTO entities (canonical_name, entity_type, mention_count, evidence_class)
        SELECT cn, et, 1, entity_evidence_class(et) FROM parsed
        ON CONFLICT (canonical_name, entity_type) DO UPDATE
            SET mention_count = entities.mention_count + 1,
                evidence_class = CASE
                    WHEN entity_evidence_rank(EXCLUDED.evidence_class)
                       > entity_evidence_rank(entities.evidence_class)
                    THEN EXCLUDED.evidence_class
                    ELSE entities.evidence_class
                END
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

-- Refresh temporal span for a single entity
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

-- Spreading activation over the bipartite entity/memory graph
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

-- v0.14: lock down entity-graph SECURITY DEFINER functions to service_role
-- only. These bypass RLS by design and were never meant to be part of
-- the public REST surface. Migration 037 keeps existing deployments in
-- sync with this schema-time grant. See migration 037 header for the
-- full rationale.
REVOKE EXECUTE ON FUNCTION link_memory_entities(uuid, text, text[]) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION entity_evidence_class(text) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION entity_evidence_rank(text) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION refresh_entity_temporal_span(bigint) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION spread_entity_activation_memories(text[], text, integer, double precision, double precision, integer) FROM anon, authenticated, PUBLIC;

-- ── Typed entity edges (v0.16 typed-edge context graph, TBU-110) ──────
-- See sql/migrations/041_entity_edges.sql, 042_entity_edge_predicates.sql,
-- 043_entity_aliases.sql for design notes.

CREATE TABLE IF NOT EXISTS entity_edges (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_id    bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate     text NOT NULL,
    object_id     bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    profile       text NOT NULL,
    fact_id       uuid,
    strength      real NOT NULL DEFAULT 1.0,
    metadata      jsonb NOT NULL DEFAULT '{}',
    derived_from jsonb NOT NULL DEFAULT '[]',
    valid_from    timestamptz NOT NULL DEFAULT now(),
    valid_to      timestamptz,
    superseded_by bigint REFERENCES entity_edges(id),
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (subject_id <> object_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS entity_edges_current_uq
    ON entity_edges(subject_id, predicate, object_id, profile)
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS entity_edges_subject_pred_current
    ON entity_edges(subject_id, predicate) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS entity_edges_object_pred_current
    ON entity_edges(object_id, predicate) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS entity_edges_profile_current
    ON entity_edges(profile) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS entity_edges_derived_from_gin
    ON entity_edges USING gin (derived_from);

ALTER TABLE entity_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_edges FORCE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON entity_edges
    FOR ALL TO anon USING (false) WITH CHECK (false);

CREATE TABLE IF NOT EXISTS entity_edge_predicates (
    predicate      text PRIMARY KEY,
    label          text NOT NULL,
    description    text,
    inverse        text,
    scope          text NOT NULL CHECK (scope IN ('entity','memory')),
    ogham_uri      text NOT NULL,
    schema_org_uri text,
    iirds_uri      text
);

-- v1 seed: 16 entity-scope predicate rows (6 inverse pairs + 4 standalone).
-- SUPERSEDES intentionally omitted per TBU-109 (redundant with valid_to).
INSERT INTO entity_edge_predicates(predicate, label, description, inverse, scope, ogham_uri, schema_org_uri, iirds_uri) VALUES
    ('DEPENDS_ON',      'depends on',       'Subject requires object to function or complete',            'DEPENDED_ON_BY', 'entity', 'https://ogham-mcp.dev/vocab#DEPENDS_ON',     NULL,                            NULL),
    ('DEPENDED_ON_BY',  'depended on by',   'Inverse of DEPENDS_ON',                                       'DEPENDS_ON',     'entity', 'https://ogham-mcp.dev/vocab#DEPENDED_ON_BY', NULL,                            NULL),
    ('OWNS',            'owns',             'Subject has ownership or authority over object',              'OWNED_BY',       'entity', 'https://ogham-mcp.dev/vocab#OWNS',           'https://schema.org/owns',       NULL),
    ('OWNED_BY',        'owned by',         'Inverse of OWNS',                                             'OWNS',           'entity', 'https://ogham-mcp.dev/vocab#OWNED_BY',       'https://schema.org/owner',      NULL),
    ('ASSIGNED_TO',     'assigned to',      'Subject is assigned to object (task -> person, item -> box)', 'HAS_ASSIGNEE',   'entity', 'https://ogham-mcp.dev/vocab#ASSIGNED_TO',    NULL,                            NULL),
    ('HAS_ASSIGNEE',    'has assignee',     'Inverse of ASSIGNED_TO',                                      'ASSIGNED_TO',    'entity', 'https://ogham-mcp.dev/vocab#HAS_ASSIGNEE',   NULL,                            NULL),
    ('DECIDED',         'decided',          'Subject decided on object (agent -> decision fact)',          NULL,             'entity', 'https://ogham-mcp.dev/vocab#DECIDED',        NULL,                            NULL),
    ('MENTIONS',        'mentions',         'Subject mentions object in a memory / message',               NULL,             'entity', 'https://ogham-mcp.dev/vocab#MENTIONS',       'https://schema.org/mentions',   NULL),
    ('BLOCKS',          'blocks',           'Subject blocks progress on object',                           'BLOCKED_BY',     'entity', 'https://ogham-mcp.dev/vocab#BLOCKS',         NULL,                            NULL),
    ('BLOCKED_BY',      'blocked by',       'Inverse of BLOCKS',                                           'BLOCKS',         'entity', 'https://ogham-mcp.dev/vocab#BLOCKED_BY',     NULL,                            NULL),
    ('PART_OF',         'part of',          'Subject is a structural component of object',                 'CONTAINS',       'entity', 'https://ogham-mcp.dev/vocab#PART_OF',        'https://schema.org/isPartOf',   NULL),
    ('CONTAINS',        'contains',         'Inverse of PART_OF',                                          'PART_OF',        'entity', 'https://ogham-mcp.dev/vocab#CONTAINS',       'https://schema.org/hasPart',    NULL),
    ('SUPPORTS',        'supports',         'Subject provides evidence for object (entity-scope)',         'CONTRADICTS',    'entity', 'https://ogham-mcp.dev/vocab#SUPPORTS',       NULL,                            NULL),
    ('CONTRADICTS',     'contradicts',      'Subject provides counter-evidence to object (entity-scope)',  'SUPPORTS',       'entity', 'https://ogham-mcp.dev/vocab#CONTRADICTS',    NULL,                            NULL),
    ('EVOLVED_INTO',    'evolved into',     'Object is a later version of subject',      NULL,             'entity', 'https://ogham-mcp.dev/vocab#EVOLVED_INTO',   NULL,                            NULL),
    ('RELATED_TO',      'related to',       'Low-signal catchall -- prefer a specific predicate',          NULL,             'entity', 'https://ogham-mcp.dev/vocab#RELATED_TO',     NULL,                            NULL)
ON CONFLICT (predicate) DO NOTHING;

ALTER TABLE entity_edge_predicates ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_edge_predicates FORCE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON entity_edge_predicates
    FOR ALL TO anon USING (false) WITH CHECK (false);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_id  bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias      text NOT NULL,
    profile    text NOT NULL,
    strength   real NOT NULL DEFAULT 1.0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(alias, profile)
);

CREATE INDEX IF NOT EXISTS entity_aliases_entity ON entity_aliases(entity_id);

ALTER TABLE entity_aliases ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases FORCE ROW LEVEL SECURITY;
CREATE POLICY "Deny anon access" ON entity_aliases
    FOR ALL TO anon USING (false) WITH CHECK (false);

-- =====================================================
-- PostgREST / Supabase Data API grants (added 2026-05)
-- =====================================================
-- On 2026-04-28 Supabase announced that tables created in `public`
-- will no longer be auto-exposed to the Data API. The change becomes
-- the default for new projects on 2026-05-30 and is enforced on all
-- existing projects on 2026-10-30.
-- Source: https://github.com/orgs/supabase/discussions/45329
--
-- Ogham's Python backend talks to Supabase via PostgREST using the
-- service_role / sb_secret_ key. Without explicit table-level
-- GRANTs, PostgREST returns "42501: permission denied" once Supabase
-- revokes its platform-level default grant.
--
-- Granted to service_role only. anon is blocked at the RLS layer
-- ("Deny anon access" policies on every table) and locked out of RPC
-- EXECUTE in migration 037. authenticated is unused by Ogham.
--
-- Migration 038_data_api_grants.sql provides the upgrade path for
-- existing self-hosted installs. topic_summaries + topic_summary_sources
-- are granted further down, after the statements that create them.
GRANT SELECT, INSERT, UPDATE, DELETE ON public.memories             TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.profile_settings     TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.memory_lifecycle     TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.memory_relationships TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.audit_log            TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.entities             TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.memory_entities      TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_edges         TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_edge_predicates TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_aliases       TO service_role;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- ══════════════════════════════════════════════════════════════════════════
-- Backported from migrations (TBU-223)
--
-- `ogham init` applies ONE schema file and no migrations, so anything a
-- migration added that was never copied here is absent forever on new
-- installs while working fine on databases that grew through the migration
-- history. 71 objects had drifted this way -- the whole wiki feature, the
-- lifecycle and graph RPCs, in_result_contradictions (047, the supersession
-- ranking shipped in v0.17.2) and two triggers.
--
-- Definitions below are the FINAL form, read from a database with the full
-- migration history applied (pg_get_functiondef / pg_get_triggerdef), not
-- copied from the migration that first created them: wiki_topic_upsert is
-- replaced by 033 and wiki_topic_search by 034, so copying from 031 would
-- have backported a stale version.
--
-- Vector dims are re-parameterised to :embedding_dim per TBU-159.
-- Held by tests/test_schema_migration_drift.py.
--
-- DELIBERATELY NOT BACKPORTED (still in tests/schema_drift_baseline.yaml):
--   * tenant isolation (019) -- tenant_id columns, current_tenant_id(),
--     idx_*_tenant_* : a managed-gateway feature, and that gateway is not
--     running. Needs a product ruling, not a silent schema change.
--   * memories.sparse_embedding (016) -- referenced nowhere in src/, and
--     `sparsevec` needs pgvector >= 0.7, so adding it to the INSTALL gate
--     would break self-hosters on older pgvector to gain a column that
--     nothing reads.
--   * backfill_recurrence() (012) -- a one-off backfill helper, called by
--     nothing in src/.
--   * the auto_link_memory(uuid, vector, text, float, int) overload -- both
--     backends call the signature already defined above (postgres
--     positionally, supabase by name with max_links); the migration's top_n
--     variant is dead.
-- ══════════════════════════════════════════════════════════════════════════

-- ── Temporal auto-extraction (migration 015) ──────────────────────────

CREATE OR REPLACE FUNCTION public.extract_occurrence_from_content()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
DECLARE
    date_str text;
    parsed date;
BEGIN
    -- Only fire if occurrence_period is not already set
    IF NEW.occurrence_period IS NOT NULL THEN
        RETURN NEW;
    END IF;

    -- Extract [Date: YYYY-MM-DD] prefix
    date_str := substring(NEW.content FROM '\[Date:\s*(\d{4}-\d{2}-\d{2})\]');
    IF date_str IS NOT NULL THEN
        BEGIN
            parsed := date_str::date;
            NEW.occurrence_period := tstzrange(
                parsed::timestamptz,
                (parsed + interval '1 day')::timestamptz
            );
        EXCEPTION WHEN OTHERS THEN
            -- Invalid date string, skip
            NULL;
        END;
    END IF;

    RETURN NEW;
END;
$function$;


-- ── Topic summaries / wiki (migrations 028-034, 040) ──────────────────

CREATE TABLE IF NOT EXISTS topic_summaries (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  topic_key text NOT NULL,
  profile_id text NOT NULL,
  content text NOT NULL,
  embedding vector(:embedding_dim),
  source_count integer NOT NULL,
  source_cursor uuid,
  source_hash bytea NOT NULL,
  token_count integer,
  importance double precision NOT NULL DEFAULT 0.5,
  model_used text NOT NULL,
  version integer NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'fresh'::text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  stale_reason text,
  tldr_one_line text,
  tldr_short text
);

ALTER TABLE topic_summaries ADD CONSTRAINT topic_summaries_profile_topic_unique UNIQUE (profile_id, topic_key);
ALTER TABLE topic_summaries ADD CONSTRAINT topic_summaries_pkey PRIMARY KEY (id);
ALTER TABLE topic_summaries ADD CONSTRAINT topic_summaries_status_valid CHECK ((status = ANY (ARRAY['fresh'::text, 'stale'::text, 'regenerating'::text])));

CREATE INDEX IF NOT EXISTS topic_summaries_embedding_hnsw_idx ON public.topic_summaries USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64') WHERE (status = 'fresh'::text);
CREATE INDEX IF NOT EXISTS topic_summaries_profile_fresh_idx ON public.topic_summaries USING btree (profile_id, updated_at DESC) WHERE (status = 'fresh'::text);
CREATE INDEX IF NOT EXISTS topic_summaries_stale_sweep_idx ON public.topic_summaries USING btree (updated_at) WHERE (status = 'fresh'::text);

CREATE TABLE IF NOT EXISTS topic_summary_sources (
  summary_id uuid NOT NULL,
  memory_id uuid NOT NULL
);

ALTER TABLE topic_summary_sources ADD CONSTRAINT topic_summary_sources_pkey PRIMARY KEY (summary_id, memory_id);
ALTER TABLE topic_summary_sources ADD CONSTRAINT topic_summary_sources_memory_id_fkey FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE;
ALTER TABLE topic_summary_sources ADD CONSTRAINT topic_summary_sources_summary_id_fkey FOREIGN KEY (summary_id) REFERENCES topic_summaries(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS topic_summary_sources_memory_id_idx ON public.topic_summary_sources USING btree (memory_id);

-- Data API grants for the wiki tables. The main GRANT block above runs before
-- these two tables exist, which is why they are granted here and not there.
-- Migration 038 covers all nine tables; a fresh install was getting seven.
-- TBU-247.
GRANT SELECT, INSERT, UPDATE, DELETE ON public.topic_summaries       TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.topic_summary_sources TO service_role;

CREATE OR REPLACE FUNCTION public.topic_summaries_set_updated_at()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'public', 'pg_catalog'
AS $function$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_upsert(p_profile text, p_topic_key text, p_content text, p_embedding vector, p_source_memory_ids uuid[], p_model_used text, p_source_cursor uuid, p_source_hash bytea, p_token_count integer DEFAULT NULL::integer, p_importance double precision DEFAULT 0.5, p_tldr_one_line text DEFAULT NULL::text, p_tldr_short text DEFAULT NULL::text)
 RETURNS topic_summaries
 LANGUAGE plpgsql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
DECLARE
    upserted topic_summaries;
BEGIN
    INSERT INTO topic_summaries (
        topic_key, profile_id, content, embedding,
        source_count, source_cursor, source_hash,
        token_count, importance, model_used,
        tldr_one_line, tldr_short
    )
    VALUES (
        p_topic_key, p_profile, p_content, p_embedding,
        cardinality(p_source_memory_ids), p_source_cursor, p_source_hash,
        p_token_count, p_importance, p_model_used,
        p_tldr_one_line, p_tldr_short
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
        tldr_one_line = EXCLUDED.tldr_one_line,
        tldr_short = EXCLUDED.tldr_short,
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
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_search(p_profile text, p_query_embedding vector, p_top_k integer DEFAULT 3, p_min_similarity double precision DEFAULT 0.0)
 RETURNS TABLE(id uuid, topic_key text, profile_id text, content text, tldr_one_line text, tldr_short text, source_count integer, source_cursor uuid, source_hash bytea, model_used text, version integer, status text, updated_at timestamp with time zone, similarity double precision)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    -- HNSW + threshold trap: combining `WHERE similarity >= threshold`
    -- with `ORDER BY <=> ... LIMIT k` defeats the index when the
    -- threshold filters out top-k. Postgres falls back to scanning the
    -- HNSW tail row-by-row. Wrap the index-driven top-k in a CTE,
    -- apply the threshold AFTER. The index path then runs unfiltered
    -- and the threshold trims the (already small) output.
    WITH top_k AS (
        SELECT id, topic_key, profile_id, content,
               tldr_one_line, tldr_short,
               source_count, source_cursor, source_hash,
               model_used, version, status, updated_at,
               1 - (embedding <=> p_query_embedding) AS similarity
          FROM topic_summaries
         WHERE profile_id = p_profile
           AND status = 'fresh'
           AND embedding IS NOT NULL
         ORDER BY embedding <=> p_query_embedding
         LIMIT p_top_k
    )
    SELECT * FROM top_k WHERE similarity >= p_min_similarity;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_get_by_key(p_profile text, p_topic_key text)
 RETURNS SETOF topic_summaries
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT * FROM topic_summaries
     WHERE profile_id = p_profile AND topic_key = p_topic_key
     LIMIT 1;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_get_affected(p_memory_id uuid)
 RETURNS TABLE(id uuid, profile_id text, topic_key text, status text, version integer)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT ts.id, ts.profile_id, ts.topic_key, ts.status, ts.version
      FROM topic_summary_sources tss
      JOIN topic_summaries ts ON ts.id = tss.summary_id
     WHERE tss.memory_id = p_memory_id;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_list_stale(p_profile text DEFAULT NULL::text, p_older_than_days integer DEFAULT NULL::integer)
 RETURNS SETOF topic_summaries
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT * FROM topic_summaries
     WHERE status = 'stale'
       AND (p_profile IS NULL OR profile_id = p_profile)
       AND (p_older_than_days IS NULL
            OR updated_at < now() - make_interval(days => p_older_than_days));
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_list_fresh_for_drift(p_profile text)
 RETURNS TABLE(id uuid, topic_key text, source_hash bytea)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT id, topic_key, source_hash
      FROM topic_summaries
     WHERE profile_id = p_profile
       AND status = 'fresh';
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_mark_stale(p_summary_id uuid, p_reason text DEFAULT NULL::text)
 RETURNS void
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    UPDATE topic_summaries
       SET status = 'stale', stale_reason = p_reason
     WHERE id = p_summary_id;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_topic_sweep_stale(p_profile text, p_older_than_days integer DEFAULT 30)
 RETURNS integer
 LANGUAGE plpgsql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
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
$function$;

CREATE OR REPLACE FUNCTION public.wiki_lint_orphans(p_profile text, p_sample_size integer DEFAULT 10, p_grace_minutes integer DEFAULT 5)
 RETURNS TABLE(id text, content text, tags text[], created_at timestamp with time zone, total_count bigint)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
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
$function$;

CREATE OR REPLACE FUNCTION public.wiki_lint_contradictions(p_profile text, p_sample_size integer DEFAULT 10)
 RETURNS TABLE(source_id text, target_id text, strength double precision, created_at timestamp with time zone, total_count bigint)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    WITH all_pairs AS (
        SELECT mr.source_id, mr.target_id, mr.strength, mr.created_at
          FROM memory_relationships mr
          JOIN memories ms ON ms.id = mr.source_id AND ms.profile = p_profile
          JOIN memories mt ON mt.id = mr.target_id AND mt.profile = p_profile
         WHERE mr.relationship = 'contradicts'
    )
    SELECT mr.source_id::text, mr.target_id::text, mr.strength, mr.created_at,
           (SELECT count(*) FROM all_pairs)
      FROM all_pairs mr
     ORDER BY mr.created_at DESC
     LIMIT p_sample_size;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_lint_stale_lifecycle(p_profile text, p_older_than_days integer DEFAULT 90, p_sample_size integer DEFAULT 10)
 RETURNS TABLE(id text, stage text, stage_entered_at timestamp with time zone, content text, total_count bigint)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
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
$function$;

CREATE OR REPLACE FUNCTION public.wiki_recompute_get_source_ids(p_profile text, p_tag text)
 RETURNS TABLE(id text)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT id::text
      FROM memories
     WHERE profile = p_profile
       AND p_tag = ANY(tags)
       AND (expires_at IS NULL OR expires_at > now())
     ORDER BY id;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_recompute_get_source_content(p_memory_ids uuid[])
 RETURNS TABLE(id text, content text)
 LANGUAGE sql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT id::text, content
      FROM memories
     WHERE id = ANY(p_memory_ids)
     ORDER BY id;
$function$;

CREATE OR REPLACE FUNCTION public.wiki_walk_graph(p_start_id uuid, p_max_depth integer DEFAULT 1, p_direction text DEFAULT 'both'::text, p_min_strength double precision DEFAULT 0.0, p_relationship_types text[] DEFAULT NULL::text[], p_result_limit integer DEFAULT 50)
 RETURNS TABLE(id uuid, content text, metadata jsonb, source text, tags text[], confidence double precision, depth integer, relationship text, edge_strength double precision, connected_from uuid, direction_used text)
 LANGUAGE plpgsql
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
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
$function$;


-- ── Lifecycle + graph RPCs (migration 035) ────────────────────────────

CREATE OR REPLACE FUNCTION public.lifecycle_advance_fresh_to_stable(p_profile text, p_cutoff timestamp with time zone, p_s_gate double precision, p_i_gate double precision)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
DECLARE
    v_count integer;
BEGIN
    WITH advanced AS (
        UPDATE memory_lifecycle AS ml
           SET stage            = 'stable',
               stage_entered_at = now(),
               updated_at       = now()
          FROM memories AS m
         WHERE ml.memory_id        = m.id
           AND ml.profile          = p_profile
           AND ml.stage            = 'fresh'
           AND ml.stage_entered_at <= p_cutoff
           AND (m.surprise >= p_s_gate OR m.importance >= p_i_gate)
        RETURNING ml.memory_id
    )
    SELECT count(*)::integer INTO v_count FROM advanced;
    RETURN v_count;
END;
$function$;

CREATE OR REPLACE FUNCTION public.lifecycle_close_editing_windows(p_profile text, p_cutoff timestamp with time zone)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
DECLARE
    v_count integer;
BEGIN
    WITH closed AS (
        UPDATE memory_lifecycle
           SET stage            = 'stable',
               stage_entered_at = now(),
               updated_at       = now()
         WHERE profile           = p_profile
           AND stage             = 'editing'
           AND stage_entered_at <= p_cutoff
        RETURNING memory_id
    )
    SELECT count(*)::integer INTO v_count FROM closed;
    RETURN v_count;
END;
$function$;

CREATE OR REPLACE FUNCTION public.lifecycle_open_editing_window(p_ids uuid[])
 RETURNS void
 LANGUAGE sql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    UPDATE memory_lifecycle
       SET stage            = 'editing',
           stage_entered_at = now(),
           updated_at       = now()
     WHERE memory_id = ANY(p_ids)
       AND stage = 'stable';
$function$;

CREATE OR REPLACE FUNCTION public.lifecycle_pipeline_counts(p_profile text)
 RETURNS TABLE(stage text, n bigint)
 LANGUAGE sql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
    SELECT stage, count(*)::bigint AS n
      FROM memory_lifecycle
     WHERE profile = p_profile
     GROUP BY stage;
$function$;


-- ── Retrieval (migrations 035, 039, 047) ──────────────────────────────

CREATE OR REPLACE FUNCTION public.in_result_contradictions(p_profile text, p_memory_ids uuid[])
 RETURNS TABLE(stale_id text, newer_id text, strength double precision)
 LANGUAGE sql
 STABLE
AS $function$
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
$function$;

CREATE OR REPLACE FUNCTION public.gap_contradictions_for_ids(p_profile text, p_memory_ids uuid[], p_sample_size integer DEFAULT 10)
 RETURNS TABLE(in_result_id text, other_id text, strength double precision, total_count bigint)
 LANGUAGE sql
 STABLE
AS $function$
    WITH edges AS (
        SELECT
            CASE WHEN mr.source_id = ANY(p_memory_ids) THEN mr.source_id ELSE mr.target_id END AS in_id,
            CASE WHEN mr.source_id = ANY(p_memory_ids) THEN mr.target_id ELSE mr.source_id END AS other_id,
            mr.strength
        FROM memory_relationships mr
        JOIN memories ms ON ms.id = mr.source_id AND ms.profile = p_profile
        JOIN memories mt ON mt.id = mr.target_id AND mt.profile = p_profile
        WHERE mr.relationship = 'contradicts'
          AND (mr.source_id = ANY(p_memory_ids) OR mr.target_id = ANY(p_memory_ids))
          AND NOT (mr.source_id = ANY(p_memory_ids) AND mr.target_id = ANY(p_memory_ids))
    )
    SELECT in_id::text, other_id::text, strength, count(*) OVER () AS total_count
    FROM edges
    LIMIT p_sample_size;
$function$;

CREATE OR REPLACE FUNCTION public.hebbian_strengthen_edges(p_sources text[], p_targets text[], p_bootstrap real, p_rate real)
 RETURNS integer
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
DECLARE
    v_count integer;
BEGIN
    -- Caller is responsible for canonicalising pairs (sorted) -- see
    -- src/ogham/graph.py docstring for deadlock + idempotency rationale.
    WITH touched AS (
        INSERT INTO memory_relationships
            (source_id, target_id, relationship, strength, created_by)
        SELECT s::uuid, t::uuid, 'related', p_bootstrap, 'hebbian'
          FROM unnest(p_sources, p_targets) AS p(s, t)
        ON CONFLICT (source_id, target_id, relationship) DO UPDATE
            SET strength = LEAST(1.0,
                                 memory_relationships.strength * (1 + p_rate))
        RETURNING source_id
    )
    SELECT count(*)::integer INTO v_count FROM touched;
    RETURN v_count;
END;
$function$;

CREATE OR REPLACE FUNCTION public.entity_graph_density(p_profile text)
 RETURNS TABLE(entities double precision, edges double precision)
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
BEGIN
    IF to_regclass('public.memory_entities') IS NULL THEN
        RETURN QUERY SELECT 0.0::double precision, 0.0::double precision;
        RETURN;
    END IF;
    RETURN QUERY EXECUTE
        'SELECT
            count(DISTINCT entity_id)::double precision AS entities,
            count(*)::double precision                  AS edges
           FROM memory_entities
          WHERE profile = $1'
        USING p_profile;
END;
$function$;

CREATE OR REPLACE FUNCTION public.suggest_unlinked_by_shared_entities(p_memory_id uuid, p_profile text, p_min_shared integer, p_limit integer)
 RETURNS TABLE(id text, shared_count bigint, shared_entities text[], content text, created_at timestamp with time zone, tags text[])
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
BEGIN
    -- Same to_regclass guard as entity_graph_density. Both `entities`
    -- and `memory_entities` are needed; check the more-derived one.
    IF to_regclass('public.memory_entities') IS NULL
       OR to_regclass('public.entities') IS NULL THEN
        RETURN;
    END IF;
    RETURN QUERY EXECUTE
        'WITH target_entities AS (
            SELECT entity_id FROM memory_entities
             WHERE memory_id = $1
        ),
        shared AS (
            SELECT
                me.memory_id,
                count(*)::bigint AS shared_count,
                array_agg(e.entity_type || '':'' || e.canonical_name) AS shared_entities
              FROM memory_entities me
              JOIN target_entities te ON te.entity_id = me.entity_id
              JOIN entities e         ON e.id        = me.entity_id
             WHERE me.memory_id != $1
               AND me.profile    = $2
             GROUP BY me.memory_id
            HAVING count(*) >= $3
        ),
        unlinked AS (
            SELECT s.* FROM shared s
             WHERE NOT EXISTS (
                 SELECT 1 FROM memory_relationships mr
                  WHERE (mr.source_id = $1 AND mr.target_id = s.memory_id)
                     OR (mr.target_id = $1 AND mr.source_id = s.memory_id)
             )
        )
        SELECT
            u.memory_id::text AS id,
            u.shared_count,
            u.shared_entities,
            m.content,
            m.created_at,
            m.tags
          FROM unlinked u
          JOIN memories m ON m.id = u.memory_id
         WHERE m.expires_at IS NULL OR m.expires_at > now()
         ORDER BY u.shared_count DESC, m.created_at DESC
         LIMIT $4'
        USING p_memory_id, p_profile, p_min_shared, p_limit;
END;
$function$;


-- ── Triggers ─────────────────────────────────────────────────────────

DROP TRIGGER IF EXISTS memories_extract_occurrence ON memories;
CREATE TRIGGER memories_extract_occurrence BEFORE INSERT OR UPDATE ON public.memories FOR EACH ROW EXECUTE FUNCTION extract_occurrence_from_content();

DROP TRIGGER IF EXISTS topic_summaries_bump_updated_at ON topic_summaries;
CREATE TRIGGER topic_summaries_bump_updated_at BEFORE UPDATE ON public.topic_summaries FOR EACH ROW EXECUTE FUNCTION topic_summaries_set_updated_at();

-- ── RLS for topic summaries (migration 032, self-guarding on the anon role) ──
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- skipping RLS setup for topic_summaries '
            '+ topic_summary_sources (non-Supabase install)';
        RETURN;
    END IF;

    -- topic_summaries
    IF NOT (SELECT rowsecurity FROM pg_tables
              WHERE tablename = 'topic_summaries' AND schemaname = 'public') THEN
        EXECUTE 'ALTER TABLE topic_summaries ENABLE ROW LEVEL SECURITY';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_policy p
          JOIN pg_class c ON c.oid = p.polrelid
         WHERE c.relname = 'topic_summaries' AND p.polname = 'Deny anon access'
    ) THEN
        EXECUTE $policy$
            CREATE POLICY "Deny anon access" ON topic_summaries
                FOR ALL TO anon
                USING (false) WITH CHECK (false)
        $policy$;
    END IF;

    -- topic_summary_sources (FK junction table)
    IF NOT (SELECT rowsecurity FROM pg_tables
              WHERE tablename = 'topic_summary_sources' AND schemaname = 'public') THEN
        EXECUTE 'ALTER TABLE topic_summary_sources ENABLE ROW LEVEL SECURITY';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_policy p
          JOIN pg_class c ON c.oid = p.polrelid
         WHERE c.relname = 'topic_summary_sources' AND p.polname = 'Deny anon access'
    ) THEN
        EXECUTE $policy$
            CREATE POLICY "Deny anon access" ON topic_summary_sources
                FOR ALL TO anon
                USING (false) WITH CHECK (false)
        $policy$;
    END IF;
END$$;

COMMIT;

-- ── Grants for the lifecycle/graph RPCs (migration 035, self-guarding) ──
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.lifecycle_advance_fresh_to_stable(text, timestamptz, float, float) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.lifecycle_close_editing_windows(text, timestamptz) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.lifecycle_open_editing_window(uuid[]) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.lifecycle_pipeline_counts(text) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.hebbian_strengthen_edges(text[], text[], real, real) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.entity_graph_density(text) TO authenticated, service_role';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.suggest_unlinked_by_shared_entities(uuid, text, integer, integer) TO authenticated, service_role';
    END IF;
END
$$;

-- ── Revoke the REST surface on the SECURITY DEFINER RPCs (migration 037) ──
-- Migration 037 revokes ELEVEN functions; only the three above the entity-edge
-- section were ever backported, so a fresh install exposed the lifecycle and
-- graph RPCs where an upgraded install did not. Runs after the 035 grants for
-- the same reason 037 is numbered after 035: the grant comes first, then the
-- narrowing. Keep this list in step with migration 037.
REVOKE EXECUTE ON FUNCTION lifecycle_advance_fresh_to_stable(text, timestamptz, double precision, double precision) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION lifecycle_close_editing_windows(text, timestamptz) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION lifecycle_open_editing_window(uuid[]) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION lifecycle_pipeline_counts(text) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION hebbian_strengthen_edges(text[], text[], real, real) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION entity_graph_density(text) FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION suggest_unlinked_by_shared_entities(uuid, text, integer, integer) FROM anon, authenticated, PUBLIC;

-- ── Comment restored with the 047 backport ──────────────────────────────
COMMENT ON FUNCTION in_result_contradictions(p_profile text, p_memory_ids uuid[]) IS
    'Contradiction pairs with both endpoints inside a result set, oriented stale -> newer by created_at. Complements gap_contradictions_for_ids, which covers only pairs reaching outside the set. See TBU-207.';
