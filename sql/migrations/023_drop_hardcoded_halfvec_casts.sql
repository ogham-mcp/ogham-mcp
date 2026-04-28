-- Migration 023: drop hardcoded halfvec(512) casts from the hybrid search
-- family so fresh installs are dim-agnostic out of the box.
--
-- Background
-- ----------
-- Migrations 013/017/018/021 carried `::halfvec(N)` casts inside function
-- bodies to steer queries onto a halfvec-typed HNSW index. That was 2x
-- faster and 2x smaller at scale, but baked a specific EMBEDDING_DIM
-- into the schema -- every dim change needed migration 021 re-run with
-- ogham.rebuild_hnsw='on' to realign both the RPCs and the index.
--
-- For v0.5.1 the default is a plain `vector_cosine_ops` HNSW index +
-- plain `m.embedding <=> query_embedding` operators in every RPC. The
-- index type auto-sizes to whatever EMBEDDING_DIM the column was
-- created with; no manual dim tracking required.
--
-- Perf trade: the halfvec index was ~2x smaller in memory for big
-- corpora. Operators at >100K rows who want halfvec memory savings
-- back: migration 021 still works -- run it with ogham.rebuild_hnsw='on'
-- after this migration.
--
-- Idempotent. Safe no-op on installs that never had halfvec casts.

begin;

-- 1. Drop and recreate the HNSW index as a plain-vector cosine index.
drop index if exists memories_embedding_idx;
create index memories_embedding_idx
    on memories using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- 2. Re-create the RPC family with halfvec casts removed. Each function
-- below is the same body that lives in src/ogham/sql/schema_postgres.sql
-- after the v0.5.1 rewrite.

-- 2a. auto_link_memory
create or replace function auto_link_memory(
    new_memory_id uuid,
    new_embedding vector(512),
    filter_profile text default 'default',
    link_threshold float default 0.85,
    top_n int default 3
)
returns int
language plpgsql
security invoker
set search_path = public, extensions
as $$
declare
    linked_count int := 0;
begin
    insert into memory_relationships (source_id, target_id, relationship_type, strength, metadata)
    select
        new_memory_id,
        id,
        'auto-link',
        similarity,
        jsonb_build_object('similarity', similarity)
    from (
        SELECT m.id, (1 - (m.embedding <=> new_embedding))::float AS similarity
        FROM memories m
        WHERE m.id <> new_memory_id
          AND m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND 1 - (m.embedding <=> new_embedding) > link_threshold
        ORDER BY m.embedding <=> new_embedding
        LIMIT top_n
    ) candidates
    on conflict do nothing;

    get diagnostics linked_count = row_count;
    return linked_count;
end;
$$;

-- 2b. hybrid_search_memories -- the primary retrieval RPC
create or replace function hybrid_search_memories(
    query_text text,
    query_embedding vector,
    match_count integer default 10,
    filter_profile text default 'default',
    filter_tags text[] default null,
    filter_source text default null,
    full_text_weight float default 0.3,
    semantic_weight float default 0.7,
    rrf_k integer default 10,
    filter_profiles text[] default null,
    query_entity_tags text[] default null,
    recency_decay float default 0.0
)
returns table(
    id uuid, content text, metadata jsonb, source text, profile text, tags text[],
    similarity float, keyword_rank float, relevance float,
    access_count integer, last_accessed_at timestamptz, confidence float,
    created_at timestamptz, updated_at timestamptz
)
language sql
set search_path = public, extensions
as $function$
with semantic as (
    select
        m.id,
        (1 - (m.embedding <=> query_embedding))::float as similarity,
        row_number() over (order by m.embedding <=> query_embedding) as rank_ix
    from memories m
    where (filter_profiles is not null and m.profile = any(filter_profiles)
           or filter_profiles is null and m.profile = filter_profile)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by m.embedding <=> query_embedding
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

-- 2c. match_memories -- legacy single-pass similarity RPC
create or replace function match_memories(
    query_embedding vector,
    match_threshold float default 0.7,
    match_count int default 10,
    filter_tags text[] default null,
    filter_source text default null,
    filter_profile text default 'default'
)
returns table (
    id uuid, content text, metadata jsonb, source text, profile text, tags text[],
    similarity float, relevance float,
    access_count int, last_accessed_at timestamptz, confidence float,
    created_at timestamptz, updated_at timestamptz
)
language sql
set search_path = public, extensions
as $$
    select
        m.id, m.content, m.metadata, m.source, m.profile, m.tags,
        (1 - (m.embedding <=> query_embedding))::float as similarity,
        (
            (1 - (m.embedding <=> query_embedding)) *
            (1.0 + ln(m.access_count + 1.0) * 0.1) *
            m.confidence
        )::float as relevance,
        m.access_count, m.last_accessed_at, m.confidence, m.created_at, m.updated_at
    from memories m
    where (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and m.profile = filter_profile
      and (m.expires_at is null or m.expires_at > now())
      and 1 - (m.embedding <=> query_embedding) > match_threshold
    order by relevance desc
    limit match_count;
$$;

-- 2d. batch_check_duplicates -- ingest-path dedup
create or replace function batch_check_duplicates(
    query_embeddings vector[],
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
    perform set_config('hnsw.ef_search', '40', true);
    results := array[]::boolean[];
    for i in 1..array_length(query_embeddings, 1) loop
        select exists(
            select 1 from memories m
            where m.profile = filter_profile
              and (m.expires_at is null or m.expires_at > now())
              and 1 - (m.embedding <=> query_embeddings[i]) > match_threshold
            limit 1
        ) into found;
        results := array_append(results, found);
    end loop;
    return results;
end;
$$;

commit;

-- Post-migration smoke: if EMBEDDING_DIM != 512 was configured, migration
-- 021_dim_aware_halfvec.sql can still be re-run on top of this to put
-- the halfvec optimization back with the right dim. The two migrations
-- are compatible in either order.
