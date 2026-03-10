-- Migration: Add graph centrality boost to search scoring
-- Run in Supabase SQL Editor
-- Both match_memories and hybrid_search_memories get a LATERAL JOIN
-- against memory_relationships to boost well-connected memories.

-- Replace match_memories with graph centrality boost
create or replace function match_memories(
    query_embedding extensions.vector(768),
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
        (1 - (m.embedding <=> query_embedding))::float as similarity,
        -- Relevance = similarity * softplus(ACT-R) * confidence * graph_boost
        -- ACT-R: B(M) = ln(n+1) - 0.5 * ln(ageDays / (n+1))
        -- softplus: ln(1 + exp(B)) keeps score positive
        -- graph_boost: (1 + sum(relationship_strength) * 0.2)
        (
            (1 - (m.embedding <=> query_embedding)) *
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
        1 - (m.embedding <=> query_embedding) > match_threshold
        and (filter_tags is null or m.tags && filter_tags)
        and (filter_source is null or m.source = filter_source)
        and m.profile = filter_profile
        and (m.expires_at is null or m.expires_at > now())
    order by relevance desc
    limit match_count;
end;
$$;

-- Replace hybrid_search_memories with graph centrality boost
create or replace function hybrid_search_memories(
    query_text text,
    query_embedding extensions.vector(768),
    match_count int default 10,
    filter_profile text default 'default',
    filter_tags text[] default null,
    filter_source text default null,
    full_text_weight float default 1.0,
    semantic_weight float default 1.0,
    rrf_k int default 60
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    source text,
    profile text,
    tags text[],
    similarity float,
    keyword_rank float,
    relevance float,
    access_count integer,
    last_accessed_at timestamptz,
    confidence float,
    created_at timestamptz,
    updated_at timestamptz
)
language sql
security invoker
set search_path = public, extensions
as $$
with semantic as (
    select
        m.id,
        row_number() over (order by m.embedding <=> query_embedding) as rank_ix,
        (1 - (m.embedding <=> query_embedding))::float as similarity
    from memories m
    where m.profile = filter_profile
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by m.embedding <=> query_embedding
    limit match_count * 2
),
keyword as (
    select
        m.id,
        row_number() over (order by ts_rank_cd(m.fts, websearch_to_tsquery(query_text)) desc) as rank_ix,
        ts_rank_cd(m.fts, websearch_to_tsquery(query_text))::float as keyword_rank
    from memories m
    where m.profile = filter_profile
      and m.fts @@ websearch_to_tsquery(query_text)
      and (filter_tags is null or m.tags && filter_tags)
      and (filter_source is null or m.source = filter_source)
      and (m.expires_at is null or m.expires_at > now())
    order by keyword_rank desc
    limit match_count * 2
),
combined as (
    select id, rank_ix, similarity, 0.0::float as keyword_rank, 'semantic' as src
    from semantic
    union all
    select id, rank_ix, 0.0::float as similarity, keyword_rank, 'keyword' as src
    from keyword
),
fused as (
    select
        c.id,
        max(c.similarity) as similarity,
        max(c.keyword_rank) as keyword_rank,
        sum(
            case when c.src = 'semantic' then semantic_weight / (rrf_k + c.rank_ix)
                 when c.src = 'keyword'  then full_text_weight / (rrf_k + c.rank_ix)
                 else 0.0 end
        ) as rrf_score
    from combined c
    group by c.id
)
select
    m.id,
    m.content,
    m.metadata,
    m.source,
    m.profile,
    m.tags,
    f.similarity,
    f.keyword_rank,
    -- Relevance = rrf_score * softplus(ACT-R) * confidence * graph_boost
    -- graph_boost: (1 + sum(relationship_strength) * 0.2)
    (
        f.rrf_score
        * ln(1.0 + exp(
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
from fused f
join memories m on m.id = f.id
left join lateral (
    select coalesce(sum(r.strength), 0.0) as graph_boost
    from memory_relationships r
    where r.target_id = m.id or r.source_id = m.id
) g on true
order by relevance desc
limit match_count;
$$;
