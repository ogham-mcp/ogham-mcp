-- Migration: Confidence Scoring
-- Run this in the Supabase SQL Editor against an existing Ogham database.
-- Safe to re-run (all statements are idempotent).

-- ============================================================
-- Add confidence column
-- ============================================================
alter table memories add column if not exists confidence float not null default 0.5;

-- ============================================================
-- Updated match_memories with confidence weighting
-- Must drop first: return type changed (added confidence column)
-- ============================================================
drop function if exists match_memories(extensions.vector, float, int, text[], text, text);

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
        -- Relevance = similarity * softplus(ACT-R base-level activation) * confidence
        -- ACT-R: B(M) = ln(n+1) - 0.5 * ln(ageDays / (n+1))
        -- softplus: ln(1 + exp(B)) keeps score positive
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
        )::float as relevance,
        m.access_count,
        m.last_accessed_at,
        m.confidence,
        m.created_at,
        m.updated_at
    from public.memories m
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

-- ============================================================
-- New RPC: Bayesian confidence update
-- signal: 0.85 = reinforce, 0.15 = contradict, 0.5 = neutral
-- ============================================================
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
