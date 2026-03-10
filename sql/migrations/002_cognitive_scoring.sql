-- Migration: Cognitive Scoring & Supabase Linter Fixes
-- Run this in the Supabase SQL Editor against an existing Ogham database.
-- Safe to re-run (all statements are idempotent).

-- ============================================================
-- Fix 1: Add search_path to update_updated_at trigger function
-- ============================================================
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

-- ============================================================
-- Fix 2-3: Drop overly permissive RLS policies
-- ============================================================
drop policy if exists "Allow all for authenticated" on memories;
drop policy if exists "Allow all for authenticated" on profile_settings;

-- ============================================================
-- Add access tracking columns
-- ============================================================
alter table memories add column if not exists access_count integer not null default 0;
alter table memories add column if not exists last_accessed_at timestamptz;

-- ============================================================
-- Updated match_memories with ACT-R temporal scoring
-- Must drop first: return type changed (added relevance, access_count, last_accessed_at)
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
        )::float as relevance,
        m.access_count,
        m.last_accessed_at,
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
-- New RPC: record access for search results
-- ============================================================
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
