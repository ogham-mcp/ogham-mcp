-- Batch duplicate checking for import dedup.
-- Accepts an array of embeddings and returns a boolean array
-- indicating whether each embedding has a match above threshold.
-- Uses a simple cosine similarity check (no ACT-R scoring needed for dedup).

create or replace function batch_check_duplicates(
    query_embeddings extensions.vector(768)[],
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
              and 1 - (m.embedding <=> query_embeddings[i]) > match_threshold
            limit 1
        ) into found;
        results := array_append(results, found);
    end loop;
    return results;
end;
$$;
