-- Replace PL/pgSQL loop with set-based UPDATE for batch_update_embeddings.
-- Single statement instead of N individual UPDATEs.

create or replace function batch_update_embeddings(
    memory_ids uuid[],
    new_embeddings extensions.vector(768)[]
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
