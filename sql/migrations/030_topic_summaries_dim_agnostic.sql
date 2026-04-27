-- Migration 030: align topic_summaries.embedding dim with memories.embedding.
--
-- Migration 028 hardcoded `embedding vector(512)` on topic_summaries because
-- the design doc was written when memories.embedding was still 512. Self-
-- hosters running at any other dim (768 = Gemini, 1024 = Mistral) hit:
--
--    psycopg.errors.DataException: expected 512 dimensions, not 768
--
-- whenever the recompute pipeline tries to upsert a synthesized summary.
-- Symptom surfaces only when wiki Tier 1 is exercised (compile_wiki tool
-- or background recompute hooks), so it lay dormant through v0.11.x.
--
-- Fix: rebuild topic_summaries.embedding with the same dim as
-- memories.embedding. We read the live memories.embedding type rather
-- than embedding a hardcoded constant, so this migration is correct for
-- all dim combos (512 / 768 / 1024) without a config knob.
--
-- HNSW index is dropped first (vector_cosine_ops won't span dims) and
-- recreated against the new column with the same partial predicate
-- (status = 'fresh') from migration 028.
--
-- Data loss: the embedding column is wiped. Existing summaries keep
-- their content + frontmatter + source links, but get embedding=NULL,
-- which `search_summaries` already filters out (`embedding IS NOT NULL`).
-- Next recompute (executor or compile_wiki) writes the new dim.
--
-- Idempotent: re-running is safe -- if memories.embedding and
-- topic_summaries.embedding are already the same dim, the migration
-- short-circuits without touching either column.

BEGIN;

DO $$
DECLARE
    memories_dim integer;
    summaries_dim integer;
BEGIN
    -- Bail cleanly if topic_summaries hasn't been created yet (migration
    -- 028 is the prerequisite). Apply 028 first, then re-run 030.
    -- Without this guard, the ALTER TABLE below would fail with a
    -- 42P01 "relation does not exist" mid-transaction.
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
         JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'topic_summaries' AND n.nspname = 'public'
    ) THEN
        RAISE NOTICE
            'topic_summaries table not found -- migration 028 must run first. '
            'Apply sql/migrations/028_topic_summaries.sql, then re-run 030.';
        RETURN;
    END IF;

    -- Read the live dim from memories.embedding. atttypmod for vector is
    -- the dimension; subtract VARHDRSZ-1 (psql's "raw modifier") for the
    -- conventional reading. pgvector stores dim directly in atttypmod, so
    -- the value is read as-is.
    SELECT atttypmod INTO memories_dim
      FROM pg_attribute a
      JOIN pg_class c ON c.oid = a.attrelid
     WHERE c.relname = 'memories' AND a.attname = 'embedding';

    SELECT atttypmod INTO summaries_dim
      FROM pg_attribute a
      JOIN pg_class c ON c.oid = a.attrelid
     WHERE c.relname = 'topic_summaries' AND a.attname = 'embedding';

    IF memories_dim IS NULL THEN
        RAISE NOTICE 'memories.embedding not found — skipping topic_summaries dim alignment';
        RETURN;
    END IF;

    IF summaries_dim = memories_dim THEN
        RAISE NOTICE
            'topic_summaries.embedding already at dim %, no-op',
            memories_dim;
        RETURN;
    END IF;

    RAISE NOTICE
        'Realigning topic_summaries.embedding from dim % to dim % (matching memories.embedding)',
        summaries_dim, memories_dim;

    -- 1. Drop the HNSW index (vector_cosine_ops won't span dims).
    EXECUTE 'DROP INDEX IF EXISTS topic_summaries_embedding_hnsw_idx';

    -- 2. Recreate the column with the new dim. Existing rows lose their
    --    embedding (cleared to NULL); content + provenance preserved.
    EXECUTE 'ALTER TABLE topic_summaries DROP COLUMN embedding';
    EXECUTE format('ALTER TABLE topic_summaries ADD COLUMN embedding vector(%s)', memories_dim);

    -- 3. Recreate the partial HNSW index from migration 028.
    EXECUTE 'CREATE INDEX IF NOT EXISTS topic_summaries_embedding_hnsw_idx '
            'ON topic_summaries USING hnsw (embedding vector_cosine_ops) '
            'WITH (m = 16, ef_construction = 64) '
            'WHERE status = ''fresh''';
END $$;

COMMIT;
