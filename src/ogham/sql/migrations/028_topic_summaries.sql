-- sql/migrations/028_topic_summaries.sql
--
-- Migration 028: precomputed per-topic synthesized summaries ("compiled
-- wiki" cache layer). Implements the Karpathy-style "compile once, cheap
-- at retrieval" pattern without sacrificing Ogham's multi-agent safety.
--
-- Design decisions (see docs/plans/2026-04-23-wiki-hybrid-tier1-spec.md):
--   * Dedicated table (not shared with memories) -- avoids the Mem0
--     mistake where summary rows rank against raw facts in search.
--   * source_hash (BYTEA, sha256 of sorted source ids) + source_cursor
--     (UUID of newest included memory) detect both "new memories
--     arrived" (cursor moved) and "existing memories edited/deleted"
--     (hash changed). Zep's cursor + hash combined.
--   * version column for optimistic locking (Letta pattern).
--   * status: fresh | stale | regenerating. Avoids Letta #3270's
--     overwrite-without-fallback bug: regeneration runs under
--     'regenerating', atomic swap to 'fresh' on success, failure
--     leaves previous fresh row intact.
--   * topic_summary_sources junction with hard FK + reverse index on
--     memory_id enables the invalidation cascade in one SQL query.
--
-- Depends on: migration 026 (memory_lifecycle separate table) for
-- scoping consistency, but does NOT read or write memory_lifecycle
-- directly -- ingest triggers fire from application code.

BEGIN;

CREATE TABLE IF NOT EXISTS topic_summaries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_key text NOT NULL,
    profile_id text NOT NULL,
    content text NOT NULL,
    embedding vector(512),
    source_count integer NOT NULL,
    source_cursor uuid,
    source_hash bytea NOT NULL,
    token_count integer,
    importance double precision NOT NULL DEFAULT 0.5,
    model_used text NOT NULL,
    version integer NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'fresh',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    stale_reason text,
    CONSTRAINT topic_summaries_status_valid
        CHECK (status IN ('fresh', 'stale', 'regenerating')),
    CONSTRAINT topic_summaries_profile_topic_unique
        UNIQUE (profile_id, topic_key)
);

-- HNSW over the raw embedding column using plain vector_cosine_ops.
-- Matches the fresh-install memories pattern post-migration-023 which
-- explicitly dropped hardcoded halfvec(512) casts to keep the schema
-- dim-agnostic (auto-sizes to whatever EMBEDDING_DIM the column was
-- created at; non-512 operators can opt into halfvec via migration
-- 021's rebuild_hnsw=on path).
--
-- WHERE status = 'fresh' is the critical piece: migration 026 taught
-- us that updating a column on a row carrying an HNSW index forces a
-- tuple rewrite into the index. Our nightly sweep flips status
-- fresh -> stale on whole swaths of rows at once; without the partial
-- predicate those flips would each re-insert a vector entry. With it,
-- stale rows tombstone-out cheaply and are naturally excluded from
-- retrieval (which is what we want -- a stale summary shouldn't win
-- over raw memories anyway).
CREATE INDEX IF NOT EXISTS topic_summaries_embedding_hnsw_idx
    ON topic_summaries USING hnsw (embedding vector_cosine_ops)
    WITH (m = '16', ef_construction = '64')
    WHERE status = 'fresh';

-- Sweep / per-profile fresh-summary list. Partial on fresh because
-- stale rows are either about to be recompiled or dropped; we don't
-- page through them in normal UI / agent lookups.
CREATE INDEX IF NOT EXISTS topic_summaries_profile_fresh_idx
    ON topic_summaries (profile_id, updated_at DESC)
    WHERE status = 'fresh';

-- Stale-detection sweep. Dedicated partial index so the nightly
-- `UPDATE ... WHERE status='fresh' AND updated_at < N days ago` runs
-- an index range scan rather than a seq scan. Intentionally narrow --
-- only columns the sweep predicate uses.
CREATE INDEX IF NOT EXISTS topic_summaries_stale_sweep_idx
    ON topic_summaries (updated_at)
    WHERE status = 'fresh';

CREATE TABLE IF NOT EXISTS topic_summary_sources (
    summary_id uuid NOT NULL REFERENCES topic_summaries(id) ON DELETE CASCADE,
    memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (summary_id, memory_id)
);

-- Reverse-cascade index on memory_id: the hot-path lookup for
-- "which summaries are affected when memory X changed?". Without
-- this, invalidation is O(N) over all summaries. With it, O(log N).
CREATE INDEX IF NOT EXISTS topic_summary_sources_memory_id_idx
    ON topic_summary_sources (memory_id);

-- updated_at auto-bump. PostgreSQL does not touch updated_at on its
-- own; without this trigger the nightly stale-sweep logic would
-- compare against a timestamp that never moves.
-- SET search_path explicit so the Supabase linter ("Function Search
-- Path Mutable") stays clean and built-ins like now() can't be shadowed
-- by an attacker with CREATE on a shadowing schema.
CREATE OR REPLACE FUNCTION topic_summaries_set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS topic_summaries_bump_updated_at ON topic_summaries;
CREATE TRIGGER topic_summaries_bump_updated_at
    BEFORE UPDATE ON topic_summaries
    FOR EACH ROW
    EXECUTE FUNCTION topic_summaries_set_updated_at();

COMMIT;
