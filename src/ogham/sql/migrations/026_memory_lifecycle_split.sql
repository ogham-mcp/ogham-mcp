-- src/ogham/sql/migrations/026_memory_lifecycle_split.sql
--
-- Migration 026: split memory lifecycle state to a separate table.
--
-- Reason: updates to memories.stage / stage_entered_at break HOT updates
-- and force tuple rewrites into the 512-dim HNSW index. At search volume
-- this causes catastrophic index bloat and autovacuum pressure.
--
-- This migration moves lifecycle state to memory_lifecycle (keyed 1:1 to
-- memories). Memories table becomes read-mostly for retrieval; HNSW is
-- untouched by lifecycle transitions.
--
-- Depends on: migration 025 (which added memories.stage / stage_entered_at).

BEGIN;

CREATE TABLE IF NOT EXISTS memory_lifecycle (
    memory_id uuid PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    profile text NOT NULL,
    stage text NOT NULL DEFAULT 'fresh',
    stage_entered_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT memory_lifecycle_stage_valid CHECK (stage IN ('fresh', 'stable', 'editing'))
);

-- Partial index for sweeps (advance_stages, close_editing_windows).
-- Partial predicate already filters on stage, so we don't need it in the
-- indexed columns -- Gemini 3.1 Pro review.
CREATE INDEX IF NOT EXISTS memory_lifecycle_transitioning_idx
    ON memory_lifecycle (profile, stage_entered_at)
    WHERE stage IN ('fresh', 'editing');

-- Full index for lifecycle_pipeline_counts which groups across all stages.
CREATE INDEX IF NOT EXISTS memory_lifecycle_profile_stage_idx
    ON memory_lifecycle (profile, stage);

-- Backfill from memories.stage/stage_entered_at (025 columns) if they exist.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memories' AND column_name = 'stage'
    ) THEN
        INSERT INTO memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
        SELECT id, profile, stage, stage_entered_at, stage_entered_at
          FROM memories
        ON CONFLICT (memory_id) DO NOTHING;
    ELSE
        -- 025 wasn't applied -- backfill all memories as fresh.
        INSERT INTO memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
        SELECT id, profile, 'fresh', created_at, created_at
          FROM memories
        ON CONFLICT (memory_id) DO NOTHING;
    END IF;
END$$;

-- Drop the old composite index on memories before dropping the columns.
DROP INDEX IF EXISTS memories_stage_idx;

-- Drop 025's columns + constraint from memories.
ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_stage_valid,
    DROP COLUMN IF EXISTS stage,
    DROP COLUMN IF EXISTS stage_entered_at;

-- Trigger: auto-init a lifecycle row when a new memory is inserted.
CREATE OR REPLACE FUNCTION init_memory_lifecycle() RETURNS trigger AS $$
BEGIN
    INSERT INTO memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
    VALUES (NEW.id, NEW.profile, 'fresh', NEW.created_at, NEW.created_at)
    ON CONFLICT (memory_id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_init_lifecycle ON memories;
CREATE TRIGGER memories_init_lifecycle
    AFTER INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION init_memory_lifecycle();

-- Trigger: keep memory_lifecycle.profile in sync when a memory is moved
-- between profiles (rare but possible). Same search_path hardening as
-- init_memory_lifecycle above.
CREATE OR REPLACE FUNCTION sync_memory_lifecycle_profile() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    IF NEW.profile IS DISTINCT FROM OLD.profile THEN
        UPDATE memory_lifecycle
           SET profile = NEW.profile, updated_at = now()
         WHERE memory_id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS memories_sync_lifecycle_profile ON memories;
CREATE TRIGGER memories_sync_lifecycle_profile
    AFTER UPDATE OF profile ON memories
    FOR EACH ROW
    EXECUTE FUNCTION sync_memory_lifecycle_profile();

COMMIT;
