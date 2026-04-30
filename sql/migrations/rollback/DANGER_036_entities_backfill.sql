-- DANGER_036: rollback of migration 036_entities_backfill.sql
--
-- Drops the entities + memory_entities tables and their helper RPC
-- functions. DESTRUCTIVE -- removes all entity-graph data on the
-- deployment. Memory rows themselves are unaffected.
--
-- This file follows the v0.12 DANGER_* convention:
--   * Filename prefix DANGER_ flags it as destructive
--   * Session-variable guard inside the BEGIN transaction (per the v0.13
--     test_danger_rollback_guards_live_inside_transaction test pattern)
--   * The harness apply_rollback() helper sets ogham.confirm_rollback
--     before executing
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_036_entities_backfill.sql

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_036 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DROP FUNCTION IF EXISTS public.link_memory_entities(uuid, text, text[]) CASCADE;
DROP FUNCTION IF EXISTS public.spread_entity_activation_memories(text[], text, int, float, float, int) CASCADE;
DROP FUNCTION IF EXISTS public.refresh_entity_temporal_span(bigint) CASCADE;
DROP TABLE IF EXISTS public.memory_entities CASCADE;
DROP TABLE IF EXISTS public.entities CASCADE;

COMMIT;
