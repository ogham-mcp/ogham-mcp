-- Migration 037: revoke anon + authenticated EXECUTE on SECURITY DEFINER RPCs
--
-- WHY THIS MIGRATION EXISTS
-- The Supabase database linter flagged 10 SECURITY DEFINER functions in
-- the public schema as callable by `anon` and `authenticated` roles via
-- PostgREST. Because the functions run with the definer's privileges
-- they bypass RLS -- anyone with the project's anon key (which is
-- public by design) could pollute the entity graph, force-promote
-- lifecycle stages, or fingerprint shared-entity neighbours of any
-- memory.
--
-- These functions are infrastructure used by the Ogham server (which
-- holds the service_role / sb_secret_ key). They were never intended
-- to be part of the public REST surface.
--
-- WHAT IT DOES
-- REVOKE EXECUTE on each function from `anon` and `authenticated`.
-- service_role retains EXECUTE because PostgreSQL's default grant
-- chain includes it. Idempotent: REVOKE on a non-existent grant is a
-- no-op.
--
-- COVERAGE
-- v0.13.1 (Hotfix A, migration 035):
--   * lifecycle_advance_fresh_to_stable
--   * lifecycle_close_editing_windows
--   * lifecycle_open_editing_window
--   * lifecycle_pipeline_counts
--   * hebbian_strengthen_edges
--   * entity_graph_density
--   * suggest_unlinked_by_shared_entities
--
-- v0.14 (migration 036):
--   * link_memory_entities
--   * refresh_entity_temporal_span
--   * spread_entity_activation_memories
--
-- VERIFICATION
-- After applying, the Supabase linter
-- (https://supabase.com/docs/guides/database/database-linter)
-- should clear all 20 warnings (10 functions x 2 roles). To verify
-- locally:
--
--   SELECT routine_name, grantee, privilege_type
--     FROM information_schema.routine_privileges
--    WHERE routine_schema = 'public'
--      AND grantee IN ('anon','authenticated','PUBLIC')
--      AND routine_name IN (...);
--
-- Should return zero rows.

-- Self-hosters on vanilla Postgres (no `anon`/`authenticated` roles) get
-- a NOTICE and the migration no-ops. PUBLIC always exists, so the
-- REVOKE FROM PUBLIC is unconditional.

DO $$
BEGIN
    -- PUBLIC always exists -- safe to REVOKE on every install.
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_advance_fresh_to_stable(text, timestamptz, double precision, double precision) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_close_editing_windows(text, timestamptz) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_open_editing_window(uuid[]) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_pipeline_counts(text) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.hebbian_strengthen_edges(text[], text[], real, real) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.entity_graph_density(text) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.suggest_unlinked_by_shared_entities(uuid, text, integer, integer) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.link_memory_entities(uuid, text, text[]) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.refresh_entity_temporal_span(bigint) FROM PUBLIC';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.spread_entity_activation_memories(text[], text, integer, double precision, double precision, integer) FROM PUBLIC';

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon/authenticated roles not found -- skipping role-specific '
            'REVOKE EXECUTE (non-Supabase install). PUBLIC revokes already applied.';
        RETURN;
    END IF;

    -- Migration 035 functions (v0.13.1)
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_advance_fresh_to_stable(text, timestamptz, double precision, double precision) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_close_editing_windows(text, timestamptz) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_open_editing_window(uuid[]) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.lifecycle_pipeline_counts(text) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.hebbian_strengthen_edges(text[], text[], real, real) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.entity_graph_density(text) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.suggest_unlinked_by_shared_entities(uuid, text, integer, integer) FROM anon, authenticated';

    -- Migration 036 functions (v0.14)
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.link_memory_entities(uuid, text, text[]) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.refresh_entity_temporal_span(bigint) FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION public.spread_entity_activation_memories(text[], text, integer, double precision, double precision, integer) FROM anon, authenticated';
END $$;
