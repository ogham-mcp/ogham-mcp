-- sql/migrations/044_unforce_rls_non_supabase.sql
--
-- Migration 044: self-heal the RLS FORCE lockout on non-Supabase installs
-- (TBU-163).
--
-- Migration 036 (entities, memory_entities, v0.14) applied FORCE ROW LEVEL
-- SECURITY unconditionally, before the anon-guard fix landed for 041-043.
-- On a non-Supabase install (no `anon` role) the anon-scoped policies are
-- never created, so FORCE + no policy = deny-all for the non-superuser
-- table owner. Editing 036 in place would not help already-applied
-- installs (they won't re-run a migration they've already applied), so
-- this is a dynamic self-heal that finds every already-forced table on
-- the current install and un-forces it -- covering 036's tables, any
-- table forced by a pre-fix run of 041-043, and any future migration that
-- makes the same mistake.
--
-- No-op on Supabase (the `anon` role is present; RLS policies govern
-- access there, so FORCE stays in place). Idempotent -- re-running finds
-- no forced tables left to un-force.
--
-- Design: docs/superpowers/specs/2026-07-01-typed-edge-context-graph-design.md
-- Rollback: sql/migrations/rollback/DANGER_044_unforce_rls_non_supabase.sql
--   (re-applying FORCE re-introduces the lockout -- see that file's header)

DO $$
DECLARE
    r record;
    n int := 0;
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE 'anon role present (Supabase install) -- leaving FORCE ROW LEVEL SECURITY intact; policies govern access';
        RETURN;
    END IF;
    FOR r IN
        SELECT relname FROM pg_class
        WHERE relkind = 'r'
          AND relforcerowsecurity
          AND relnamespace = 'public'::regnamespace
    LOOP
        EXECUTE format('ALTER TABLE public.%I NO FORCE ROW LEVEL SECURITY', r.relname);
        n := n + 1;
        RAISE NOTICE 'un-forced RLS on % (non-Supabase install; owner/app retains access)', r.relname;
    END LOOP;
    RAISE NOTICE 'TBU-163 self-heal: un-forced % table(s)', n;
END $$;
