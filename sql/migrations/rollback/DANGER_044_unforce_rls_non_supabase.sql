-- sql/migrations/rollback/DANGER_044_unforce_rls_non_supabase.sql
--
-- Rollback of migration 044_unforce_rls_non_supabase.sql.
--
-- DANGER: this re-applies FORCE ROW LEVEL SECURITY to the tables 044 is
-- known to un-force (entities, memory_entities, entity_edges,
-- entity_edge_predicates, entity_aliases). On a non-Supabase install
-- (no `anon` role, no Deny-anon policies) that is EXACTLY the TBU-163
-- lockout 044 exists to fix -- re-running this will lock the
-- non-superuser table owner out of those tables again. There is no
-- legitimate reason to run this outside of testing the forward/rollback
-- migration harness itself. Genuinely dangerous / rarely wanted.
--
-- Note this is necessarily approximate, not an exact inverse: 044 is a
-- dynamic self-heal (it discovers and un-forces whatever public tables
-- happen to be forced at run time), so there is no recorded list of what
-- it actually touched on a given install. This rollback re-forces the
-- known table set that 036/041-043 originally forced -- it will not
-- re-force any other table 044 may have un-forced on an install with
-- custom RLS usage beyond ogham's own schema.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_044_unforce_rls_non_supabase.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_044 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'entities', 'memory_entities',
        'entity_edges', 'entity_edge_predicates', 'entity_aliases'
    ]
    LOOP
        IF to_regclass('public.' || t) IS NOT NULL THEN
            EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY', t);
        END IF;
    END LOOP;
END
$$;

COMMIT;
