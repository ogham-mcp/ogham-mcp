-- src/ogham/sql/migrations/028_topic_summaries_rollback.sql
--
-- Test-harness rollback for migration 028. Drops topic_summary_sources
-- and topic_summaries. No session-variable guard -- that lives on the
-- canonical DANGER_028 copy under sql/migrations/rollback/ and is the
-- only path intended for production use.

BEGIN;

-- Migration 031 wiki RPC functions hard-depend on the topic_summaries
-- relation (`RETURNS topic_summaries`, `RETURNS SETOF topic_summaries`).
-- Drop them first so the table drop doesn't fail with
-- DependentObjectsStillExist. Loop over pg_proc rather than spelling
-- each signature -- arg lists drift over time and explicit DROPs go
-- stale silently.
DO $$
DECLARE
    fn record;
BEGIN
    FOR fn IN
        SELECT n.nspname, p.proname, oidvectortypes(p.proargtypes) AS args
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND p.proname LIKE 'wiki\_%' ESCAPE '\'
    LOOP
        EXECUTE format(
            'DROP FUNCTION IF EXISTS %I.%I(%s)',
            fn.nspname, fn.proname, fn.args
        );
    END LOOP;
END $$;

DROP TRIGGER IF EXISTS topic_summaries_bump_updated_at ON topic_summaries;
DROP FUNCTION IF EXISTS topic_summaries_set_updated_at();
DROP TABLE IF EXISTS topic_summary_sources;
DROP TABLE IF EXISTS topic_summaries;

COMMIT;
