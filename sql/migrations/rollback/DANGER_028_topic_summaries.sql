-- sql/migrations/rollback/DANGER_028_topic_summaries.sql
--
-- ┌─────────────────────────────────────────────────────────────────────┐
-- │ DANGER: ROLLBACK MIGRATION                                          │
-- │                                                                     │
-- │ This script DROPs topic_summaries and topic_summary_sources and    │
-- │ restores the pre-028 state. Any compiled-wiki caches are lost --   │
-- │ they will be regenerated on next retrieval, but the backlog of     │
-- │ not-yet-recompiled source edits is also lost.                       │
-- │                                                                     │
-- │ Rollback is for development / recovery, not for routine operation. │
-- │ The Python MCP server v0.11.1+ expects topic_summaries to exist    │
-- │ and will fail after this rollback unless you also downgrade the    │
-- │ server to v0.11.0 or older.                                         │
-- │                                                                     │
-- │ To run this intentionally:                                          │
-- │   psql $DATABASE_URL <<'EOF'                                        │
-- │   SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';           │
-- │   \i sql/migrations/rollback/DANGER_028_topic_summaries.sql         │
-- │   EOF                                                               │
-- │                                                                     │
-- │ Piping this file naively (psql $URL < this_file) will FAIL by      │
-- │ design -- the session variable is checked before anything else.    │
-- └─────────────────────────────────────────────────────────────────────┘

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION USING
            MESSAGE = 'Rollback refused -- explicit opt-in required.',
            HINT = 'Run "SET ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'';" in the same session before this script. See file header for details.';
    END IF;
END$$;

BEGIN;

-- Migration 031 wiki RPC functions hard-depend on the topic_summaries
-- relation (RETURNS topic_summaries / RETURNS SETOF topic_summaries).
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
