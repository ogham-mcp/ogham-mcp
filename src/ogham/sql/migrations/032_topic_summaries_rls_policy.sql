-- sql/migrations/032_topic_summaries_rls_policy.sql
--
-- Wiki layer RLS hygiene. Migration 028 created topic_summaries +
-- topic_summary_sources without explicit RLS setup. On Supabase, RLS
-- gets enabled implicitly on any public-schema table, but no policy
-- attached means PostgREST denies all anon traffic AND Supabase's
-- linter flags the table as "RLS enabled, no policies" (lint code
-- 0008_rls_enabled_no_policy).
--
-- Mirrors the pattern in migration 027 (audit_log) and the existing
-- "Deny anon access" policies on memories / memory_relationships /
-- memory_lifecycle / entities / memory_entities. The Python server
-- writes via the secret_key (service_role) which bypasses RLS, so
-- functional behaviour is unchanged.
--
-- Self-hosters on vanilla Postgres (no `anon` role) get a NOTICE and
-- the migration no-ops -- safe to apply against any backend.
--
-- Idempotent. Apply order: 028 -> 030 -> 031 -> 032 (this).

BEGIN;
SET LOCAL search_path = public, pg_catalog;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- skipping RLS setup for topic_summaries '
            '+ topic_summary_sources (non-Supabase install)';
        RETURN;
    END IF;

    -- topic_summaries
    IF NOT (SELECT rowsecurity FROM pg_tables
              WHERE tablename = 'topic_summaries' AND schemaname = 'public') THEN
        EXECUTE 'ALTER TABLE topic_summaries ENABLE ROW LEVEL SECURITY';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_policy p
          JOIN pg_class c ON c.oid = p.polrelid
         WHERE c.relname = 'topic_summaries' AND p.polname = 'Deny anon access'
    ) THEN
        EXECUTE $policy$
            CREATE POLICY "Deny anon access" ON topic_summaries
                FOR ALL TO anon
                USING (false) WITH CHECK (false)
        $policy$;
    END IF;

    -- topic_summary_sources (FK junction table)
    IF NOT (SELECT rowsecurity FROM pg_tables
              WHERE tablename = 'topic_summary_sources' AND schemaname = 'public') THEN
        EXECUTE 'ALTER TABLE topic_summary_sources ENABLE ROW LEVEL SECURITY';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_policy p
          JOIN pg_class c ON c.oid = p.polrelid
         WHERE c.relname = 'topic_summary_sources' AND p.polname = 'Deny anon access'
    ) THEN
        EXECUTE $policy$
            CREATE POLICY "Deny anon access" ON topic_summary_sources
                FOR ALL TO anon
                USING (false) WITH CHECK (false)
        $policy$;
    END IF;
END$$;

COMMIT;
