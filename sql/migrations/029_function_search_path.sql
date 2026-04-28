-- sql/migrations/029_function_search_path.sql
--
-- Harden trigger functions from migrations 026 + 028 against the
-- "Function Search Path Mutable" warning Supabase raises (and the real
-- underlying security risk: a user with CREATE on any schema on the
-- function's search_path can shadow built-ins and hijack the trigger
-- -- e.g. redefine `now()`, intercept `INSERT INTO memory_lifecycle`,
-- etc.).
--
-- Fix: explicit SET search_path = public, pg_catalog on each function.
-- pg_catalog is included so built-ins resolve through the trusted
-- catalog path even if an attacker controls schema search order.
--
-- Three functions are patched:
--   - init_memory_lifecycle (026)              trigger on memories INSERT
--   - sync_memory_lifecycle_profile (026)      trigger on memories UPDATE OF profile
--   - topic_summaries_set_updated_at (028)     trigger on topic_summaries UPDATE
--
-- Idempotent. Safe to re-run. No data migration.

BEGIN;

CREATE OR REPLACE FUNCTION init_memory_lifecycle() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    INSERT INTO memory_lifecycle (memory_id, profile, stage, stage_entered_at, updated_at)
    VALUES (NEW.id, NEW.profile, 'fresh', NEW.created_at, NEW.created_at)
    ON CONFLICT (memory_id) DO NOTHING;
    RETURN NEW;
END;
$$;

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

-- 028's updated_at trigger. Fine-grained: any UPDATE to topic_summaries
-- bumps updated_at via this BEFORE trigger.
CREATE OR REPLACE FUNCTION topic_summaries_set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path = public, pg_catalog
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

COMMIT;
