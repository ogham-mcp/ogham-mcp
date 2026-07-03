-- sql/migrations/rollback/DANGER_043_entity_aliases.sql
--
-- Rollback of migration 043_entity_aliases.sql.
-- DANGER: DROPs entity_aliases and destroys all alias mappings.
-- Development / recovery use only.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_043_entity_aliases.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_043 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DROP INDEX IF EXISTS entity_aliases_entity;
DROP TABLE IF EXISTS entity_aliases;

COMMIT;
