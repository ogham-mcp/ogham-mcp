-- sql/migrations/rollback/DANGER_046_edge_provenance.sql
--
-- Rollback of migration 046_edge_provenance.sql.
-- DANGER: DROPs the derived_from column and destroys provenance-lineage
-- data. Development / recovery use only.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_046_edge_provenance.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_046 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DROP INDEX IF EXISTS entity_edges_derived_from_gin;
ALTER TABLE entity_edges DROP COLUMN IF EXISTS derived_from;

COMMIT;
