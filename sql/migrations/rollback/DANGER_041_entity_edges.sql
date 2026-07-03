-- sql/migrations/rollback/DANGER_041_entity_edges.sql
--
-- Rollback of migration 041_entity_edges.sql.
-- DANGER: DROPs entity_edges and destroys all typed entity-edge data,
-- including superseded history. Development / recovery use only.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_041_entity_edges.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_041 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DROP INDEX IF EXISTS entity_edges_profile_current;
DROP INDEX IF EXISTS entity_edges_object_pred_current;
DROP INDEX IF EXISTS entity_edges_subject_pred_current;
DROP INDEX IF EXISTS entity_edges_current_uq;
DROP TABLE IF EXISTS entity_edges;

COMMIT;
