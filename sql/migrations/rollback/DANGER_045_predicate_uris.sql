-- sql/migrations/rollback/DANGER_045_predicate_uris.sql
--
-- Rollback of migration 045_predicate_uris.sql.
-- DANGER: DROPs ogham_uri / schema_org_uri / iirds_uri and destroys the
-- predicate URI data. Development / recovery use only.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_045_predicate_uris.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_045 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

ALTER TABLE entity_edge_predicates DROP COLUMN IF EXISTS ogham_uri;
ALTER TABLE entity_edge_predicates DROP COLUMN IF EXISTS schema_org_uri;
ALTER TABLE entity_edge_predicates DROP COLUMN IF EXISTS iirds_uri;

COMMIT;
