-- sql/migrations/rollback/DANGER_042_entity_edge_predicates.sql
--
-- Rollback of migration 042_entity_edge_predicates.sql.
-- DANGER: DROPs entity_edge_predicates and destroys the controlled
-- predicate vocabulary. Any entity_edges rows referencing these
-- predicates by string will orphan (predicate is text, not an FK, so
-- this does not cascade -- existing edges simply reference a value that
-- no longer validates against the vocab table). Development / recovery
-- use only.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_042_entity_edge_predicates.sql
--
-- Piping this file naively (psql $URL < this_file, without ON_ERROR_STOP)
-- will FAIL by design -- the session variable is checked before anything
-- else runs.

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_042 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

DROP TABLE IF EXISTS entity_edge_predicates;

COMMIT;
