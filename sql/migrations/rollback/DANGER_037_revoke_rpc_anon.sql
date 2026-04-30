-- DANGER_037: rollback of migration 037_revoke_rpc_anon.sql
--
-- Restores EXECUTE on the 10 SECURITY DEFINER functions to anon and
-- authenticated roles. **Only run this if you have a specific reason
-- to expose these functions on the public REST surface.** They bypass
-- RLS by design (SECURITY DEFINER) so granting EXECUTE to anon/auth
-- effectively grants admin-level access to those code paths.
--
-- Manual usage:
--     SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
--     \i sql/migrations/rollback/DANGER_037_revoke_rpc_anon.sql

BEGIN;

DO $$
BEGIN
    IF current_setting('ogham.confirm_rollback', true) IS DISTINCT FROM 'I-KNOW-WHAT-I-AM-DOING' THEN
        RAISE EXCEPTION 'Refusing to run DANGER_037 rollback. Set ogham.confirm_rollback = ''I-KNOW-WHAT-I-AM-DOING'' first.';
    END IF;
END
$$;

GRANT EXECUTE ON FUNCTION public.lifecycle_advance_fresh_to_stable(text, timestamptz, double precision, double precision) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.lifecycle_close_editing_windows(text, timestamptz) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.lifecycle_open_editing_window(uuid[]) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.lifecycle_pipeline_counts(text) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.hebbian_strengthen_edges(text[], text[], real, real) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.entity_graph_density(text) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.suggest_unlinked_by_shared_entities(uuid, text, integer, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.link_memory_entities(uuid, text, text[]) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.refresh_entity_temporal_span(bigint) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.spread_entity_activation_memories(text[], text, integer, double precision, double precision, integer) TO anon, authenticated;

COMMIT;
