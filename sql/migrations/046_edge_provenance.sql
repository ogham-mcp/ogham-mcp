-- sql/migrations/046_edge_provenance.sql
--
-- Migration 046: edge provenance lineage (v0.17, TBU-124). Provenance /
-- data-lineage on graph edges is a standard pattern (cf. W3C PROV); the
-- derived_from JSONB shape here was informed by surveying several existing
-- memory-project designs.
--
--   * derived_from jsonb NOT NULL DEFAULT '[]': array of
--     {source_edge_id, source_memory_id, reasoning?} objects recording the
--     evidence a typed edge was derived from (edge->edge and edge->memory).
--   * GIN index for find_derivatives' containment (@>) queries.
--
-- Additive + idempotent. DEFAULT '[]' backfills existing rows at ALTER time
-- (no UPDATE). Safe on existing v0.16 installs. Supabase-SQL-editor-safe.
-- Design: docs/superpowers/specs/2026-07-08-provenance-chains-design.md
-- Rollback: sql/migrations/rollback/DANGER_046_edge_provenance.sql

BEGIN;

ALTER TABLE entity_edges ADD COLUMN IF NOT EXISTS derived_from jsonb NOT NULL DEFAULT '[]';
CREATE INDEX IF NOT EXISTS entity_edges_derived_from_gin
    ON entity_edges USING gin (derived_from);

COMMIT;
