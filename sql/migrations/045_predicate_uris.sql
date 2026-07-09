-- sql/migrations/045_predicate_uris.sql
--
-- Migration 045: add portable URI columns to entity_edge_predicates
-- (v0.17 typed-edge follow-on, TBU-129). Anticipated by migration 042's
-- header note ("schema_org_uri, iirds_uri ... land via a follow-up
-- migration once TBU-129 arrives").
--
--   * ogham_uri (NOT NULL): stable Ogham-namespace identity for every
--     predicate -- https://ogham-mcp.dev/vocab#<PREDICATE>.
--   * schema_org_uri (nullable): honest Schema.org alignment, populated
--     ONLY for the five predicates with a genuine equivalent property
--     (WebFetch-verified against schema.org 2026-07-06). NULL otherwise.
--   * iirds_uri (nullable): reserved for TBU-128 -- all NULL here.
--
-- Additive + idempotent: safe to re-run and safe on existing v0.16 installs.
-- Design: docs/superpowers/specs/2026-07-06-predicate-uris-design.md
-- Rollback: sql/migrations/rollback/DANGER_045_predicate_uris.sql

BEGIN;

ALTER TABLE entity_edge_predicates ADD COLUMN IF NOT EXISTS ogham_uri      text;
ALTER TABLE entity_edge_predicates ADD COLUMN IF NOT EXISTS schema_org_uri text;
ALTER TABLE entity_edge_predicates ADD COLUMN IF NOT EXISTS iirds_uri      text;

-- every predicate gets a stable ogham_uri derived from its name
UPDATE entity_edge_predicates SET ogham_uri = 'https://ogham-mcp.dev/vocab#' || predicate;

-- the five verified Schema.org alignments (owns/owner + isPartOf/hasPart pairs + mentions)
UPDATE entity_edge_predicates SET schema_org_uri = 'https://schema.org/owns'     WHERE predicate = 'OWNS';
UPDATE entity_edge_predicates SET schema_org_uri = 'https://schema.org/owner'    WHERE predicate = 'OWNED_BY';
UPDATE entity_edge_predicates SET schema_org_uri = 'https://schema.org/mentions' WHERE predicate = 'MENTIONS';
UPDATE entity_edge_predicates SET schema_org_uri = 'https://schema.org/isPartOf' WHERE predicate = 'PART_OF';
UPDATE entity_edge_predicates SET schema_org_uri = 'https://schema.org/hasPart'  WHERE predicate = 'CONTAINS';
-- iirds_uri intentionally left NULL (TBU-128)

ALTER TABLE entity_edge_predicates ALTER COLUMN ogham_uri SET NOT NULL;

COMMIT;
