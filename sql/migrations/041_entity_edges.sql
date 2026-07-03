-- sql/migrations/041_entity_edges.sql
--
-- Migration 041: entity_edges -- typed subject/predicate/object edges with
-- write-time temporal supersession (v0.16 typed-edge context graph, TBU-110).
--
-- Each row is a directed edge between two `entities` rows, scoped to a
-- `profile`. `predicate` is free text (not an enum -- see migration 042 for
-- the controlled vocabulary table that constrains allowed values at the
-- application layer via `make_predicate()`; enums are painful to migrate).
--
-- Supersession is write-time, not a background job: when store_triple()
-- writes a new edge that matches an existing (subject_id, predicate,
-- object_id, profile) current row, the old row's `valid_to` is set to
-- now() and `superseded_by` points at the new row's id. The partial unique
-- index below enforces "at most one current edge per (subject, predicate,
-- object, profile)" while still allowing the historical superseded rows to
-- coexist -- `WHERE valid_to IS NULL` scopes the uniqueness to the current
-- generation only.
--
-- Design: docs/superpowers/specs/2026-07-01-typed-edge-context-graph-design.md
-- Depends on: entities (migration 032 backfill / schema.sql).
-- Rollback: sql/migrations/rollback/DANGER_041_entity_edges.sql

BEGIN;

CREATE TABLE IF NOT EXISTS entity_edges (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_id    bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate     text NOT NULL,
    object_id     bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    profile       text NOT NULL,
    fact_id       uuid,
    strength      real NOT NULL DEFAULT 1.0,
    metadata      jsonb NOT NULL DEFAULT '{}',
    valid_from    timestamptz NOT NULL DEFAULT now(),
    valid_to      timestamptz,
    superseded_by bigint REFERENCES entity_edges(id),
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (subject_id <> object_id)
);

-- Uniqueness on the CURRENT edge only. Lets supersession add a row without
-- violating uniqueness -- the superseded row's valid_to is stamped first.
CREATE UNIQUE INDEX IF NOT EXISTS entity_edges_current_uq
    ON entity_edges(subject_id, predicate, object_id, profile)
    WHERE valid_to IS NULL;

-- Fast subject/object lookups with predicate filter, current-only --
-- these back query_join's hop-by-hop traversal.
CREATE INDEX IF NOT EXISTS entity_edges_subject_pred_current
    ON entity_edges(subject_id, predicate) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS entity_edges_object_pred_current
    ON entity_edges(object_id, predicate) WHERE valid_to IS NULL;

-- Profile scope index for per-profile queries.
CREATE INDEX IF NOT EXISTS entity_edges_profile_current
    ON entity_edges(profile) WHERE valid_to IS NULL;

-- RLS: deny anon access. ENABLE is unconditional -- it works on vanilla
-- Postgres with no role dependency and never blocks the table owner on
-- its own. FORCE is guarded INSIDE the anon-existence check below rather
-- than applied unconditionally (TBU-163): FORCE ROW LEVEL SECURITY
-- subjects even the table owner to RLS, so "FORCE + no policy" is
-- deny-all, including the owner -- fine on Supabase (the Deny-anon
-- policy exists alongside it) but a full lockout for a non-superuser app
-- role that owns these tables on a non-Supabase install. Self-hosters on
-- vanilla Postgres (no `anon` role) get a NOTICE and RLS stays
-- enabled-but-not-forced -- the owner/app role keeps full access and no
-- anon policy is needed since there's no anon role to deny. Mirrors the
-- guard pattern in migration 036 (which predates this fix and is out of
-- scope here).
ALTER TABLE entity_edges ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- leaving entity_edges RLS enabled but '
            'not forced (non-Supabase install; owner/app role keeps access)';
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE entity_edges FORCE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS "Deny anon access" ON entity_edges';
    EXECUTE 'CREATE POLICY "Deny anon access" ON entity_edges '
            'FOR ALL TO anon USING (false) WITH CHECK (false)';
END $$;

-- Data API grant: service_role needs explicit GRANT once Supabase revokes
-- its platform-level default (see migration 038 for full rationale).
-- Guarded the same way -- service_role does not exist on vanilla Postgres.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        RAISE NOTICE
            'service_role not found -- skipping Data API grant for '
            'entity_edges (non-Supabase install)';
        RETURN;
    END IF;

    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_edges TO service_role';
END
$$;

COMMIT;
