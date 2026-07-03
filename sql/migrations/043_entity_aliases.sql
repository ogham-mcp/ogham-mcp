-- sql/migrations/043_entity_aliases.sql
--
-- Migration 043: entity_aliases -- surface forms mapping to a canonical
-- entity_id (v0.16 typed-edge context graph, TBU-110).
--
-- Backs alias resolution for store_triple / query_join: a caller can pass
-- "auth" and have it resolve to the AuthService entity within a profile,
-- without needing the canonical name or numeric id. UNIQUE(alias, profile)
-- keeps alias resolution unambiguous per-profile -- the same surface form
-- can point at different entities in different profiles.
--
-- Design: docs/superpowers/specs/2026-07-01-typed-edge-context-graph-design.md
--   (see Alias / canonicalization section)
-- Depends on: entities (migration 032 backfill / schema.sql).
-- Rollback: sql/migrations/rollback/DANGER_043_entity_aliases.sql

BEGIN;

CREATE TABLE IF NOT EXISTS entity_aliases (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_id  bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias      text NOT NULL,
    profile    text NOT NULL,
    strength   real NOT NULL DEFAULT 1.0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(alias, profile)
);

CREATE INDEX IF NOT EXISTS entity_aliases_entity ON entity_aliases(entity_id);

-- RLS: deny anon access. Guarded the same way as migration 041/042 --
-- FORCE lives INSIDE the anon-existence check (TBU-163) so a
-- non-Supabase install (no `anon` role) leaves RLS enabled-but-not-forced
-- instead of locking out the non-superuser owner. Self-hosters get a
-- NOTICE and no FORCE/policy instead of a silent deny-all.
ALTER TABLE entity_aliases ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- leaving entity_aliases RLS enabled '
            'but not forced (non-Supabase install; owner/app role keeps '
            'access)';
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE entity_aliases FORCE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS "Deny anon access" ON entity_aliases';
    EXECUTE 'CREATE POLICY "Deny anon access" ON entity_aliases '
            'FOR ALL TO anon USING (false) WITH CHECK (false)';
END $$;

-- Data API grant: service_role needs explicit GRANT once Supabase revokes
-- its platform-level default (see migration 038 for full rationale).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        RAISE NOTICE
            'service_role not found -- skipping Data API grant for '
            'entity_aliases (non-Supabase install)';
        RETURN;
    END IF;

    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_aliases TO service_role';
END
$$;

COMMIT;
