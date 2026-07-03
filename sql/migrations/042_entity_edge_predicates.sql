-- sql/migrations/042_entity_edge_predicates.sql
--
-- Migration 042: entity_edge_predicates -- controlled vocabulary table +
-- v1 seed (v0.16 typed-edge context graph, TBU-110).
--
-- `entity_edges.predicate` is free text, not an enum (enums are painful to
-- migrate). Instead, the application layer validates every predicate
-- against this table at construction time via `make_predicate()` --
-- callers cannot store a triple whose predicate isn't seeded here.
--
-- TBU-109 amendments (2026-07-02, applied directly -- no prior shipped
-- version to migrate away from):
--   * `scope` column added: `entity` (entity-to-entity edges, this table's
--     only v1 consumer) vs `memory` (reserved for a future memory-scope
--     predicate use, not used by v0.16).
--   * SUPERSEDES dropped from the v1 vocab -- redundant with the
--     `valid_to` / `superseded_by` write-time supersession mechanism on
--     entity_edges itself; a predicate row would only invite confusion
--     between "this edge supersedes that edge" (a predicate) and "this
--     edge IS superseded" (a column). One mechanism, not two.
--   * `domain_types`, `range_types`, `schema_org_uri`, `iirds_uri` columns
--     dropped from the day-one schema -- YAGNI. They land via a follow-up
--     migration once TBU-129 (v0.17 Schema.org URI columns) or a real
--     constraint-enforcement need arrives.
--
-- Design: docs/superpowers/specs/2026-07-01-typed-edge-context-graph-design.md
--   (see Controlled predicate vocabulary section, amended per TBU-109)
-- Rollback: sql/migrations/rollback/DANGER_042_entity_edge_predicates.sql

BEGIN;

CREATE TABLE IF NOT EXISTS entity_edge_predicates (
    predicate   text PRIMARY KEY,
    label       text NOT NULL,
    description text,
    inverse     text,
    scope       text NOT NULL CHECK (scope IN ('entity','memory'))
);

-- v1 seed: 16 entity-scope predicate rows (6 inverse pairs + 4 standalone).
-- SUPERSEDES intentionally omitted per TBU-109 (redundant with valid_to).
INSERT INTO entity_edge_predicates(predicate, label, description, inverse, scope) VALUES
    ('DEPENDS_ON',      'depends on',       'Subject requires object to function or complete',            'DEPENDED_ON_BY', 'entity'),
    ('DEPENDED_ON_BY',  'depended on by',   'Inverse of DEPENDS_ON',                                       'DEPENDS_ON',     'entity'),
    ('OWNS',            'owns',             'Subject has ownership or authority over object',              'OWNED_BY',       'entity'),
    ('OWNED_BY',        'owned by',         'Inverse of OWNS',                                             'OWNS',           'entity'),
    ('ASSIGNED_TO',     'assigned to',      'Subject is assigned to object (task -> person, item -> box)', 'HAS_ASSIGNEE',   'entity'),
    ('HAS_ASSIGNEE',    'has assignee',     'Inverse of ASSIGNED_TO',                                      'ASSIGNED_TO',    'entity'),
    ('DECIDED',         'decided',          'Subject decided on object (agent -> decision fact)',          NULL,             'entity'),
    ('MENTIONS',        'mentions',         'Subject mentions object in a memory / message',               NULL,             'entity'),
    ('BLOCKS',          'blocks',           'Subject blocks progress on object',                           'BLOCKED_BY',     'entity'),
    ('BLOCKED_BY',      'blocked by',       'Inverse of BLOCKS',                                           'BLOCKS',         'entity'),
    ('PART_OF',         'part of',          'Subject is a structural component of object',                 'CONTAINS',       'entity'),
    ('CONTAINS',        'contains',         'Inverse of PART_OF',                                          'PART_OF',        'entity'),
    ('SUPPORTS',        'supports',         'Subject provides evidence for object (entity-scope)',         'CONTRADICTS',    'entity'),
    ('CONTRADICTS',     'contradicts',      'Subject provides counter-evidence to object (entity-scope)',  'SUPPORTS',       'entity'),
    ('EVOLVED_INTO',    'evolved into',     'Object is a later version of subject (matches NATEOB1)',      NULL,             'entity'),
    ('RELATED_TO',      'related to',       'Low-signal catchall -- prefer a specific predicate',          NULL,             'entity')
ON CONFLICT (predicate) DO NOTHING;

-- RLS: deny anon access. Guarded the same way as migration 041 -- FORCE
-- lives INSIDE the anon-existence check (TBU-163) so a non-Supabase
-- install (no `anon` role) leaves RLS enabled-but-not-forced instead of
-- locking out the non-superuser owner. Self-hosters get a NOTICE and no
-- FORCE/policy instead of a silent deny-all.
ALTER TABLE entity_edge_predicates ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- leaving entity_edge_predicates RLS '
            'enabled but not forced (non-Supabase install; owner/app role '
            'keeps access)';
        RETURN;
    END IF;

    EXECUTE 'ALTER TABLE entity_edge_predicates FORCE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS "Deny anon access" ON entity_edge_predicates';
    EXECUTE 'CREATE POLICY "Deny anon access" ON entity_edge_predicates '
            'FOR ALL TO anon USING (false) WITH CHECK (false)';
END $$;

-- Data API grant: service_role needs explicit GRANT once Supabase revokes
-- its platform-level default (see migration 038 for full rationale).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        RAISE NOTICE
            'service_role not found -- skipping Data API grant for '
            'entity_edge_predicates (non-Supabase install)';
        RETURN;
    END IF;

    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON public.entity_edge_predicates TO service_role';
END
$$;

COMMIT;
