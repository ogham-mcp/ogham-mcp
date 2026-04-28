-- Migration 019: Tenant isolation Phase 2 (gateway #95, blocks #94 GDPR delete)
--
-- Closes the gap from gateway/src/ogham_gateway/routes/memories.py:112 TODO:
-- "ogham core uses its own sync pool without SET LOCAL. Replace with
-- tenant-scoped SQL when migrating to gateway-owned queries."
--
-- Phase 1 (2026-03-16) wired SET LOCAL app.tenant_id into the embedding
-- cache. Phase 2 extends the same pattern to the memories and
-- memory_relationships tables so the gateway can offer DB-enforced tenant
-- isolation -- the load-bearing requirement for GDPR right-to-erasure
-- (#94) and clinical-positioning procurement security questionnaires.
--
-- ARCHITECTURE:
--   * New tenant_id uuid column on memories and memory_relationships.
--     Nullable. Backfilled to NULL on existing rows.
--   * RLS policies that filter by current_setting('app.tenant_id', true).
--   * NULL fallback: when no tenant context is set (self-hosted, ogham
--     CLI, migrations, REPL), the policy returns all rows. This preserves
--     backward compatibility for self-hosted users who run schema.sql
--     fresh -- they never call set_tenant_context() so the contextvar
--     stays None and queries see everything as before.
--   * When tenant context IS set (always true for gateway requests),
--     queries are filtered to that tenant_id only.
--   * Variable name app.tenant_id is intentionally identical to the
--     embedding_cache RLS established in Phase 1 -- one consistent
--     namespace across all tenant-scoped tables.
--
-- BETA DATA: existing rows have tenant_id = NULL. After this migration
-- they will be visible only when no tenant context is set (i.e. via
-- self-hosted CLI). The gateway will not see them because it always
-- sets a context. The non-derkevinburns beta accounts will be wiped
-- before launch as part of #94, leaving derkevinburns's data which can
-- be backfilled with a tenant_id manually.
--
-- APPLIES TO: each regional Supabase project (US Virginia, EU Frankfurt,
-- AP Singapore). Run via direct connection (not pooled) because of
-- the ALTER TABLE statements -- Neon PgBouncer drops DDL silently on
-- the pooled endpoint.

-- ── memories table ────────────────────────────────────────────────────

ALTER TABLE memories ADD COLUMN IF NOT EXISTS tenant_id uuid;

-- B-tree index for tenant lookups (most queries will filter by tenant_id
-- via the RLS policy, so this index is critical for query performance).
CREATE INDEX IF NOT EXISTS idx_memories_tenant_id
    ON memories (tenant_id) WHERE tenant_id IS NOT NULL;

-- Composite index supporting the common (tenant, profile, time) pattern
-- used by list_recent_memories and many search variants.
CREATE INDEX IF NOT EXISTS idx_memories_tenant_profile_created
    ON memories (tenant_id, profile, created_at DESC) WHERE tenant_id IS NOT NULL;

-- Helper: returns current tenant_id from session config or NULL.
-- Wrapping in a STABLE function with EXCEPTION handling avoids the
-- "invalid input syntax for type uuid" error that fires when Postgres
-- evaluates the OR clause's cast on an empty session variable instead
-- of short-circuiting on the IS NULL check. STABLE = evaluated once
-- per query, not once per row, which is critical for performance under
-- RLS at scale. Discovered during Phase 2 RLS isolation testing
-- 2026-04-07.
--
-- SET search_path = pg_catalog closes the search-path-shadowing class
-- of vulnerability (Supabase linter rule 0011 function_search_path_mutable).
-- A hostile user with CREATE on a schema in their search_path could
-- shadow current_setting / nullif / the uuid cast and intercept the
-- function on every RLS check. All three primitives live in pg_catalog
-- so we pin to that only. Added 2026-04-08 after EU linter caught it.
CREATE OR REPLACE FUNCTION current_tenant_id() RETURNS uuid
    LANGUAGE plpgsql STABLE
    SET search_path = pg_catalog
    AS $$
BEGIN
    RETURN nullif(current_setting('app.tenant_id', true), '')::uuid;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$;

-- New tenant-scoped RLS policy. Coexists with the existing "Deny anon
-- access" policy from schema.sql. Service role bypasses both.
DROP POLICY IF EXISTS "Tenant scoped access" ON memories;
CREATE POLICY "Tenant scoped access" ON memories
    FOR ALL
    USING (
        -- Self-hosted / no context: see everything (single-tenant fallback)
        current_tenant_id() IS NULL
        -- Multi-tenant: only rows for THIS tenant
        OR tenant_id = current_tenant_id()
    )
    WITH CHECK (
        current_tenant_id() IS NULL
        OR tenant_id = current_tenant_id()
    );

-- ── memory_relationships table ────────────────────────────────────────

ALTER TABLE memory_relationships ADD COLUMN IF NOT EXISTS tenant_id uuid;

CREATE INDEX IF NOT EXISTS idx_relationships_tenant_id
    ON memory_relationships (tenant_id) WHERE tenant_id IS NOT NULL;

DROP POLICY IF EXISTS "Tenant scoped access" ON memory_relationships;
CREATE POLICY "Tenant scoped access" ON memory_relationships
    FOR ALL
    USING (
        current_tenant_id() IS NULL
        OR tenant_id = current_tenant_id()
    )
    WITH CHECK (
        current_tenant_id() IS NULL
        OR tenant_id = current_tenant_id()
    );

-- Note: profile_settings is not currently used by the gateway and
-- contains low-volume per-profile TTL config. Adding tenant scoping
-- here is deferred to a follow-up migration if/when it becomes
-- multi-tenant relevant. For now it stays single-tenant.

-- ── Verification ──────────────────────────────────────────────────────
--
-- After applying, verify with:
--
--   -- 1. Columns exist
--   SELECT column_name FROM information_schema.columns
--   WHERE table_name = 'memories' AND column_name = 'tenant_id';
--
--   -- 2. Policies exist
--   SELECT policyname FROM pg_policies
--   WHERE tablename = 'memories' AND policyname = 'Tenant scoped access';
--
--   -- 3. Self-hosted fallback works (no tenant context, returns all rows)
--   RESET app.tenant_id;
--   SELECT count(*) FROM memories;  -- should return all rows
--
--   -- 4. Tenant context filters correctly
--   SET LOCAL app.tenant_id = '00000000-0000-0000-0000-000000000001';
--   SELECT count(*) FROM memories;  -- should return only matching rows
