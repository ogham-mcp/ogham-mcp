"""PostgresBackend — direct Postgres/Neon driver using psycopg."""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ogham.config import settings
from ogham.retry import with_retry

logger = logging.getLogger(__name__)

# Tenant context for multi-tenant deployments (e.g. the gateway).
#
# When set, every connection checkout in this backend will run
# `SELECT set_config('app.tenant_id', <id>, true)` so that Postgres
# row-level security policies on memories / memory_relationships /
# embeddings_cache can scope queries to the current tenant via
# `current_setting('app.tenant_id', true)`.
#
# Variable name `app.tenant_id` is intentionally consistent with the
# existing Phase 1 tenant_context helper at
# gateway/src/ogham_gateway/middleware/tenant.py and the embedding_cache
# RLS policies that have been in production since 2026-03-16. This
# Phase 2 refactor extends the same convention to ALL ogham core DB
# operations, not just embedding cache.
#
# Self-hosted users never set this contextvar; the checkout helper
# becomes a no-op and behaviour is identical to before. This means the
# refactor introduces ZERO behavioural change for single-tenant
# deployments while enabling DB-enforced isolation for multi-tenant
# deployments without changing the public ogham API.
#
# `set_config(..., true)` is transaction-local, which is compatible
# with PgBouncer transaction-pooling mode (Neon, Supabase pooled
# endpoints). Unlike `SET LOCAL`, set_config supports parameterised
# values so we don't need an f-string + UUID-validation pattern.
_tenant_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ogham_tenant_id", default=None
)


def set_tenant_context(tenant_id: str | None) -> None:
    """Set the current tenant ID for subsequent DB operations.

    Multi-tenant callers (e.g. the Ogham gateway) call this in their
    request middleware after authenticating the caller. Self-hosted
    callers do not need to call this -- the default `None` is a no-op
    and behaviour is identical to before this function existed.
    """
    _tenant_id_var.set(tenant_id)


def get_tenant_context() -> str | None:
    """Return the currently set tenant ID, or None."""
    return _tenant_id_var.get()


# Allowlist of valid memory table columns to prevent SQL injection via dynamic column names
_ALLOWED_MEMORY_COLUMNS = frozenset(
    {
        "content",
        "embedding",
        "metadata",
        "source",
        "profile",
        "tags",
        "expires_at",
        "access_count",
        "last_accessed_at",
        "confidence",
    }
)

# Columns to SELECT when we don't want the embedding or fts vectors.
_COLUMNS = (
    "id, content, metadata, source, profile, tags,"
    " created_at, updated_at, expires_at, access_count, last_accessed_at, confidence"
)


def _embedding_literal(embedding: list[float]) -> str:
    """Format as Postgres vector literal '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class PostgresBackend:
    """DatabaseBackend implementation using psycopg + connection pool."""

    def __init__(self) -> None:
        self._pool: ConnectionPool | None = None

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            if not settings.database_url:
                raise RuntimeError("DATABASE_URL is required for PostgresBackend")
            self._pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=int(os.environ.get("OGHAM_POOL_MIN", "1")),
                max_size=int(os.environ.get("OGHAM_POOL_MAX", "5")),
                kwargs={"row_factory": dict_row},
            )
            self._ensure_columns()
        return self._pool

    def _ensure_columns(self) -> None:
        """Auto-add columns introduced in newer versions.

        Runs on first connection so upgraders don't need manual migrations.
        All statements use IF NOT EXISTS -- safe to run repeatedly.
        """
        migrations = [
            # v0.7.0: importance, surprise, compression
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance real DEFAULT 0.5",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS surprise real DEFAULT 0.5",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS compression_level integer DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS original_content text",
            # v0.8.0: temporal columns
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS occurrence_period tstzrange",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS recurrence_days integer[]",
        ]
        try:
            with self._pool.connection() as conn:  # type: ignore[union-attr]
                for sql in migrations:
                    conn.execute(sql)
                conn.commit()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Auto-migration skipped: %s", e)

    # ── Helper ────────────────────────────────────────────────────────

    @contextmanager
    def _checkout(self):
        """Check out a connection from the pool with tenant context applied.

        If `set_tenant_context()` has been called on the current task /
        thread, this runs `SELECT set_config('app.tenant_id', <id>, true)`
        before yielding the connection. The `true` makes it transaction-local,
        so it works correctly with PgBouncer transaction pooling. Variable
        name matches the existing Phase 1 tenant_context helper used by
        embedding_cache RLS.

        Self-hosted callers never set the contextvar -- this becomes a
        plain `with pool.connection()` and behaves exactly as before.
        """
        with self._get_pool().connection() as conn:
            tenant_id = _tenant_id_var.get()
            if tenant_id is not None:
                conn.execute(
                    "SELECT set_config('app.tenant_id', %s, true)",
                    (tenant_id,),
                )
            yield conn

    def _execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        fetch: str = "all",
    ) -> Any:
        """Central query execution.

        fetch: "all" -> list[dict], "one" -> dict|None,
               "scalar" -> single value, "none" -> None
        """
        with self._checkout() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch == "none":
                    return None
                if fetch == "scalar":
                    row = cur.fetchone()
                    if row is None:
                        return None
                    # dict_row returns a dict — get the first value
                    return next(iter(row.values()))
                if fetch == "one":
                    return cur.fetchone()
                # fetch == "all"
                return cur.fetchall()

    # ── Core CRUD ─────────────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        embedding: list[float],
        profile: str,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        expires_at: str | None = None,
        importance: float = 0.5,
        surprise: float = 0.5,
        recurrence_days: list[int] | None = None,
    ) -> dict[str, Any]:
        cols = [
            "content",
            "embedding",
            "profile",
            "metadata",
            "source",
            "tags",
            "importance",
            "surprise",
        ]
        vals = [
            "%(content)s",
            "%(embedding)s::vector",
            "%(profile)s",
            "%(metadata)s",
            "%(source)s",
            "%(tags)s",
            "%(importance)s",
            "%(surprise)s",
        ]
        params: dict[str, Any] = {
            "content": content,
            "embedding": _embedding_literal(embedding),
            "profile": profile,
            "metadata": Jsonb(metadata or {}),
            "source": source,
            "tags": tags or [],
            "importance": importance,
            "surprise": surprise,
        }
        if expires_at is not None:
            cols.append("expires_at")
            vals.append("%(expires_at)s")
            params["expires_at"] = expires_at
        if recurrence_days is not None:
            cols.append("recurrence_days")
            vals.append("%(recurrence_days)s")
            params["recurrence_days"] = recurrence_days

        sql = f"INSERT INTO memories ({', '.join(cols)}) VALUES ({', '.join(vals)}) RETURNING *"
        row = self._execute(sql, params, fetch="one")
        if row is None:
            raise RuntimeError("INSERT returned no data")
        # Drop large vector fields from result
        row.pop("embedding", None)
        row.pop("fts", None)
        return row

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert multiple memories in a single multi-row VALUES INSERT.

        Every row in a batch must carry the same column set -- the benchmark
        and export_import.py paths already satisfy this. Varying shapes would
        force us back to per-row execute, so we validate upfront.

        Prior implementation looped ``cur.execute()`` per row. At 100-row
        harness batches * hundreds of benchmark questions that turned a
        clean LME ingest into an hours-long run. Now one execute per batch,
        RETURNING order matches input order (PostgreSQL preserves VALUES
        order for RETURNING).
        """
        if not rows:
            return []

        # Freeze the column set from the first row; require consistency.
        cols = list(rows[0].keys())
        col_set = set(cols)
        for col in cols:
            if col not in _ALLOWED_MEMORY_COLUMNS:
                raise ValueError(f"Unknown column: {col!r}")
        for i, row in enumerate(rows):
            if set(row.keys()) != col_set:
                raise ValueError(
                    f"store_memories_batch: row {i} has columns {set(row.keys())}, "
                    f"expected {col_set}. All rows in a batch must share columns."
                )

        # Build ONE multi-row VALUES clause + a flat params dict keyed by
        # (col, row_index) so psycopg can bind them by name.
        values_clauses: list[str] = []
        params: dict[str, Any] = {}
        for i, row in enumerate(rows):
            placeholders = []
            for col in cols:
                param_name = f"{col}_{i}"
                val = row[col]
                if col == "embedding":
                    placeholders.append(f"%({param_name})s::vector")
                    params[param_name] = val if isinstance(val, str) else _embedding_literal(val)
                elif col == "metadata":
                    placeholders.append(f"%({param_name})s")
                    params[param_name] = Jsonb(val) if not isinstance(val, Jsonb) else val
                else:
                    placeholders.append(f"%({param_name})s")
                    params[param_name] = val
            values_clauses.append(f"({', '.join(placeholders)})")

        sql = (
            f"INSERT INTO memories ({', '.join(cols)}) "
            f"VALUES {', '.join(values_clauses)} RETURNING *"
        )

        results: list[dict[str, Any]] = []
        with self._checkout() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for result in cur.fetchall():
                    result.pop("embedding", None)
                    result.pop("fts", None)
                    results.append(result)
        return results

    def update_memory(
        self, memory_id: str, updates: dict[str, Any], profile: str
    ) -> dict[str, Any]:
        if not updates:
            raise ValueError("No updates provided")
        for key in updates:
            if key not in _ALLOWED_MEMORY_COLUMNS:
                raise ValueError(f"Unknown column: {key!r}")
        set_clauses = []
        params: dict[str, Any] = {"id": memory_id, "profile": profile}
        for key, val in updates.items():
            param = f"u_{key}"
            if key == "metadata":
                set_clauses.append(f"{key} = %({param})s")
                params[param] = Jsonb(val)
            elif key == "embedding":
                set_clauses.append(f"{key} = %({param})s::vector")
                params[param] = val if isinstance(val, str) else _embedding_literal(val)
            else:
                set_clauses.append(f"{key} = %({param})s")
                params[param] = val

        sql = (
            f"UPDATE memories SET {', '.join(set_clauses)}"
            f" WHERE id = %(id)s AND profile = %(profile)s"
            f" RETURNING {_COLUMNS}"
        )
        row = self._execute(sql, params, fetch="one")
        if row is None:
            raise KeyError(f"Memory {memory_id!r} not found in profile {profile!r}")
        return row

    def get_memory_by_id(self, memory_id: str, profile: str) -> dict[str, Any] | None:
        sql = f"SELECT {_COLUMNS} FROM memories WHERE id = %(id)s AND profile = %(profile)s"
        return self._execute(sql, {"id": memory_id, "profile": profile}, fetch="one")

    def delete_memory(self, memory_id: str, profile: str) -> bool:
        row = self._execute(
            "DELETE FROM memories WHERE id = %(id)s AND profile = %(profile)s RETURNING id",
            {"id": memory_id, "profile": profile},
            fetch="one",
        )
        return row is not None

    # ── Hebbian Decay ──────────────────────────────────────────────────

    def apply_hebbian_decay(self, profile: str, batch_size: int = 1000) -> int:
        """Run Hebbian decay on a profile. Returns count of decayed memories."""
        try:
            result = self._execute(
                "SELECT apply_hebbian_decay(%(profile)s, %(batch_size)s) AS decayed",
                {"profile": profile, "batch_size": batch_size},
                fetch="scalar",
            )
            return result or 0
        except Exception:
            logger.debug("Hebbian decay skipped (function may not exist yet)")
            return 0

    def count_decay_eligible(self, profile: str) -> int:
        """Dry-run: count memories eligible for decay."""
        try:
            result = self._execute(
                """SELECT count(*)::integer FROM memories
                   WHERE profile = %(profile)s
                     AND importance > 0.05
                     AND (expires_at IS NULL OR expires_at > now())
                     AND (last_accessed_at IS NULL
                          OR last_accessed_at < now() - interval '7 days')""",
                {"profile": profile},
                fetch="scalar",
            )
            return result or 0
        except Exception:
            return 0

    # ── Audit ─────────────────────────────────────────────────────────

    def emit_audit_event(
        self,
        profile: str,
        operation: str,
        resource_id: str | None = None,
        outcome: str = "success",
        source: str | None = None,
        embedding_model: str | None = None,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        result_ids: list[str] | None = None,
        result_count: int | None = None,
        query_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append-only audit event. Fails silently if audit_log table missing."""
        try:
            self._execute(
                """INSERT INTO audit_log
                   (profile, operation, resource_id, outcome, source,
                    embedding_model, tokens_used, cost_usd,
                    result_ids, result_count, query_hash, metadata)
                   VALUES (%(profile)s, %(operation)s, %(resource_id)s, %(outcome)s,
                           %(source)s, %(embedding_model)s, %(tokens_used)s, %(cost_usd)s,
                           %(result_ids)s, %(result_count)s, %(query_hash)s, %(metadata)s)""",
                {
                    "profile": profile,
                    "operation": operation,
                    "resource_id": resource_id,
                    "outcome": outcome,
                    "source": source,
                    "embedding_model": embedding_model,
                    "tokens_used": tokens_used,
                    "cost_usd": cost_usd,
                    "result_ids": result_ids,
                    "result_count": result_count,
                    "query_hash": query_hash,
                    "metadata": Jsonb(metadata or {}),
                },
                fetch="none",
            )
        except Exception:
            logger.debug("Audit event skipped (table may not exist yet)")

    def query_audit_log(
        self,
        profile: str,
        limit: int = 50,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query audit log for a profile. Returns empty list if table missing."""
        try:
            conditions = ["profile = %(profile)s"]
            params: dict[str, Any] = {"profile": profile, "limit": limit}
            if operation:
                conditions.append("operation = %(operation)s")
                params["operation"] = operation
            where = " AND ".join(conditions)
            rows = self._execute(
                f"SELECT * FROM audit_log WHERE {where} ORDER BY event_time DESC LIMIT %(limit)s",
                params,
                fetch="all",
            )
            return rows or []
        except Exception:
            logger.debug("Audit query skipped (table may not exist yet)")
            return []

    # ── Search & Retrieval ────────────────────────────────────────────

    @with_retry(max_attempts=2, base_delay=0.3)
    def search_memories(
        self,
        query_embedding: list[float],
        profile: str,
        threshold: float | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "embedding": _embedding_literal(query_embedding),
            "threshold": threshold or settings.default_match_threshold,
            "limit": limit or settings.default_match_count,
            "tags": tags,
            "source": source,
            "profile": profile,
        }
        return self._execute(
            "SELECT * FROM match_memories("
            "  %(embedding)s::vector, %(threshold)s::float, %(limit)s::integer,"
            "  %(tags)s, %(source)s, %(profile)s"
            ")",
            params,
        )

    @with_retry(max_attempts=2, base_delay=0.3)
    def hybrid_search_memories(
        self,
        query_text: str,
        query_embedding: list[float],
        profile: str,
        limit: int | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        profiles: list[str] | None = None,
        query_entity_tags: list[str] | None = None,
        recency_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query_text": query_text,
            "embedding": _embedding_literal(query_embedding),
            "limit": limit or settings.default_match_count,
            "profile": profile,
            "tags": tags,
            "source": source,
            "profiles": profiles,
            "query_entity_tags": query_entity_tags,
            "recency_decay": recency_decay,
        }
        return self._execute(
            "SELECT * FROM hybrid_search_memories("
            "  %(query_text)s, %(embedding)s::vector, %(limit)s::integer,"
            "  %(profile)s, %(tags)s, %(source)s,"
            "  0.3::float, 0.7::float, 10::integer, %(profiles)s, %(query_entity_tags)s,"
            "  %(recency_decay)s::float"
            ")",
            params,
        )

    @with_retry(max_attempts=2, base_delay=0.3)
    def list_recent_memories(
        self,
        profile: str,
        limit: int = 10,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        where = ["profile = %(profile)s", "(expires_at IS NULL OR expires_at > now())"]
        params: dict[str, Any] = {"profile": profile, "limit": limit}
        if source:
            where.append("source = %(source)s")
            params["source"] = source
        if tags:
            where.append("tags && %(tags)s")
            params["tags"] = tags

        sql = (
            f"SELECT {_COLUMNS} FROM memories"
            f" WHERE {' AND '.join(where)}"
            " ORDER BY created_at DESC LIMIT %(limit)s"
        )
        return self._execute(sql, params)

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_all_memories_full(self, profile: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_created_at: str | None = None
        last_id: str | None = None
        while True:
            params: dict[str, Any] = {"profile": profile, "batch": batch}
            conditions = [
                "profile = %(profile)s",
                "(expires_at IS NULL OR expires_at > now())",
            ]
            if last_created_at is not None:
                conditions.append(
                    "(created_at > %(last_created_at)s OR "
                    "(created_at = %(last_created_at)s AND id > %(last_id)s))"
                )
                params["last_created_at"] = last_created_at
                params["last_id"] = last_id
            where = " AND ".join(conditions)
            sql = (
                f"SELECT {_COLUMNS} FROM memories WHERE {where}"
                " ORDER BY created_at, id LIMIT %(batch)s"
            )
            page = self._execute(sql, params)
            rows.extend(page)
            if len(page) < batch:
                break
            last_row = page[-1]
            last_created_at = str(last_row["created_at"])
            last_id = str(last_row["id"])
        return rows

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_all_memories_content(self, profile: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_id: str | None = None
        while True:
            params: dict[str, Any] = {"batch": batch}
            conditions: list[str] = []
            if profile:
                conditions.append("profile = %(profile)s")
                params["profile"] = profile
            if last_id is not None:
                conditions.append("id > %(last_id)s")
                params["last_id"] = last_id
            where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"SELECT id, content FROM memories{where} ORDER BY id LIMIT %(batch)s"
            page = self._execute(sql, params)
            rows.extend(page)
            if len(page) < batch:
                break
            last_id = str(page[-1]["id"])
        return rows

    # ── Batch & Embedding Ops ─────────────────────────────────────────

    @with_retry(max_attempts=2, base_delay=0.3)
    def batch_check_duplicates(
        self,
        query_embeddings: list[list[float]],
        profile: str,
        threshold: float = 0.8,
    ) -> list[bool]:
        if not query_embeddings:
            return []
        emb_literals = [_embedding_literal(e) for e in query_embeddings]
        result = self._execute(
            "SELECT batch_check_duplicates(  %(embeddings)s::vector[], %(threshold)s, %(profile)s)",
            {"embeddings": emb_literals, "threshold": threshold, "profile": profile},
            fetch="scalar",
        )
        return result if result is not None else []

    def batch_update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        emb_literals = [_embedding_literal(e) for e in embeddings]
        result = self._execute(
            "SELECT batch_update_embeddings(%(ids)s::uuid[], %(embeddings)s::vector[])",
            {"ids": ids, "embeddings": emb_literals},
            fetch="scalar",
        )
        return result if isinstance(result, int) else 0

    # ── Access & Confidence ───────────────────────────────────────────

    def record_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        self._execute(
            "SELECT record_access(%(ids)s::uuid[])",
            {"ids": memory_ids},
            fetch="none",
        )

    def update_confidence(self, memory_id: str, signal: float, profile: str) -> float:
        result = self._execute(
            "SELECT update_confidence(%(id)s, %(signal)s, %(profile)s)",
            {"id": memory_id, "signal": signal, "profile": profile},
            fetch="scalar",
        )
        return result if isinstance(result, float) else 0.5

    # ── Profile & Stats ───────────────────────────────────────────────

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_memory_stats(self, profile: str) -> dict[str, Any]:
        result = self._execute(
            "SELECT get_memory_stats_sql(%(profile)s)",
            {"profile": profile},
            fetch="scalar",
        )
        if not result:
            return {"profile": profile, "total": 0, "sources": {}, "top_tags": []}
        if isinstance(result, dict):
            return result
        # Should already be a dict from jsonb, but handle list edge case
        return result[0] if isinstance(result, list) else result

    @with_retry(max_attempts=2, base_delay=0.3)
    def list_profiles(self) -> list[dict[str, Any]]:
        return self._execute("SELECT * FROM get_profile_counts()")

    def get_profile_ttl(self, profile: str) -> int | None:
        row = self._execute(
            "SELECT ttl_days FROM profile_settings WHERE profile = %(profile)s",
            {"profile": profile},
            fetch="one",
        )
        if row is None:
            return None
        return row.get("ttl_days")

    def set_profile_ttl(self, profile: str, ttl_days: int | None) -> dict[str, Any]:
        row = self._execute(
            "INSERT INTO profile_settings (profile, ttl_days)"
            " VALUES (%(profile)s, %(ttl_days)s)"
            " ON CONFLICT (profile) DO UPDATE SET ttl_days = EXCLUDED.ttl_days"
            " RETURNING *",
            {"profile": profile, "ttl_days": ttl_days},
            fetch="one",
        )
        return row

    def cleanup_expired(self, profile: str) -> int:
        result = self._execute(
            "SELECT cleanup_expired_memories(%(profile)s)",
            {"profile": profile},
            fetch="scalar",
        )
        return result if isinstance(result, int) else 0

    def count_expired(self, profile: str) -> int:
        result = self._execute(
            "SELECT count_expired_memories(%(profile)s)",
            {"profile": profile},
            fetch="scalar",
        )
        return result if isinstance(result, int) else 0

    # ── Relationships / Knowledge Graph ───────────────────────────────

    @with_retry(max_attempts=2, base_delay=0.3)
    def auto_link_memory(
        self,
        memory_id: str,
        embedding: list[float],
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
    ) -> int:
        result = self._execute(
            "SELECT auto_link_memory("
            "  %(memory_id)s, %(embedding)s::vector,"
            "  %(threshold)s, %(max_links)s, %(profile)s"
            ")",
            {
                "memory_id": memory_id,
                "embedding": _embedding_literal(embedding),
                "threshold": threshold,
                "max_links": max_links,
                "profile": profile,
            },
            fetch="scalar",
        )
        return result if isinstance(result, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int:
        result = self._execute(
            "SELECT link_unlinked_memories("
            "  %(profile)s, %(threshold)s, %(max_links)s, %(batch_size)s"
            ")",
            {
                "profile": profile,
                "threshold": threshold,
                "max_links": max_links,
                "batch_size": batch_size,
            },
            fetch="scalar",
        )
        return result if isinstance(result, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def explore_memory_graph(
        self,
        query_text: str,
        query_embedding: list[float],
        profile: str,
        limit: int = 5,
        depth: int = 1,
        min_strength: float = 0.5,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._execute(
            "SELECT * FROM explore_memory_graph("
            "  %(query_text)s, %(embedding)s::vector, %(profile)s,"
            "  %(limit)s::integer, %(depth)s::integer, %(min_strength)s::float,"
            "  %(tags)s, %(source)s"
            ")",
            {
                "query_text": query_text,
                "embedding": _embedding_literal(query_embedding),
                "profile": profile,
                "limit": limit,
                "depth": depth,
                "min_strength": min_strength,
                "tags": tags,
                "source": source,
            },
        )

    def spread_entity_activation(
        self,
        entity_tags: list[str],
        profile: str,
        max_depth: int = 2,
        decay: float = 0.65,
        min_activation: float = 0.05,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Walk the entity graph and return activated memories."""
        return self._execute(
            "SELECT memory_id, activation"
            " FROM spread_entity_activation_memories("
            "  %(tags)s, %(profile)s, %(max_depth)s,"
            "  %(decay)s, %(min_activation)s, %(max_results)s"
            ")",
            {
                "tags": entity_tags,
                "profile": profile,
                "max_depth": max_depth,
                "decay": decay,
                "min_activation": min_activation,
                "max_results": max_results,
            },
        )

    def refresh_entity_temporal_span(self, entity_id: int) -> None:
        """Update temporal_span for a single entity after ingest."""
        self._execute(
            "SELECT refresh_entity_temporal_span(%(eid)s)",
            {"eid": entity_id},
            fetch="none",
        )

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        strength: float = 1.0,
        created_by: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._execute(
            "INSERT INTO memory_relationships"
            " (source_id, target_id, relationship, strength, created_by, metadata)"
            " VALUES (%(source_id)s, %(target_id)s, %(relationship)s::relationship_type,"
            "         %(strength)s, %(created_by)s, %(metadata)s)"
            " RETURNING *",
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship": relationship,
                "strength": strength,
                "created_by": created_by,
                "metadata": Jsonb(metadata or {}),
            },
            fetch="one",
        )
        if row is None:
            raise RuntimeError("Insert returned no data for relationship")
        return row

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        min_strength: float = 0.5,
        relationship_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._execute(
            "SELECT * FROM get_related_memories("
            "  %(memory_id)s, %(depth)s::integer, %(min_strength)s::float,"
            "  %(types)s::relationship_type[], %(limit)s::integer"
            ")",
            {
                "memory_id": memory_id,
                "depth": depth,
                "min_strength": min_strength,
                "types": relationship_types,
                "limit": limit,
            },
        )

    # ========================================================================
    # Wiki Tier 1 (v0.12) — call into the migration 031 functions via psycopg
    # so the SQL is in one place (server-side) and both backends share it.
    # SupabaseBackend mirrors these via PostgREST rpc() in supabase.py.
    # ========================================================================

    def wiki_topic_search(
        self,
        profile: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        return (
            self._execute(
                "SELECT * FROM wiki_topic_search("
                "  %(profile)s, %(emb)s::vector, %(top_k)s, %(min_sim)s"
                ")",
                {
                    "profile": profile,
                    "emb": query_embedding,
                    "top_k": top_k,
                    "min_sim": min_similarity,
                },
                fetch="all",
            )
            or []
        )

    def wiki_topic_upsert(
        self,
        profile: str,
        topic_key: str,
        content: str,
        embedding: list[float],
        source_memory_ids: list[str],
        model_used: str,
        source_cursor: str | None,
        source_hash: bytes,
        token_count: int | None = None,
        importance: float = 0.5,
    ) -> dict[str, Any]:
        row = self._execute(
            "SELECT * FROM wiki_topic_upsert("
            "  %(profile)s, %(topic_key)s, %(content)s, %(embedding)s::vector,"
            "  %(memory_ids)s::uuid[], %(model_used)s,"
            "  %(source_cursor)s::uuid, %(source_hash)s,"
            "  %(token_count)s, %(importance)s"
            ")",
            {
                "profile": profile,
                "topic_key": topic_key,
                "content": content,
                "embedding": embedding,
                "memory_ids": source_memory_ids,
                "model_used": model_used,
                "source_cursor": source_cursor,
                "source_hash": source_hash,
                "token_count": token_count,
                "importance": importance,
            },
            fetch="one",
        )
        return dict(row) if row else {}

    def wiki_topic_get_by_key(self, profile: str, topic_key: str) -> dict[str, Any] | None:
        row = self._execute(
            "SELECT * FROM wiki_topic_get_by_key(%(profile)s, %(topic_key)s)",
            {"profile": profile, "topic_key": topic_key},
            fetch="one",
        )
        return dict(row) if row else None

    def wiki_topic_get_affected(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT * FROM wiki_topic_get_affected(%(id)s::uuid)",
            {"id": memory_id},
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    def wiki_topic_mark_stale(self, summary_id: str, reason: str | None = None) -> None:
        self._execute(
            "SELECT wiki_topic_mark_stale(%(id)s::uuid, %(reason)s)",
            {"id": summary_id, "reason": reason},
            fetch="none",
        )

    def wiki_topic_sweep_stale(self, profile: str, older_than_days: int = 30) -> int:
        n = self._execute(
            "SELECT wiki_topic_sweep_stale(%(profile)s, %(days)s)",
            {"profile": profile, "days": older_than_days},
            fetch="scalar",
        )
        return int(n or 0)

    def wiki_topic_list_stale(
        self, profile: str | None = None, older_than_days: int | None = None
    ) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT * FROM wiki_topic_list_stale(%(profile)s, %(days)s)",
            {"profile": profile, "days": older_than_days},
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    def wiki_topic_list_fresh_for_drift(self, profile: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT * FROM wiki_topic_list_fresh_for_drift(%(profile)s)",
            {"profile": profile},
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    def wiki_topic_list_all(self, profile: str) -> list[dict[str, Any]]:
        # Direct table read -- the export path needs every column for
        # frontmatter (content, model_used, version, status, source_count,
        # updated_at, source_hash). The existing list functions either
        # filter by status or return a column subset, neither of which
        # fits the exporter's needs.
        rows = self._execute(
            "SELECT id, profile_id, topic_key, content, source_count, model_used, "
            "version, status, source_hash, updated_at "
            "FROM topic_summaries WHERE profile_id = %(profile)s "
            "ORDER BY topic_key",
            {"profile": profile},
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    def wiki_recompute_get_source_ids(self, profile: str, tag: str) -> list[str]:
        rows = self._execute(
            "SELECT id FROM wiki_recompute_get_source_ids(%(profile)s, %(tag)s)",
            {"profile": profile, "tag": tag},
            fetch="all",
        )
        return [r["id"] for r in rows or []]

    def wiki_recompute_get_source_content(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        rows = self._execute(
            "SELECT id, content FROM wiki_recompute_get_source_content(%(ids)s::uuid[])",
            {"ids": memory_ids},
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    def wiki_walk_graph(
        self,
        start_id: str,
        max_depth: int = 1,
        direction: str = "both",
        min_strength: float = 0.0,
        relationship_types: list[str] | None = None,
        result_limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT * FROM wiki_walk_graph("
            "  %(start_id)s::uuid, %(max_depth)s, %(direction)s,"
            "  %(min_strength)s, %(types)s::text[], %(result_limit)s"
            ")",
            {
                "start_id": start_id,
                "max_depth": max_depth,
                "direction": direction,
                "min_strength": min_strength,
                "types": relationship_types,
                "result_limit": result_limit,
            },
            fetch="all",
        )
        return [dict(r) for r in rows or []]

    @staticmethod
    def _split_count_sample(rows: list[Any]) -> tuple[int, list[dict[str, Any]]]:
        if not rows:
            return 0, []
        first = dict(rows[0]) if not isinstance(rows[0], dict) else rows[0]
        total = int(first.get("total_count") or 0)
        sample = [
            {
                k: v
                for k, v in (dict(r) if not isinstance(r, dict) else r).items()
                if k != "total_count"
            }
            for r in rows
        ]
        return total, sample

    def wiki_lint_contradictions(self, profile: str, sample_size: int = 10) -> dict[str, Any]:
        rows = self._execute(
            "SELECT * FROM wiki_lint_contradictions(%(profile)s, %(n)s)",
            {"profile": profile, "n": sample_size},
            fetch="all",
        )
        count, sample = self._split_count_sample(rows or [])
        return {"count": count, "sample": sample}

    def wiki_lint_orphans(
        self, profile: str, sample_size: int = 10, grace_minutes: int = 5
    ) -> dict[str, Any]:
        rows = self._execute(
            "SELECT * FROM wiki_lint_orphans(%(profile)s, %(n)s, %(grace)s)",
            {"profile": profile, "n": sample_size, "grace": grace_minutes},
            fetch="all",
        )
        count, sample = self._split_count_sample(rows or [])
        return {"count": count, "sample": sample}

    def wiki_lint_stale_lifecycle(
        self,
        profile: str,
        older_than_days: int = 90,
        sample_size: int = 10,
    ) -> dict[str, Any]:
        rows = self._execute(
            "SELECT * FROM wiki_lint_stale_lifecycle(%(profile)s, %(days)s, %(n)s)",
            {"profile": profile, "days": older_than_days, "n": sample_size},
            fetch="all",
        )
        count, sample = self._split_count_sample(rows or [])
        return {"count": count, "sample": sample, "older_than_days": older_than_days}
