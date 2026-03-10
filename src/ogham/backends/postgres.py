"""PostgresBackend — direct Postgres/Neon driver using psycopg."""

from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ogham.config import settings
from ogham.retry import with_retry

logger = logging.getLogger(__name__)

# Allowlist of valid memory table columns to prevent SQL injection via dynamic column names
_ALLOWED_MEMORY_COLUMNS = frozenset({
    "content", "embedding", "metadata", "source", "profile", "tags",
    "expires_at", "access_count", "last_accessed_at", "confidence",
})

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
                min_size=1,
                max_size=5,
                kwargs={"row_factory": dict_row},
            )
        return self._pool

    # ── Helper ────────────────────────────────────────────────────────

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
        with self._get_pool().connection() as conn:
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
    ) -> dict[str, Any]:
        cols = ["content", "embedding", "profile", "metadata", "source", "tags"]
        vals = [
            "%(content)s",
            "%(embedding)s::vector",
            "%(profile)s",
            "%(metadata)s",
            "%(source)s",
            "%(tags)s",
        ]
        params: dict[str, Any] = {
            "content": content,
            "embedding": _embedding_literal(embedding),
            "profile": profile,
            "metadata": Jsonb(metadata or {}),
            "source": source,
            "tags": tags or [],
        }
        if expires_at is not None:
            cols.append("expires_at")
            vals.append("%(expires_at)s")
            params["expires_at"] = expires_at

        sql = f"INSERT INTO memories ({', '.join(cols)}) VALUES ({', '.join(vals)}) RETURNING *"
        row = self._execute(sql, params, fetch="one")
        if row is None:
            raise RuntimeError("INSERT returned no data")
        # Drop large vector fields from result
        row.pop("embedding", None)
        row.pop("fts", None)
        return row

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert multiple memories. Each row comes pre-formatted from export_import.py."""
        if not rows:
            return []
        results: list[dict[str, Any]] = []
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cols = list(row.keys())
                    for col in cols:
                        if col not in _ALLOWED_MEMORY_COLUMNS:
                            raise ValueError(f"Unknown column: {col!r}")
                    placeholders = []
                    params: dict[str, Any] = {}
                    for col in cols:
                        param_name = col
                        val = row[col]
                        if col == "embedding":
                            placeholders.append(f"%({param_name})s::vector")
                            # Already a string from export_import.py
                            params[param_name] = (
                                val if isinstance(val, str) else _embedding_literal(val)
                            )
                        elif col == "metadata":
                            placeholders.append(f"%({param_name})s")
                            params[param_name] = Jsonb(val) if not isinstance(val, Jsonb) else val
                        else:
                            placeholders.append(f"%({param_name})s")
                            params[param_name] = val

                    sql = (
                        f"INSERT INTO memories ({', '.join(cols)})"
                        f" VALUES ({', '.join(placeholders)}) RETURNING *"
                    )
                    cur.execute(sql, params)
                    result = cur.fetchone()
                    if result:
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

    def delete_memory(self, memory_id: str, profile: str) -> bool:
        row = self._execute(
            "DELETE FROM memories WHERE id = %(id)s AND profile = %(profile)s RETURNING id",
            {"id": memory_id, "profile": profile},
            fetch="one",
        )
        return row is not None

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
            "  %(embedding)s::vector, %(threshold)s, %(limit)s,"
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
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query_text": query_text,
            "embedding": _embedding_literal(query_embedding),
            "limit": limit or settings.default_match_count,
            "profile": profile,
            "tags": tags,
            "source": source,
        }
        return self._execute(
            "SELECT * FROM hybrid_search_memories("
            "  %(query_text)s, %(embedding)s::vector, %(limit)s,"
            "  %(profile)s, %(tags)s, %(source)s"
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
            "SELECT batch_check_duplicates("
            "  %(embeddings)s::vector[], %(threshold)s, %(profile)s"
            ")",
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
            "  %(limit)s, %(depth)s, %(min_strength)s,"
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
            "  %(memory_id)s, %(depth)s, %(min_strength)s,"
            "  %(types)s::relationship_type[], %(limit)s"
            ")",
            {
                "memory_id": memory_id,
                "depth": depth,
                "min_strength": min_strength,
                "types": relationship_types,
                "limit": limit,
            },
        )
