"""Supabase backend — wraps PostgREST directly (no supabase SDK needed)."""

import logging
from typing import Any

from postgrest import SyncPostgrestClient

from ogham.config import settings
from ogham.retry import with_retry

logger = logging.getLogger(__name__)


class SupabaseBackend:
    """DatabaseBackend implementation backed by Supabase/PostgREST.

    Uses postgrest-py directly instead of the supabase SDK to avoid
    heavy transitive dependencies (storage3, pyiceberg, pyroaring).
    """

    def __init__(self) -> None:
        self._client: SyncPostgrestClient | None = None

    def _get_client(self) -> SyncPostgrestClient:
        if self._client is None:
            if not settings.supabase_url:
                raise RuntimeError("SUPABASE_URL is required for SupabaseBackend")
            if not settings.supabase_key:
                raise RuntimeError("SUPABASE_KEY is required for SupabaseBackend")
            if settings.bare_postgrest:
                base_url = settings.supabase_url
            else:
                base_url = f"{settings.supabase_url}/rest/v1"

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._client = SyncPostgrestClient(
                    base_url,
                    headers={
                        "apikey": settings.supabase_key,
                        "Authorization": f"Bearer {settings.supabase_key}",
                        "Prefer": "return=representation",
                    },
                    timeout=120,
                )
        return self._client

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
        row = {
            "content": content,
            "embedding": str(embedding),
            "profile": profile,
            "metadata": metadata or {},
            "source": source,
            "tags": tags or [],
            "importance": importance,
            "surprise": surprise,
        }
        if expires_at is not None:
            row["expires_at"] = expires_at
        if recurrence_days is not None:
            row["recurrence_days"] = recurrence_days
        result = self._get_client().from_("memories").insert(row).execute()
        if not result.data:
            raise RuntimeError(
                "Insert returned no data — check Supabase connection and table permissions"
            )
        return result.data[0]

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        result = self._get_client().from_("memories").insert(rows).execute()
        if not result.data:
            raise RuntimeError(
                "Batch insert returned no data — check Supabase connection and table permissions"
            )
        return result.data

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
            "query_embedding": str(query_embedding),
            "match_threshold": threshold or settings.default_match_threshold,
            "match_count": limit or settings.default_match_count,
            "filter_profile": profile,
        }
        if tags:
            params["filter_tags"] = tags
        if source:
            params["filter_source"] = source

        result = self._get_client().rpc("match_memories", params).execute()
        return result.data

    @with_retry(max_attempts=2, base_delay=0.3)
    def batch_check_duplicates(
        self,
        query_embeddings: list[list[float]],
        profile: str,
        threshold: float = 0.8,
    ) -> list[bool]:
        if not query_embeddings:
            return []
        params = {
            "query_embeddings": [str(e) for e in query_embeddings],
            "match_threshold": threshold,
            "filter_profile": profile,
        }
        result = self._get_client().rpc("batch_check_duplicates", params).execute()
        return result.data

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
        # query_entity_tags and recency_decay are accepted for API parity
        # with PostgresBackend but not yet implemented in the Supabase RPC.
        params: dict[str, Any] = {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "match_count": limit or settings.default_match_count,
            "filter_profile": profile,
        }
        if tags:
            params["filter_tags"] = tags
        if source:
            params["filter_source"] = source
        if profiles:
            params["filter_profiles"] = profiles

        result = self._get_client().rpc("hybrid_search_memories", params).execute()
        return result.data

    @with_retry(max_attempts=2, base_delay=0.3)
    def list_recent_memories(
        self,
        profile: str,
        limit: int = 10,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            self._get_client()
            .from_("memories")
            .select(
                "id, content, metadata, source, profile, tags,"
                " created_at, updated_at, expires_at, access_count, last_accessed_at, confidence"
            )
        )
        query = query.eq("profile", profile)
        query = query.or_("expires_at.is.null,expires_at.gt.now()")
        if source:
            query = query.eq("source", source)
        if tags:
            query = query.overlaps("tags", tags)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_memory_stats(self, profile: str) -> dict[str, Any]:
        result = (
            self._get_client().rpc("get_memory_stats_sql", {"filter_profile": profile}).execute()
        )
        if not result.data:
            return {"profile": profile, "total": 0, "sources": {}, "top_tags": []}
        if isinstance(result.data, dict):
            return result.data
        return result.data[0]

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_all_memories_full(self, profile: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_created_at: str | None = None
        last_id: str | None = None
        while True:
            query = (
                self._get_client()
                .from_("memories")
                .select(
                    "id, content, metadata, source, profile, tags,"
                    " created_at, updated_at, expires_at,"
                    " access_count, last_accessed_at, confidence"
                )
            )
            query = query.eq("profile", profile)
            query = query.or_("expires_at.is.null,expires_at.gt.now()")
            if last_created_at is not None:
                query = query.or_(
                    f"created_at.gt.{last_created_at},"
                    f"and(created_at.eq.{last_created_at},id.gt.{last_id})"
                )
            result = query.order("created_at").order("id").limit(batch).execute()
            rows.extend(result.data)
            if len(result.data) < batch:
                break
            last_row = result.data[-1]
            last_created_at = last_row["created_at"]
            last_id = last_row["id"]
        return rows

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_all_memories_content(self, profile: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_id: str | None = None
        while True:
            query = self._get_client().from_("memories").select("id, content")
            if profile:
                query = query.eq("profile", profile)
            if last_id is not None:
                query = query.gt("id", last_id)
            result = query.order("id").limit(batch).execute()
            rows.extend(result.data)
            if len(result.data) < batch:
                break
            last_id = result.data[-1]["id"]
        return rows

    @with_retry(max_attempts=2, base_delay=0.3)
    def list_profiles(self) -> list[dict[str, Any]]:
        result = self._get_client().rpc("get_profile_counts", {}).execute()
        return result.data

    def batch_update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        result = (
            self._get_client()
            .rpc(
                "batch_update_embeddings",
                {"memory_ids": ids, "new_embeddings": [str(e) for e in embeddings]},
            )
            .execute()
        )
        return result.data if isinstance(result.data, int) else 0

    def record_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        self._get_client().rpc("record_access", {"memory_ids": memory_ids}).execute()

    def update_confidence(self, memory_id: str, signal: float, profile: str) -> float:
        result = (
            self._get_client()
            .rpc(
                "update_confidence",
                {"memory_id": memory_id, "signal": signal, "memory_profile": profile},
            )
            .execute()
        )
        return result.data if isinstance(result.data, float) else 0.5

    def get_memory_by_id(self, memory_id: str, profile: str) -> dict[str, Any] | None:
        result = (
            self._get_client()
            .from_("memories")
            .select("*")
            .eq("id", memory_id)
            .eq("profile", profile)
            .execute()
        )
        if not result.data:
            return None
        row = result.data[0]
        row.pop("embedding", None)
        row.pop("fts", None)
        return row

    def delete_memory(self, memory_id: str, profile: str) -> bool:
        result = (
            self._get_client()
            .from_("memories")
            .delete()
            .eq("id", memory_id)
            .eq("profile", profile)
            .execute()
        )
        return len(result.data) > 0

    def update_memory(
        self, memory_id: str, updates: dict[str, Any], profile: str
    ) -> dict[str, Any]:
        result = (
            self._get_client()
            .from_("memories")
            .update(updates)
            .eq("id", memory_id)
            .eq("profile", profile)
            .execute()
        )
        if not result.data:
            raise KeyError(f"Memory {memory_id!r} not found in profile {profile!r}")
        return result.data[0]

    def get_profile_ttl(self, profile: str) -> int | None:
        result = (
            self._get_client()
            .from_("profile_settings")
            .select("ttl_days")
            .eq("profile", profile)
            .execute()
        )
        if not result.data:
            return None
        return result.data[0].get("ttl_days")

    def set_profile_ttl(self, profile: str, ttl_days: int | None) -> dict[str, Any]:
        row = {"profile": profile, "ttl_days": ttl_days}
        result = self._get_client().from_("profile_settings").upsert(row).execute()
        return result.data[0]

    def cleanup_expired(self, profile: str) -> int:
        result = (
            self._get_client()
            .rpc("cleanup_expired_memories", {"target_profile": profile})
            .execute()
        )
        return result.data if isinstance(result.data, int) else 0

    def count_expired(self, profile: str) -> int:
        result = (
            self._get_client().rpc("count_expired_memories", {"target_profile": profile}).execute()
        )
        return result.data if isinstance(result.data, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def auto_link_memory(
        self,
        memory_id: str,
        embedding: list[float],
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
    ) -> int:
        result = (
            self._get_client()
            .rpc(
                "auto_link_memory",
                {
                    "new_memory_id": memory_id,
                    "new_embedding": str(embedding),
                    "link_threshold": threshold,
                    "max_links": max_links,
                    "filter_profile": profile,
                },
            )
            .execute()
        )
        return result.data if isinstance(result.data, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int:
        result = (
            self._get_client()
            .rpc(
                "link_unlinked_memories",
                {
                    "filter_profile": profile,
                    "link_threshold": threshold,
                    "max_links": max_links,
                    "batch_size": batch_size,
                },
            )
            .execute()
        )
        return result.data if isinstance(result.data, int) else 0

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
        params: dict[str, Any] = {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "filter_profile": profile,
            "match_count": limit,
            "traversal_depth": depth,
            "min_strength": min_strength,
        }
        if tags:
            params["filter_tags"] = tags
        if source:
            params["filter_source"] = source

        result = self._get_client().rpc("explore_memory_graph", params).execute()
        return result.data

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        strength: float = 1.0,
        created_by: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship": relationship,
            "strength": strength,
            "created_by": created_by,
            "metadata": metadata or {},
        }
        result = self._get_client().from_("memory_relationships").insert(row).execute()
        if not result.data:
            raise RuntimeError("Insert returned no data for relationship")
        return result.data[0]

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        min_strength: float = 0.5,
        relationship_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "start_id": memory_id,
            "max_depth": depth,
            "min_strength": min_strength,
            "result_limit": limit,
        }
        if relationship_types:
            params["filter_types"] = relationship_types

        result = self._get_client().rpc("get_related_memories", params).execute()
        return result.data

    def apply_hebbian_decay(self, profile: str, batch_size: int = 1000) -> int:
        return 0

    def count_decay_eligible(self, profile: str) -> int:
        return 0

    def emit_audit_event(self, **kwargs: Any) -> None:
        pass  # Supabase audit: add when RPC function exists

    def query_audit_log(
        self, profile: str, limit: int = 50, operation: str | None = None
    ) -> list[dict[str, Any]]:
        return []  # Supabase audit: add when RPC function exists
