"""Supabase backend — wraps the PostgREST / Supabase client."""

import logging
from typing import Any

import httpx
from supabase import Client, ClientOptions, create_client
from yarl import URL as YarlURL

from ogham.config import settings
from ogham.retry import with_retry

logger = logging.getLogger(__name__)


class SupabaseBackend:
    """DatabaseBackend implementation backed by Supabase (PostgREST)."""

    def __init__(self) -> None:
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_key,
                options=ClientOptions(
                    postgrest_client_timeout=120,
                    httpx_client=httpx.Client(timeout=120, verify=True),
                ),
            )
            # Bare PostgREST (no Kong gateway) doesn't serve /rest/v1/ prefix.
            if settings.bare_postgrest:
                self._client.postgrest.base_url = YarlURL(settings.supabase_url)
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
    ) -> dict[str, Any]:
        row = {
            "content": content,
            "embedding": str(embedding),
            "profile": profile,
            "metadata": metadata or {},
            "source": source,
            "tags": tags or [],
        }
        if expires_at is not None:
            row["expires_at"] = expires_at
        result = self._get_client().table("memories").insert(row).execute()
        if not result.data:
            raise RuntimeError(
                "Insert returned no data — check Supabase connection and table permissions"
            )
        return result.data[0]

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert multiple memories in a single PostgREST request.

        Each row must already have: content, embedding (as str), profile, metadata, source, tags.
        Optionally: expires_at.
        """
        if not rows:
            return []
        result = self._get_client().table("memories").insert(rows).execute()
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
        """Check multiple embeddings for duplicates in a single RPC call."""
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
    ) -> list[dict[str, Any]]:
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
            .table("memories")
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
        """Get memory statistics for a profile using SQL aggregation."""
        result = (
            self._get_client().rpc("get_memory_stats_sql", {"filter_profile": profile}).execute()
        )
        if not result.data:
            return {
                "profile": profile,
                "total": 0,
                "sources": {},
                "top_tags": [],
            }
        if isinstance(result.data, dict):
            return result.data
        return result.data[0]

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_all_memories_full(self, profile: str) -> list[dict[str, Any]]:
        """Fetch all memory fields (except embedding) for a profile. Excludes expired.

        Uses keyset pagination (cursor on created_at, id) to avoid
        OFFSET performance degradation on large tables.
        """
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_created_at: str | None = None
        last_id: str | None = None
        while True:
            query = (
                self._get_client()
                .table("memories")
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
        """Fetch id and content for memories. Optionally filter by profile.

        Uses keyset pagination on id to avoid OFFSET degradation.
        """
        rows: list[dict[str, Any]] = []
        batch = 1000
        last_id: str | None = None
        while True:
            query = self._get_client().table("memories").select("id, content")
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
        """Get distinct profiles with memory counts using SQL aggregation."""
        result = self._get_client().rpc("get_profile_counts", {}).execute()
        return result.data

    def batch_update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        """Batch update embeddings for multiple memories in a single RPC call."""
        result = self._get_client().rpc(
            "batch_update_embeddings",
            {
                "memory_ids": ids,
                "new_embeddings": [str(e) for e in embeddings],
            },
        ).execute()
        return result.data if isinstance(result.data, int) else 0

    def record_access(self, memory_ids: list[str]) -> None:
        """Increment access_count and update last_accessed_at for returned search results."""
        if not memory_ids:
            return
        self._get_client().rpc("record_access", {"memory_ids": memory_ids}).execute()

    def update_confidence(self, memory_id: str, signal: float, profile: str) -> float:
        """Update confidence via Bayesian posterior. Returns new confidence value."""
        result = self._get_client().rpc(
            "update_confidence",
            {"memory_id": memory_id, "signal": signal, "memory_profile": profile},
        ).execute()
        return result.data if isinstance(result.data, float) else 0.5

    def delete_memory(self, memory_id: str, profile: str) -> bool:
        result = (
            self._get_client()
            .table("memories")
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
            .table("memories")
            .update(updates)
            .eq("id", memory_id)
            .eq("profile", profile)
            .execute()
        )
        if not result.data:
            raise KeyError(f"Memory {memory_id!r} not found in profile {profile!r}")
        return result.data[0]

    def get_profile_ttl(self, profile: str) -> int | None:
        """Get the TTL in days for a profile. Returns None if not set."""
        result = (
            self._get_client()
            .table("profile_settings")
            .select("ttl_days")
            .eq("profile", profile)
            .execute()
        )
        if not result.data:
            return None
        return result.data[0].get("ttl_days")

    def set_profile_ttl(self, profile: str, ttl_days: int | None) -> dict[str, Any]:
        """Set or clear the TTL for a profile. Pass None to remove TTL."""
        row = {"profile": profile, "ttl_days": ttl_days}
        result = self._get_client().table("profile_settings").upsert(row).execute()
        return result.data[0]

    def cleanup_expired(self, profile: str) -> int:
        """Delete expired memories for a profile. Returns count of deleted rows."""
        result = (
            self._get_client()
            .rpc("cleanup_expired_memories", {"target_profile": profile})
            .execute()
        )
        return result.data if isinstance(result.data, int) else 0

    def count_expired(self, profile: str) -> int:
        """Count expired memories for a profile."""
        result = (
            self._get_client()
            .rpc("count_expired_memories", {"target_profile": profile})
            .execute()
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
        """Auto-link a memory to similar existing memories. Returns count of links created."""
        result = self._get_client().rpc(
            "auto_link_memory",
            {
                "new_memory_id": memory_id,
                "new_embedding": str(embedding),
                "link_threshold": threshold,
                "max_links": max_links,
                "filter_profile": profile,
            },
        ).execute()
        return result.data if isinstance(result.data, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int:
        """Bulk backfill auto-links for memories with no outgoing auto edges.

        Returns processed count.
        """
        result = self._get_client().rpc(
            "link_unlinked_memories",
            {
                "filter_profile": profile,
                "link_threshold": threshold,
                "max_links": max_links,
                "batch_size": batch_size,
            },
        ).execute()
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
        """Explore knowledge graph: hybrid search seeds + relationship traversal."""
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
        """Create a relationship edge between two memories."""
        row = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship": relationship,
            "strength": strength,
            "created_by": created_by,
            "metadata": metadata or {},
        }
        result = self._get_client().table("memory_relationships").insert(row).execute()
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
        """Traverse relationship graph from a memory. Returns connected memories."""
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
