"""Supabase backend — wraps PostgREST directly (no supabase SDK needed)."""

import logging
from collections.abc import Mapping
from typing import Any, cast

from postgrest import SyncPostgrestClient

from ogham.config import settings
from ogham.retry import with_retry

logger = logging.getLogger(__name__)


def _rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError(f"Expected PostgREST list response, got {type(data).__name__}")
    return [_row(item) for item in data]


def _row(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected PostgREST row response, got {type(data).__name__}")
    return {str(key): value for key, value in data.items()}


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
        return _rows(result.data)[0]

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        result = self._get_client().from_("memories").insert(rows).execute()
        if not result.data:
            raise RuntimeError(
                "Batch insert returned no data — check Supabase connection and table permissions"
            )
        return _rows(result.data)

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
        return _rows(result.data)

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
        return cast(list[bool], result.data)

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
        return _rows(result.data)

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
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def get_memory_stats(self, profile: str) -> dict[str, Any]:
        result = (
            self._get_client().rpc("get_memory_stats_sql", {"filter_profile": profile}).execute()
        )
        if not result.data:
            return {"profile": profile, "total": 0, "sources": {}, "top_tags": []}
        if isinstance(result.data, dict):
            return _row(result.data)
        return _rows(result.data)[0]

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
            page = _rows(result.data)
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
            query = self._get_client().from_("memories").select("id, content")
            if profile:
                query = query.eq("profile", profile)
            if last_id is not None:
                query = query.gt("id", last_id)
            result = query.order("id").limit(batch).execute()
            page = _rows(result.data)
            rows.extend(page)
            if len(page) < batch:
                break
            last_id = str(page[-1]["id"])
        return rows

    @with_retry(max_attempts=2, base_delay=0.3)
    def list_profiles(self) -> list[dict[str, Any]]:
        result = self._get_client().rpc("get_profile_counts", {}).execute()
        return _rows(result.data)

    def batch_update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        params: dict[str, Any] = {
            "memory_ids": ids,
            "new_embeddings": [str(e) for e in embeddings],
        }
        result = self._get_client().rpc("batch_update_embeddings", params).execute()
        return result.data if isinstance(result.data, int) else 0

    def record_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        params: dict[str, Any] = {"memory_ids": memory_ids}
        self._get_client().rpc("record_access", params).execute()

    def update_confidence(self, memory_id: str, signal: float, profile: str) -> float:
        params: dict[str, Any] = {
            "memory_id": memory_id,
            "signal": signal,
            "memory_profile": profile,
        }
        result = self._get_client().rpc("update_confidence", params).execute()
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
        row = _rows(result.data)[0]
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
        return len(_rows(result.data)) > 0

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
        return _rows(result.data)[0]

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
        ttl_days = _rows(result.data)[0].get("ttl_days")
        return ttl_days if isinstance(ttl_days, int) else None

    def set_profile_ttl(self, profile: str, ttl_days: int | None) -> dict[str, Any]:
        row = {"profile": profile, "ttl_days": ttl_days}
        result = self._get_client().from_("profile_settings").upsert(row).execute()
        return _rows(result.data)[0]

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
        params: dict[str, Any] = {
            "new_memory_id": memory_id,
            "new_embedding": str(embedding),
            "link_threshold": threshold,
            "max_links": max_links,
            "filter_profile": profile,
        }
        result = self._get_client().rpc("auto_link_memory", params).execute()
        return result.data if isinstance(result.data, int) else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int:
        params: dict[str, Any] = {
            "filter_profile": profile,
            "link_threshold": threshold,
            "max_links": max_links,
            "batch_size": batch_size,
        }
        result = self._get_client().rpc("link_unlinked_memories", params).execute()
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
        return _rows(result.data)

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
        return _rows(result.data)[0]

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
        return _rows(result.data)

    def spread_entity_activation(
        self,
        entity_tags: list[str],
        profile: str,
        max_depth: int = 2,
        decay: float = 0.65,
        min_activation: float = 0.05,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        return []

    def apply_hebbian_decay(self, profile: str, batch_size: int = 1000) -> int:
        return 0

    def count_decay_eligible(self, profile: str) -> int:
        return 0

    def emit_audit_event(self, *args: Any, **kwargs: Any) -> None:
        pass  # Supabase audit: add when RPC function exists

    def query_audit_log(
        self, profile: str, limit: int = 50, operation: str | None = None
    ) -> list[dict[str, Any]]:
        return []  # Supabase audit: add when RPC function exists

    # ========================================================================
    # Wiki Tier 1 (v0.12) — RPC-backed methods that mirror migration 031
    # functions. All dispatch through PostgREST rpc() against pre-registered
    # functions so PostgREST's no-arbitrary-SQL constraint is respected.
    # ========================================================================

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_search(
        self,
        profile: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_profile": profile,
            "p_query_embedding": query_embedding,
            "p_top_k": top_k,
            "p_min_similarity": min_similarity,
        }
        result = self._get_client().rpc("wiki_topic_search", params).execute()
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
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
        # PostgREST returns bytea as a hex string by default; the migration's
        # function takes bytea. Pass as hex-encoded \x-prefixed string.
        params: dict[str, Any] = {
            "p_profile": profile,
            "p_topic_key": topic_key,
            "p_content": content,
            "p_embedding": embedding,
            "p_source_memory_ids": source_memory_ids,
            "p_model_used": model_used,
            "p_source_cursor": source_cursor,
            "p_source_hash": "\\x" + source_hash.hex(),
            "p_token_count": token_count,
            "p_importance": importance,
        }
        result = self._get_client().rpc("wiki_topic_upsert", params).execute()
        # Function returns a single row (RETURNS topic_summaries).
        if isinstance(result.data, list):
            rows = _rows(result.data)
            return rows[0] if rows else {}
        return _row(result.data) if result.data else {}

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_get_by_key(self, profile: str, topic_key: str) -> dict[str, Any] | None:
        result = (
            self._get_client()
            .rpc(
                "wiki_topic_get_by_key",
                {"p_profile": profile, "p_topic_key": topic_key},
            )
            .execute()
        )
        if not result.data:
            return None
        if isinstance(result.data, list):
            rows = _rows(result.data)
            return rows[0] if rows else None
        return _row(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_get_affected(self, memory_id: str) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"p_memory_id": memory_id}
        result = self._get_client().rpc("wiki_topic_get_affected", params).execute()
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_mark_stale(self, summary_id: str, reason: str | None = None) -> None:
        params: dict[str, Any] = {"p_summary_id": summary_id, "p_reason": reason}
        self._get_client().rpc(
            "wiki_topic_mark_stale",
            params,
        ).execute()

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_sweep_stale(self, profile: str, older_than_days: int = 30) -> int:
        params: dict[str, Any] = {"p_profile": profile, "p_older_than_days": older_than_days}
        result = self._get_client().rpc("wiki_topic_sweep_stale", params).execute()
        data: Any = result.data
        if isinstance(data, list):
            if not data:
                return 0
            first: Any = data[0]
            return 0 if isinstance(first, Mapping) else int(first)
        return int(data) if data is not None else 0

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_list_stale(
        self, profile: str | None = None, older_than_days: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"p_profile": profile, "p_older_than_days": older_than_days}
        result = self._get_client().rpc("wiki_topic_list_stale", params).execute()
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_list_fresh_for_drift(self, profile: str) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"p_profile": profile}
        result = self._get_client().rpc("wiki_topic_list_fresh_for_drift", params).execute()
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_topic_list_all(self, profile: str) -> list[dict[str, Any]]:
        # Direct PostgREST table read -- the exporter wants the full row,
        # which neither list_stale nor list_fresh_for_drift returns.
        result = (
            self._get_client()
            .table("topic_summaries")
            .select(
                "id,profile_id,topic_key,content,source_count,model_used,"
                "version,status,source_hash,updated_at"
            )
            .eq("profile_id", profile)
            .order("topic_key")
            .execute()
        )
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_recompute_get_source_ids(self, profile: str, tag: str) -> list[str]:
        params: dict[str, Any] = {"p_profile": profile, "p_tag": tag}
        result = self._get_client().rpc("wiki_recompute_get_source_ids", params).execute()
        data: Any = result.data
        if not isinstance(data, list):
            return []
        ids: list[str] = []
        for item in data:
            source_id = item.get("id") if isinstance(item, Mapping) else item
            if source_id is not None:
                ids.append(str(source_id))
        return ids

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_recompute_get_source_content(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        params: dict[str, Any] = {"p_memory_ids": memory_ids}
        result = self._get_client().rpc("wiki_recompute_get_source_content", params).execute()
        return _rows(result.data)

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_walk_graph(
        self,
        start_id: str,
        max_depth: int = 1,
        direction: str = "both",
        min_strength: float = 0.0,
        relationship_types: list[str] | None = None,
        result_limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "p_start_id": start_id,
            "p_max_depth": max_depth,
            "p_direction": direction,
            "p_min_strength": min_strength,
            "p_relationship_types": relationship_types,
            "p_result_limit": result_limit,
        }
        result = self._get_client().rpc("wiki_walk_graph", params).execute()
        return _rows(result.data)

    def _split_count_sample(self, rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        """Lint RPCs return total_count column on every row; split it out."""
        if not rows:
            return 0, []
        total = int(rows[0].get("total_count") or 0)
        sample = [{k: v for k, v in r.items() if k != "total_count"} for r in rows]
        return total, sample

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_lint_contradictions(self, profile: str, sample_size: int = 10) -> dict[str, Any]:
        params: dict[str, Any] = {"p_profile": profile, "p_sample_size": sample_size}
        result = self._get_client().rpc("wiki_lint_contradictions", params).execute()
        count, sample = self._split_count_sample(_rows(result.data))
        return {"count": count, "sample": sample}

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_lint_orphans(
        self, profile: str, sample_size: int = 10, grace_minutes: int = 5
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "p_profile": profile,
            "p_sample_size": sample_size,
            "p_grace_minutes": grace_minutes,
        }
        result = self._get_client().rpc("wiki_lint_orphans", params).execute()
        count, sample = self._split_count_sample(_rows(result.data))
        return {"count": count, "sample": sample}

    @with_retry(max_attempts=2, base_delay=0.3)
    def wiki_lint_stale_lifecycle(
        self,
        profile: str,
        older_than_days: int = 90,
        sample_size: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "p_profile": profile,
            "p_older_than_days": older_than_days,
            "p_sample_size": sample_size,
        }
        result = self._get_client().rpc("wiki_lint_stale_lifecycle", params).execute()
        count, sample = self._split_count_sample(_rows(result.data))
        return {"count": count, "sample": sample, "older_than_days": older_than_days}
