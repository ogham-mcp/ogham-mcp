"""Gateway backend -- calls the Ogham managed gateway API over HTTPS.

Used when DATABASE_BACKEND=gateway. The MCP server becomes a thin
client that delegates all database operations to the hosted gateway.
Users set OGHAM_API_KEY and OGHAM_GATEWAY_URL instead of database credentials.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GatewayBackend:
    """REST client implementing the DatabaseBackend protocol via the gateway API."""

    def __init__(self, url: str, api_key: str) -> None:
        self._url = url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._url,
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=30.0,
        )

    def _post(self, path: str, data: dict) -> Any:
        resp = self._client.post(path, json=data)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:
        resp = self._client.delete(path)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, data: dict) -> Any:
        resp = self._client.put(path, json=data)
        resp.raise_for_status()
        return resp.json()

    # ── Core CRUD ────────────────────────────────────────────────────

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
        # Gateway handles embedding + scoring server-side
        return self._post(
            "/api/v1/memories",
            {
                "content": content,
                "profile": profile,
                "source": source,
                "tags": tags,
                "metadata": metadata,
            },
        )

    def store_memories_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Batch via individual stores for now
        return [self._post("/api/v1/memories", row) for row in rows]

    def get_memory_by_id(self, memory_id: str, profile: str) -> dict[str, Any] | None:
        try:
            return self._get(f"/api/v1/memories/{memory_id}", {"profile": profile})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def update_memory(
        self, memory_id: str, updates: dict[str, Any], profile: str
    ) -> dict[str, Any]:
        return self._put(f"/api/v1/memories/{memory_id}", {**updates, "profile": profile})

    def delete_memory(self, memory_id: str, profile: str) -> bool:
        try:
            self._delete(f"/api/v1/memories/{memory_id}")
            return True
        except httpx.HTTPStatusError:
            return False

    # ── Search & Retrieval ───────────────────────────────────────────

    def search_memories(
        self,
        query_embedding: list[float],
        profile: str,
        threshold: float | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._post(
            "/api/v1/search",
            {
                "query": "",
                "profile": profile,
                "limit": limit,
                "tags": tags,
            },
        )

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
        body: dict[str, Any] = {
            "query": query_text,
            "profile": profile,
            "limit": limit,
            "tags": tags,
        }
        if profiles:
            body["profiles"] = profiles
        return self._post("/api/v1/search", body)

    def list_recent_memories(
        self,
        profile: str,
        limit: int = 10,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._get(
            "/api/v1/memories",
            {"profile": profile, "limit": limit},
        )

    def get_all_memories_full(self, profile: str) -> list[dict[str, Any]]:
        return self._get("/api/v1/memories", {"profile": profile, "limit": 10000})

    def get_all_memories_content(self, profile: str | None = None) -> list[dict[str, Any]]:
        return self.get_all_memories_full(profile or "default")

    # ── Batch & Embedding Ops ────────────────────────────────────────

    def batch_check_duplicates(
        self, query_embeddings: list[list[float]], profile: str, threshold: float = 0.8
    ) -> list[bool]:
        # Not supported via gateway yet
        return [False] * len(query_embeddings)

    def batch_update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        # Embeddings managed server-side
        return 0

    # ── Access & Confidence ──────────────────────────────────────────

    def record_access(self, memory_ids: list[str]) -> None:
        # Handled server-side during search
        pass

    def update_confidence(self, memory_id: str, signal: float, profile: str) -> float:
        result = self._post(
            f"/api/v1/memories/{memory_id}/confidence",
            {"signal": signal, "profile": profile},
        )
        return result.get("confidence", 0.5)

    # ── Profile & Stats ──────────────────────────────────────────────

    def get_memory_stats(self, profile: str) -> dict[str, Any]:
        return self._get("/api/v1/profiles", {"profile": profile})

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._get("/api/v1/profiles")

    def get_profile_ttl(self, profile: str) -> int | None:
        # TTL managed server-side
        return None

    def set_profile_ttl(self, profile: str, ttl_days: int | None) -> dict[str, Any]:
        return {"profile": profile, "ttl_days": ttl_days}

    def cleanup_expired(self, profile: str) -> int:
        return 0

    def count_expired(self, profile: str) -> int:
        return 0

    # ── Relationships / Knowledge Graph ──────────────────────────────

    def auto_link_memory(
        self,
        memory_id: str,
        embedding: list[float],
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
    ) -> int:
        # Auto-linking handled server-side
        return 0

    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int:
        return 0

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
        return self._post(
            "/api/v1/search",
            {
                "query": query_text,
                "profile": profile,
                "limit": limit,
                "graph_depth": depth,
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
        return {"source_id": source_id, "target_id": target_id}

    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        min_strength: float = 0.5,
        relationship_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/api/v1/memories/{memory_id}/related",
            {"depth": depth, "limit": limit},
        )

    def apply_hebbian_decay(self, profile: str, batch_size: int = 1000) -> int:
        return 0

    def count_decay_eligible(self, profile: str) -> int:
        return 0

    def emit_audit_event(self, *args: Any, **kwargs: Any) -> None:
        pass  # Gateway handles its own audit server-side

    def query_audit_log(
        self, profile: str, limit: int = 50, operation: str | None = None
    ) -> list[dict[str, Any]]:
        return []  # Gateway audit: query via gateway API when available

    # ========================================================================
    # Wiki Tier 1 (v0.12) — gateway-backed implementations are stubbed.
    # The managed-cloud product is parked; wiki tools require self-hosted
    # backends (postgres or supabase) for v0.12. Will land alongside the
    # gateway rehome (#90) when managed product comes back.
    # ========================================================================

    def _wiki_unsupported(self, op: str) -> None:
        raise NotImplementedError(
            f"GatewayBackend does not support wiki Tier 1 op {op!r}. "
            "Use DATABASE_BACKEND=postgres or DATABASE_BACKEND=supabase."
        )

    def wiki_topic_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_topic_search")
        return []

    def wiki_topic_upsert(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._wiki_unsupported("wiki_topic_upsert")
        return {}

    def wiki_topic_get_by_key(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        self._wiki_unsupported("wiki_topic_get_by_key")
        return None

    def wiki_topic_get_affected(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_topic_get_affected")
        return []

    def wiki_topic_mark_stale(self, *args: Any, **kwargs: Any) -> None:
        self._wiki_unsupported("wiki_topic_mark_stale")

    def wiki_topic_sweep_stale(self, *args: Any, **kwargs: Any) -> int:
        self._wiki_unsupported("wiki_topic_sweep_stale")
        return 0

    def wiki_topic_list_stale(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_topic_list_stale")
        return []

    def wiki_topic_list_fresh_for_drift(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_topic_list_fresh_for_drift")
        return []

    def wiki_topic_list_all(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_topic_list_all")
        return []

    def wiki_recompute_get_source_ids(self, *args: Any, **kwargs: Any) -> list[str]:
        self._wiki_unsupported("wiki_recompute_get_source_ids")
        return []

    def wiki_recompute_get_source_content(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_recompute_get_source_content")
        return []

    def wiki_walk_graph(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._wiki_unsupported("wiki_walk_graph")
        return []

    def wiki_lint_contradictions(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._wiki_unsupported("wiki_lint_contradictions")
        return {"count": 0, "sample": []}

    def wiki_lint_orphans(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._wiki_unsupported("wiki_lint_orphans")
        return {"count": 0, "sample": []}

    def wiki_lint_stale_lifecycle(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._wiki_unsupported("wiki_lint_stale_lifecycle")
        return {"count": 0, "sample": []}
