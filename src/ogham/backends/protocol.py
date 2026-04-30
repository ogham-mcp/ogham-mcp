"""DatabaseBackend protocol — the contract every backend driver must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol that all database backend drivers must implement.

    Every method returns plain dicts/lists — no ORM objects.
    """

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
    ) -> dict[str, Any]: ...

    def store_memories_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def update_memory(
        self,
        memory_id: str,
        updates: dict[str, Any],
        profile: str,
    ) -> dict[str, Any]: ...

    def get_memory_by_id(
        self,
        memory_id: str,
        profile: str,
    ) -> dict[str, Any] | None: ...

    def delete_memory(
        self,
        memory_id: str,
        profile: str,
    ) -> bool: ...

    # ── Search & Retrieval ───────────────────────────────────────────

    def search_memories(
        self,
        query_embedding: list[float],
        profile: str,
        threshold: float | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]: ...

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
    ) -> list[dict[str, Any]]: ...

    def list_recent_memories(
        self,
        profile: str,
        limit: int = 10,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_all_memories_full(
        self,
        profile: str,
    ) -> list[dict[str, Any]]: ...

    def get_all_memories_content(
        self,
        profile: str | None = None,
    ) -> list[dict[str, Any]]: ...

    # ── Batch & Embedding Ops ────────────────────────────────────────

    def batch_check_duplicates(
        self,
        query_embeddings: list[list[float]],
        profile: str,
        threshold: float = 0.8,
    ) -> list[bool]: ...

    def batch_update_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
    ) -> int: ...

    # ── Access & Confidence ──────────────────────────────────────────

    def record_access(
        self,
        memory_ids: list[str],
    ) -> None: ...

    def update_confidence(
        self,
        memory_id: str,
        signal: float,
        profile: str,
    ) -> float: ...

    # ── Profile & Stats ──────────────────────────────────────────────

    def get_memory_stats(
        self,
        profile: str,
    ) -> dict[str, Any]: ...

    def list_profiles(self) -> list[dict[str, Any]]: ...

    def get_profile_ttl(
        self,
        profile: str,
    ) -> int | None: ...

    def set_profile_ttl(
        self,
        profile: str,
        ttl_days: int | None,
    ) -> dict[str, Any]: ...

    def cleanup_expired(
        self,
        profile: str,
    ) -> int: ...

    def count_expired(
        self,
        profile: str,
    ) -> int: ...

    # ── Relationships / Knowledge Graph ──────────────────────────────

    def auto_link_memory(
        self,
        memory_id: str,
        embedding: list[float],
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
    ) -> int: ...

    def link_unlinked_memories(
        self,
        profile: str,
        threshold: float = 0.85,
        max_links: int = 5,
        batch_size: int = 100,
    ) -> int: ...

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
    ) -> list[dict[str, Any]]: ...

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        strength: float = 1.0,
        created_by: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        min_strength: float = 0.5,
        relationship_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...

    # ── Hebbian Decay ─────────────────────────────────────────────────

    def apply_hebbian_decay(self, profile: str, batch_size: int = 1000) -> int: ...
    def count_decay_eligible(self, profile: str) -> int: ...

    # ── Lifecycle (v0.13.1 — migration 035 RPC parity) ────────────────

    def lifecycle_advance_stages(
        self,
        profile: str,
        cutoff_iso: str,
        surprise_gate: float,
        importance_gate: float,
    ) -> int: ...

    def lifecycle_close_editing_windows(
        self,
        profile: str,
        cutoff_iso: str,
    ) -> int: ...

    def lifecycle_open_editing_window(
        self,
        memory_ids: list[str],
    ) -> None: ...

    def lifecycle_pipeline_counts(
        self,
        profile: str,
    ) -> dict[str, int]: ...

    # ── Hebbian co-retrieval edges (v0.13.1) ──────────────────────────

    def hebbian_strengthen_edges(
        self,
        sources: list[str],
        targets: list[str],
        bootstrap: float,
        rate: float,
    ) -> int: ...

    # ── Entity graph density signal (v0.13.1) ─────────────────────────

    def entity_graph_density(
        self,
        profile: str,
    ) -> tuple[float, float]: ...

    """Returns (entity_count, edge_count) -- both as floats so callers can
    compute density = edges / entities without re-casting."""

    # ── Entity graph writes (v0.14) ───────────────────────────────────

    def link_memory_entities(
        self,
        memory_id: str,
        profile: str,
        entity_tags: list[str],
    ) -> int: ...

    """Upsert entities and link to a memory. ``entity_tags`` are
    ``"type:name"`` strings from ``ogham.extraction.extract_entities``.
    Returns the number of new (memory, entity) edges inserted (0 on a
    re-run since memory_entities has a unique constraint)."""

    # ── Hidden-link suggestions (v0.13.1) ─────────────────────────────

    def suggest_unlinked_by_shared_entities(
        self,
        memory_id: str,
        profile: str,
        min_shared: int,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    # ── Audit ────────────────────────────────────────────────────────

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
    ) -> None: ...

    def query_audit_log(
        self,
        profile: str,
        limit: int = 50,
        operation: str | None = None,
    ) -> list[dict[str, Any]]: ...
