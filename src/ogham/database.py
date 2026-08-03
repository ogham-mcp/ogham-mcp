"""Database facade — delegates to the configured backend driver.

All callers import from this module; the backend is selected at runtime
based on ``settings.database_backend`` (defaults to ``"supabase"``).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from ogham.backends.protocol import DatabaseBackend
from ogham.entity_graph import V1_PREDICATES, EntityGraph

logger = logging.getLogger(__name__)

_backend: DatabaseBackend | None = None
_entity_graph: EntityGraph | None = None


def set_tenant_context(tenant_id: str | None) -> None:
    """Set the tenant ID for subsequent DB operations on this task / thread.

    Multi-tenant deployments (e.g. the Ogham gateway) call this in their
    request middleware after authenticating the caller. Self-hosted users
    do not need to call this -- the default `None` is a no-op.

    Currently only the PostgresBackend honours this contextvar. The Supabase
    backend (PostgREST-based) is for self-hosted single-tenant use and does
    not need or use it. If a multi-tenant deployment ever wants to use the
    Supabase backend, it would need its own JWT-based scoping mechanism.
    """
    # Lazy import to avoid loading psycopg in self-hosted Supabase setups.
    from ogham.backends.postgres import set_tenant_context as _set

    _set(tenant_id)


def get_tenant_context() -> str | None:
    """Return the currently set tenant ID, or None."""
    from ogham.backends.postgres import get_tenant_context as _get

    return _get()


def _reset_backend() -> None:
    """Reset the backend singleton. Used by tests."""
    global _backend
    _backend = None


def _validate_schema_fingerprint(backend: Any, settings: Any) -> None:
    """Refuse to serve if ``settings.embedding_dim`` disagrees with the DB column dim.

    TBU-159 blocker guard: dim-parameterizing the shipping schemas (Design
    Council Option A) is worse than doing nothing without this check --
    a stale-dim DB with a bumped ``EMBEDDING_DIM`` looks like it works
    (writes succeed) while reads silently fail (dimension mismatch on the
    vector comparison), which is much harder to diagnose than a clean
    startup failure.

    Only enforced for backends that expose raw SQL introspection
    (``PostgresBackend`` via psycopg -- same ``hasattr(backend, "_execute")``
    discriminator used by ``health.py``/``health_dimensions.py``). The
    Supabase/PostgREST backend only exposes named RPC functions -- there is
    no ``pg_attribute`` introspection RPC in the shipped schema, and adding
    one is out of scope for TBU-159 -- so that path (and the HTTP-proxying
    ``GatewayBackend``) logs a warning and returns instead of raising.
    """
    if not hasattr(backend, "_execute"):
        logger.warning(
            "Schema-fingerprint guard skipped: backend %s has no raw SQL "
            "introspection path (TBU-159 covers PostgresBackend only). "
            "Verify EMBEDDING_DIM=%s matches the applied schema manually.",
            type(backend).__name__,
            settings.embedding_dim,
        )
        return

    try:
        # NOTE: pgvector's `vector` type stores atttypmod == N directly (no
        # +4 VARHDRSZ-style offset the way numeric/varchar do). Verified
        # empirically against Docker postgres-scratch: applying vector(512)
        # yields atttypmod=512, format_type() = 'vector(512)'. An earlier
        # draft of this guard subtracted 4 by analogy with other typmod
        # conventions -- that was wrong for pgvector and would have flagged
        # every correctly-applied schema as mismatched.
        actual_dim = backend._execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'",
            fetch="scalar",
        )
    except Exception as e:
        # Schema may not be applied yet (fresh install / `ogham init`) --
        # don't block startup on a guard that assumes the table exists.
        logger.warning("Schema-fingerprint check skipped: %s", e)
        return

    if actual_dim is None:
        # memories.embedding column not found -- same "not applied yet" case.
        return

    if actual_dim != settings.embedding_dim:
        raise RuntimeError(
            f"Schema dim mismatch: DB memories.embedding is vector({actual_dim}) "
            f"but settings.embedding_dim={settings.embedding_dim}. "
            f"Fix ONE of: "
            f"(a) re-embed all rows to {settings.embedding_dim} and re-apply schema, or "
            f"(b) swap EMBEDDING_DIM={actual_dim} in .env to match the applied schema. "
            f"Ogham refuses to serve with silent dim mismatch (writes succeed, reads fail "
            f"asymmetrically)."
        )


def get_backend() -> DatabaseBackend:
    """Return (and lazily create) the singleton backend instance."""
    global _backend
    if _backend is None:
        from ogham.config import settings

        backend_name = getattr(settings, "database_backend", "supabase")
        if backend_name == "gateway":
            from ogham.backends.gateway import GatewayBackend

            new_backend: DatabaseBackend = GatewayBackend(
                settings.gateway_url, settings.gateway_api_key
            )
        elif backend_name == "postgres":
            from ogham.backends.postgres import PostgresBackend

            new_backend = PostgresBackend()
        else:
            from ogham.backends.supabase import SupabaseBackend

            new_backend = SupabaseBackend()
        # Validate BEFORE publishing to the singleton -- if this raises, the
        # next get_backend() call retries from scratch instead of silently
        # returning the already-flagged mismatched backend.
        _validate_schema_fingerprint(new_backend, settings)
        _backend = new_backend
    return _backend


def get_client():
    """Backwards-compatible access to the underlying database client.

    Only works for backends that expose ``_get_client()`` (e.g. SupabaseBackend).
    """
    backend = get_backend()
    if not hasattr(backend, "_get_client"):
        raise RuntimeError(f"Backend {type(backend).__name__!r} does not expose a raw client")
    return cast(Any, backend)._get_client()


def _reset_entity_graph() -> None:
    """Reset the entity-graph singleton. Used by tests."""
    global _entity_graph
    _entity_graph = None


def get_entity_graph_and_vocab() -> tuple[EntityGraph, frozenset[str]]:
    """Return (and lazily create) the singleton EntityGraph + allowed predicate vocab.

    Mirrors ``get_backend()`` / ``get_client()`` above: composes a concrete
    ``PostgresEntityGraph`` / ``SupabaseEntityGraph`` from the already-cached
    ``DatabaseBackend`` instance's pool/client (``_get_pool()`` /
    ``_get_client()``) rather than opening a second connection. This is the
    entity-graph tools' composition root -- ``ogham.tools.entity_graph`` calls
    this facade instead of importing ``ogham.postgres.entity_graph`` /
    ``ogham.supabase.entity_graph`` directly, keeping the tools module
    backend-agnostic (same boundary ``get_client()`` already enforces for the
    memory tools).

    Vocab is the hard-coded ``ogham.entity_graph.V1_PREDICATES`` constant
    (see that module's docstring for why) rather than a query against
    ``entity_edge_predicates`` -- no DB round-trip at server start.

    Raises:
        RuntimeError: if the configured backend is ``gateway`` -- no
            GatewayEntityGraph exists yet (tracked as a follow-up).
    """
    global _entity_graph
    if _entity_graph is None:
        backend = get_backend()
        from ogham.config import settings

        backend_name = getattr(settings, "database_backend", "supabase")
        if backend_name == "postgres":
            from ogham.postgres.entity_graph import PostgresEntityGraph

            pool = cast(Any, backend)._get_pool()
            _entity_graph = PostgresEntityGraph(pool, V1_PREDICATES)
        elif backend_name == "gateway":
            raise RuntimeError(
                "entity graph tools are not supported on the gateway backend yet -- "
                "switch DATABASE_BACKEND to 'postgres' or 'supabase' to use "
                "store_triple / query_join."
            )
        else:
            from ogham.supabase.entity_graph import SupabaseEntityGraph

            client = cast(Any, backend)._get_client()
            _entity_graph = SupabaseEntityGraph(client, V1_PREDICATES)
    return _entity_graph, V1_PREDICATES


# ── Thin delegates — one per public function ────────────────────────────


def store_memory(
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
    return get_backend().store_memory(
        content,
        embedding,
        profile,
        metadata,
        source,
        tags,
        expires_at,
        importance=importance,
        surprise=surprise,
        recurrence_days=recurrence_days,
    )


def get_memory_by_id(memory_id: str, profile: str) -> dict[str, Any] | None:
    return get_backend().get_memory_by_id(memory_id, profile)


def find_by_metadata_kv(key: str, value: str, profile: str) -> dict[str, Any] | None:
    """Return the first memory in ``profile`` whose ``metadata[key] == value``, or None.

    Used by tracker importers (e.g. ``ogham.tools.import_linear``) to dedupe
    on ``metadata.tracker_external_id`` before re-storing an issue.

    # build-less: linear scan over get_all_memories_full() -- every backend
    # (postgres, supabase, gateway) already implements that method, so this
    # needs zero backend-protocol or schema changes. Fine for tracker-import
    # dedupe (tens-hundreds of issues per run). Upgrade trigger: if a profile
    # grows large enough (~thousands of memories) that per-issue full scans
    # become a measurable cost, add an indexed metadata lookup on the backend
    # instead.
    """
    for memory in get_backend().get_all_memories_full(profile):
        if (memory.get("metadata") or {}).get(key) == value:
            return memory
    return None


def store_memories_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return get_backend().store_memories_batch(rows)


def upsert_memory(memory: dict[str, Any]) -> dict[str, Any]:
    """INSERT or UPDATE a memory keyed by ``id``. Delegates to the active backend."""
    return get_backend().upsert_memory(memory)


def search_memories(
    query_embedding: list[float],
    profile: str,
    threshold: float | None = None,
    limit: int | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    return get_backend().search_memories(query_embedding, profile, threshold, limit, tags, source)


def batch_check_duplicates(
    query_embeddings: list[list[float]],
    profile: str,
    threshold: float = 0.8,
) -> list[bool]:
    return get_backend().batch_check_duplicates(query_embeddings, profile, threshold)


def hybrid_search_memories(
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
    return get_backend().hybrid_search_memories(
        query_text,
        query_embedding,
        profile,
        limit,
        tags,
        source,
        profiles,
        query_entity_tags=query_entity_tags,
        recency_decay=recency_decay,
    )


def graph_augmented_search(
    query_text: str,
    query_embedding: list[float],
    profile: str,
    limit: int = 10,
    graph_depth: int = 1,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search + follow relationship edges for connected memories."""
    initial = hybrid_search_memories(query_text, query_embedding, profile, limit, tags, source)
    if not initial or graph_depth < 1:
        return initial

    seen_ids = {r["id"] for r in initial}
    augmented = list(initial)

    for result in initial[:5]:
        related = get_related_memories(result["id"], depth=graph_depth, min_strength=0.5)
        for rel in related:
            if rel["id"] not in seen_ids:
                seen_ids.add(rel["id"])
                rel["relevance"] = result.get("relevance", 0.5) * rel.get("edge_strength", 0.5)
                augmented.append(rel)

    augmented.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return augmented[:limit]


def list_recent_memories(
    profile: str,
    limit: int = 10,
    source: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    return get_backend().list_recent_memories(profile, limit, source, tags)


def get_memory_stats(profile: str) -> dict[str, Any]:
    return get_backend().get_memory_stats(profile)


def get_all_memories_full(profile: str) -> list[dict[str, Any]]:
    return get_backend().get_all_memories_full(profile)


def get_all_memories_content(profile: str | None = None) -> list[dict[str, Any]]:
    return get_backend().get_all_memories_content(profile)


def list_profiles() -> list[dict[str, Any]]:
    return get_backend().list_profiles()


def batch_update_embeddings(ids: list[str], embeddings: list[list[float]]) -> int:
    return get_backend().batch_update_embeddings(ids, embeddings)


def record_access(memory_ids: list[str]) -> None:
    return get_backend().record_access(memory_ids)


def update_confidence(memory_id: str, signal: float, profile: str) -> float:
    return get_backend().update_confidence(memory_id, signal, profile)


def delete_memory(memory_id: str, profile: str) -> bool:
    return get_backend().delete_memory(memory_id, profile)


def find_memory_ids_by_prefix(prefix: str, profile: str, limit: int = 10) -> list[str]:
    return get_backend().find_memory_ids_by_prefix(prefix, profile, limit)


def update_memory(memory_id: str, updates: dict[str, Any], profile: str) -> dict[str, Any]:
    return get_backend().update_memory(memory_id, updates, profile)


def get_profile_ttl(profile: str) -> int | None:
    return get_backend().get_profile_ttl(profile)


def set_profile_ttl(profile: str, ttl_days: int | None) -> dict[str, Any]:
    return get_backend().set_profile_ttl(profile, ttl_days)


def cleanup_expired(profile: str) -> int:
    return get_backend().cleanup_expired(profile)


def count_expired(profile: str) -> int:
    return get_backend().count_expired(profile)


def apply_hebbian_decay(profile: str, batch_size: int = 1000) -> int:
    return get_backend().apply_hebbian_decay(profile, batch_size)


def count_decay_eligible(profile: str) -> int:
    return get_backend().count_decay_eligible(profile)


def emit_audit_event(**kwargs: Any) -> None:
    backend = cast(Any, get_backend())
    backend.emit_audit_event(**kwargs)


def query_audit_log(
    profile: str, limit: int = 50, operation: str | None = None
) -> list[dict[str, Any]]:
    return get_backend().query_audit_log(profile, limit, operation)


def spread_entity_activation(
    entity_tags: list[str],
    profile: str,
    max_depth: int = 2,
    decay: float = 0.65,
    min_activation: float = 0.05,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    backend = cast(Any, get_backend())
    return cast(
        list[dict[str, Any]],
        backend.spread_entity_activation(
            entity_tags,
            profile,
            max_depth,
            decay,
            min_activation,
            max_results,
        ),
    )


def auto_link_memory(
    memory_id: str,
    embedding: list[float],
    profile: str,
    threshold: float = 0.85,
    max_links: int = 5,
) -> int:
    return get_backend().auto_link_memory(memory_id, embedding, profile, threshold, max_links)


def link_unlinked_memories(
    profile: str,
    threshold: float = 0.85,
    max_links: int = 5,
    batch_size: int = 100,
) -> int:
    return get_backend().link_unlinked_memories(profile, threshold, max_links, batch_size)


def explore_memory_graph(
    query_text: str,
    query_embedding: list[float],
    profile: str,
    limit: int = 5,
    depth: int = 1,
    min_strength: float = 0.5,
    tags: list[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    return get_backend().explore_memory_graph(
        query_text, query_embedding, profile, limit, depth, min_strength, tags, source
    )


def create_relationship(
    source_id: str,
    target_id: str,
    relationship: str,
    strength: float = 1.0,
    created_by: str = "user",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_backend().create_relationship(
        source_id, target_id, relationship, strength, created_by, metadata
    )


def get_related_memories(
    memory_id: str,
    depth: int = 1,
    min_strength: float = 0.5,
    relationship_types: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return get_backend().get_related_memories(
        memory_id, depth, min_strength, relationship_types, limit
    )


def in_result_contradictions(profile: str, memory_ids: list[str]) -> list[dict[str, Any]]:
    """Contradiction pairs with BOTH endpoints inside memory_ids (migration 047).

    Returns [{"stale_id": .., "newer_id": .., "strength": ..}], oriented by
    created_at. Backends that cannot express the filter return [].
    """
    return cast(Any, get_backend()).in_result_contradictions(profile, memory_ids)


def gap_out_of_result_contradictions(
    profile: str, memory_ids: list[str], *, sample_size: int = 10
) -> dict[str, Any]:
    """Contradiction edges where one endpoint is in memory_ids and the other is outside it.

    Returns {"count": int, "pairs": [{"in_result_id": .., "other_id": .., "strength": ..}]}.
    Dispatches to gap_contradictions_for_ids SQL function (migration 039).
    """
    return cast(Any, get_backend()).gap_out_of_result_contradictions(
        profile, memory_ids, sample_size=sample_size
    )


def walk_memory_graph(
    start_id: str,
    depth: int = 1,
    direction: str = "both",
    min_strength: float = 0.0,
    relationship_types: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Direction-aware graph walk from a known memory id.

    Thin facade over backend.wiki_walk_graph (migration 031 function
    `wiki_walk_graph`). PostgresBackend dispatches via psycopg;
    SupabaseBackend via PostgREST rpc(). Direction validation +
    depth bounds checked client-side here so callers get a clean
    error before the round-trip; the SQL function also enforces.

      * 'outgoing' -- edges where memory_id is source (this -> other)
      * 'incoming' -- edges where memory_id is target (other -> this)
      * 'both' -- traditional bidirectional traversal
    """
    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(f"direction must be 'outgoing', 'incoming', or 'both'; got {direction!r}")
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if depth > 5:
        raise ValueError("depth must be <= 5; use explore_knowledge for broader walks")

    backend = cast(Any, get_backend())
    return cast(
        list[dict[str, Any]],
        backend.wiki_walk_graph(
            start_id=start_id,
            max_depth=depth,
            direction=direction,
            min_strength=min_strength,
            relationship_types=relationship_types,
            result_limit=limit,
        ),
    )
