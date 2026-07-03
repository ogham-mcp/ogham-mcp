"""MCP tool wrappers over EntityGraph.

The ``_impl`` functions below (``store_triple_impl``, ``query_join_impl``,
``_serialize_result``) are pure: they import ONLY from ``ogham.entity_graph``
(the domain module) and take the concrete backend as an injected ``graph``
parameter, typed against the ``EntityGraph`` Protocol. They MUST NOT import
``ogham.postgres.entity_graph`` or ``ogham.supabase.entity_graph`` directly
-- easy to unit-test without spinning FastMCP or a real backend.

The ``@mcp.tool`` wrapper(s) below the impls compose a concrete backend via
the ``ogham.database.get_entity_graph_and_vocab()`` facade (mirrors
``get_backend()`` / ``get_client()``, which the memory tools already use the
same way) -- so this module still never imports a concrete backend module
directly; the facade is the trust boundary. Tools self-register here at
import time (``@mcp.tool`` decorator), same convention as every other file
under ``ogham/tools/`` -- ``ogham.server`` imports this module once for that
side effect.
"""

from __future__ import annotations

from typing import Annotated, Iterable
from uuid import UUID

from pydantic import BeforeValidator

from ogham.app import mcp
from ogham.database import get_entity_graph_and_vocab
from ogham.entity_graph import EntityGraph, JoinResult, Predicate, make_predicate
from ogham.tools.memory import DictAny, _coerce_list, get_active_profile

# `predicate_path` is required (no default) so it can't reuse `ListStr`
# (`list[str] | None`) from ogham.tools.memory -- same BeforeValidator
# coercion, but the annotation itself stays non-optional.
_PredicatePath = Annotated[list[str], BeforeValidator(_coerce_list)]


def store_triple_impl(
    *,
    graph: EntityGraph,
    allowed_predicates: Iterable[str],
    subject: str,
    predicate: str,
    object_: str,
    profile: str,
    source_memory_id: str | None,
    metadata: dict | None,
) -> dict:
    """Store a (subject, predicate, object) triple with write-time supersession."""
    pred = make_predicate(predicate, allowed_predicates)
    fact_id = UUID(source_memory_id) if source_memory_id else None
    edge_id = graph.store_triple(
        subject=subject,
        predicate=pred,
        object_=object_,
        source_memory_id=fact_id,
        profile=profile,
        metadata=metadata,
    )
    return {"edge_id": edge_id}


@mcp.tool
def store_triple(
    subject: str,
    predicate: str,
    object_: str,
    profile: str | None = None,
    source_memory_id: str | None = None,
    metadata: DictAny = None,
) -> dict:
    """Store a typed (subject, predicate, object) triple in the entity graph.

    Write-time supersession: any current edge with the same (subject,
    predicate, object, profile) has its valid_to set to now(); the new
    edge becomes current. ``predicate`` must be in the v1 vocabulary (see
    entity_edge_predicates seed / ``ogham.entity_graph.V1_PREDICATES``).

    Args:
        profile: Target profile. Defaults to the active profile.
    """
    p = profile or get_active_profile()
    graph, allowed = get_entity_graph_and_vocab()
    return store_triple_impl(
        graph=graph,
        allowed_predicates=allowed,
        subject=subject,
        predicate=predicate,
        object_=object_,
        profile=p,
        source_memory_id=source_memory_id,
        metadata=metadata,
    )


def query_join_impl(
    *,
    graph: EntityGraph,
    allowed_predicates: Iterable[str],
    start_entity: str,
    predicate_path: list[str],
    profile: str,
    hop_limit: int,
    direction: str = "outgoing",
) -> dict:
    """Walk the graph along predicate_path from start_entity.

    Returns a serialized JoinResult, or the empty-result shape if no path
    resolves.
    """
    if hop_limit is None or hop_limit < 1:
        raise ValueError("hop_limit is required and must be >= 1")
    typed_path: list[Predicate] = [make_predicate(p, allowed_predicates) for p in predicate_path]
    result = graph.query_join(
        start_entity=start_entity,
        predicate_path=typed_path,
        profile=profile,
        hop_limit=hop_limit,
        direction=direction,
    )
    if result is None:
        return {"entities": [], "edges": [], "citations": []}
    return _serialize_result(result)


def _serialize_result(r: JoinResult) -> dict:
    return {
        "entities": [
            {"id": e.id, "canonical_name": e.canonical_name, "entity_type": e.entity_type}
            for e in r.entities
        ],
        "edges": [
            {
                "id": e.id,
                "subject_id": e.subject_id,
                "predicate": str(e.predicate),
                "object_id": e.object_id,
                "profile": e.profile,
                "fact_id": str(e.fact_id) if e.fact_id else None,
                "strength": e.strength,
                "metadata": e.metadata,
                "valid_from": e.valid_from.isoformat(),
                "valid_to": e.valid_to.isoformat() if e.valid_to else None,
            }
            for e in r.edges
        ],
        "citations": [str(c) for c in r.citations],
    }


@mcp.tool
def query_join(
    start_entity: str,
    predicate_path: _PredicatePath,
    hop_limit: int,
    direction: str = "outgoing",
    profile: str | None = None,
) -> dict:
    """Walk the entity graph along ``predicate_path`` from ``start_entity``.

    Strict predicate-path traversal: no fuzzy match, no semantic ranking.
    The path either resolves or returns the empty-result shape. ``predicate``
    values in ``predicate_path`` must be in the v1 vocabulary (see
    entity_edge_predicates seed / ``ogham.entity_graph.V1_PREDICATES``).
    Reads only current edges (``valid_to IS NULL``).

    Args:
        hop_limit: Maximum number of hops to traverse. Required -- callers
            must declare intent (per TBU-109); no default is provided.
        profile: Target profile. Defaults to the active profile.
    """
    p = profile or get_active_profile()
    graph, allowed = get_entity_graph_and_vocab()
    return query_join_impl(
        graph=graph,
        allowed_predicates=allowed,
        start_entity=start_entity,
        predicate_path=predicate_path,
        profile=p,
        hop_limit=hop_limit,
        direction=direction,
    )
