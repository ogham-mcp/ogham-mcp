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
from ogham.entity_graph import (
    PREDICATE_URIS,
    EntityEdge,
    EntityGraph,
    JoinResult,
    Predicate,
    make_predicate,
    validate_derived_from,
)
from ogham.provenance import find_derivatives as _find
from ogham.provenance import trace_provenance as _trace
from ogham.tools.memory import DictAny, _coerce_list, get_active_profile

# `predicate_path` is required (no default) so it can't reuse `ListStr`
# (`list[str] | None`) from ogham.tools.memory -- same BeforeValidator
# coercion, but the annotation itself stays non-optional.
_PredicatePath = Annotated[list[str], BeforeValidator(_coerce_list)]

# `derived_from` is a list of objects (not list[str]/dict like the existing
# ListStr/DictAny helpers), reusing the same JSON-string coercion so callers
# over stdio can pass either a real list or a JSON-encoded string.
_DerivedFrom = Annotated[list[dict] | None, BeforeValidator(_coerce_list)]


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
    derived_from: list[dict] | None = None,
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
        derived_from=validate_derived_from(derived_from),
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
    derived_from: _DerivedFrom = None,
) -> dict:
    """Store a typed (subject, predicate, object) triple in the entity graph.

    Write-time supersession: any current edge with the same (subject,
    predicate, object, profile) has its valid_to set to now(); the new
    edge becomes current. ``predicate`` must be in the v1 vocabulary (see
    entity_edge_predicates seed / ``ogham.entity_graph.V1_PREDICATES``).

    Args:
        profile: Target profile. Defaults to the active profile.
        derived_from: Optional lineage array recording the evidence this
            edge was derived from -- other edges and/or source memories.
            Each element: {"source_edge_id": int, "source_memory_id": str,
            "reasoning": str}, at least one of the two ids required. See
            ``trace_provenance`` / ``find_derivatives``.
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
        derived_from=derived_from,
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


def _serialize_edge(e: EntityEdge) -> dict:
    return {
        "id": e.id,
        "subject_id": e.subject_id,
        "predicate": str(e.predicate),
        "object_id": e.object_id,
        "profile": e.profile,
        "fact_id": str(e.fact_id) if e.fact_id else None,
        "strength": e.strength,
        "metadata": e.metadata,
        "derived_from": e.derived_from,
        "valid_from": e.valid_from.isoformat(),
        "valid_to": e.valid_to.isoformat() if e.valid_to else None,
    }


def _serialize_result(r: JoinResult) -> dict:
    return {
        "entities": [
            {"id": e.id, "canonical_name": e.canonical_name, "entity_type": e.entity_type}
            for e in r.entities
        ],
        "edges": [_serialize_edge(e) for e in r.edges],
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


def describe_predicates_impl(*, uris: dict[str, dict[str, str | None]]) -> list[dict]:
    """Return the predicate vocabulary with its portable URIs, sorted by name.

    Pure over an injected ``uris`` mapping (``ogham.entity_graph.PREDICATE_URIS``)
    -- no DB round-trip, mirroring the vocab-in-Python design.
    """
    return [
        {
            "predicate": pred,
            "ogham_uri": u["ogham"],
            "schema_org_uri": u["schema_org"],
            "iirds_uri": u["iirds"],
        }
        for pred, u in sorted(uris.items())
    ]


@mcp.tool
def describe_predicates() -> list[dict]:
    """List the entity-graph predicate vocabulary with portable URIs.

    Each predicate carries a stable ``ogham_uri`` identity
    (``https://ogham-mcp.dev/vocab#<PREDICATE>``) and, where a genuine
    equivalent exists, a ``schema_org_uri`` alignment. ``iirds_uri`` is
    reserved (all null pending TBU-128). Read-only; no arguments.
    """
    return describe_predicates_impl(uris=PREDICATE_URIS)


def trace_provenance_impl(*, graph, edge_id: int, profile: str, max_depth: int = 10) -> dict:
    tree = _trace(graph, edge_id, profile, max_depth)
    return {
        "nodes": [_serialize_edge(n) for n in tree.nodes],
        "links": tree.links,
        "root_memories": tree.root_memories,
    }


@mcp.tool
def trace_provenance(edge_id: int, max_depth: int = 10, profile: str | None = None) -> dict:
    """Walk an edge's derivation lineage back to root evidence (memories).

    Returns {nodes, links, root_memories}. Reads superseded edges too --
    provenance is historical. profile defaults to the active profile.
    """
    p = profile or get_active_profile()
    graph, _allowed = get_entity_graph_and_vocab()
    return trace_provenance_impl(graph=graph, edge_id=edge_id, profile=p, max_depth=max_depth)


def find_derivatives_impl(
    *, graph, source_id: int | str, profile: str, max_depth: int = 10
) -> dict:
    return {"edges": [_serialize_edge(e) for e in _find(graph, source_id, profile, max_depth)]}


@mcp.tool
def find_derivatives(source_id: str, max_depth: int = 10, profile: str | None = None) -> dict:
    """Find every edge that (transitively) cites source_id in its derived_from.

    source_id is an edge id (numeric string) or a memory uuid. Impact analysis:
    "if this fact is retracted, what depends on it?" profile defaults to active.
    """
    p = profile or get_active_profile()
    graph, _allowed = get_entity_graph_and_vocab()
    sid: int | str = int(source_id) if source_id.isdigit() else source_id
    return find_derivatives_impl(graph=graph, source_id=sid, profile=p, max_depth=max_depth)
