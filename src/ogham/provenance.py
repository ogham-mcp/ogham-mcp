"""Provenance-chain walks over the entity graph (TBU-125/126).

Backend-agnostic BFS: works against anything exposing the EntityGraph
``fetch_edge`` + ``find_citing_edges`` primitives, so the tree logic is
unit-testable with a fake graph and not duplicated per backend.
"""

from __future__ import annotations

from typing import Protocol

from ogham.entity_graph import EntityEdge, ProvenanceTree


class _ProvenanceGraph(Protocol):
    def fetch_edge(self, edge_id: int, profile: str) -> EntityEdge | None: ...
    def find_citing_edges(
        self, *, source_edge_id: int | None, source_memory_id: str | None, profile: str
    ) -> list[EntityEdge]: ...


def trace_provenance(
    graph: _ProvenanceGraph, edge_id: int, profile: str, max_depth: int = 10
) -> ProvenanceTree:
    """Walk backward through derived_from to root evidence. fact_id folds in as a root."""
    start = graph.fetch_edge(edge_id, profile)
    if start is None:
        return ProvenanceTree(nodes=[], links=[], root_memories=[])

    order: list[int] = [start.id]
    nodes_by_id: dict[int, EntityEdge] = {start.id: start}
    links: list[dict] = []
    roots: list[str] = []
    seen_roots: set[str] = set()

    def add_root(mid: str | None) -> None:
        if mid and mid not in seen_roots:
            seen_roots.add(mid)
            roots.append(mid)

    add_root(str(start.fact_id) if start.fact_id else None)
    frontier = [start]
    visited = {start.id}
    depth = 0
    while frontier and depth < max_depth:
        nxt: list[EntityEdge] = []
        for edge in frontier:
            for el in edge.derived_from:
                add_root(el.get("source_memory_id"))
                se = el.get("source_edge_id")
                if se is None:
                    continue
                links.append(
                    {"from_edge_id": edge.id, "to_edge_id": se, "reasoning": el.get("reasoning")}
                )
                if se in visited:
                    continue
                visited.add(se)
                parent = graph.fetch_edge(se, profile)
                if parent is None:
                    continue  # dangling id -- tolerated
                nodes_by_id[se] = parent
                order.append(se)
                add_root(str(parent.fact_id) if parent.fact_id else None)
                nxt.append(parent)
        frontier = nxt
        depth += 1

    return ProvenanceTree(nodes=[nodes_by_id[i] for i in order], links=links, root_memories=roots)


def find_derivatives(
    graph: _ProvenanceGraph, source_id: int | str, profile: str, max_depth: int = 10
) -> list[EntityEdge]:
    """Walk forward: edges whose derived_from cites source_id, transitively."""

    def _citing(sid: int | str) -> list[EntityEdge]:
        if isinstance(sid, int):
            return graph.find_citing_edges(
                source_edge_id=sid, source_memory_id=None, profile=profile
            )
        if sid.isdigit():
            return graph.find_citing_edges(
                source_edge_id=int(sid), source_memory_id=None, profile=profile
            )
        return graph.find_citing_edges(source_edge_id=None, source_memory_id=sid, profile=profile)

    results: list[EntityEdge] = []
    seen: set[int] = set()
    frontier = _citing(source_id)
    depth = 0
    while frontier and depth < max_depth:
        nxt: list[EntityEdge] = []
        pending: set[int] = set()  # dedupes fan-in within this pass (diamond derivation)
        for edge in frontier:
            if edge.id in seen:
                continue
            seen.add(edge.id)
            results.append(edge)
            for child in graph.find_citing_edges(
                source_edge_id=edge.id, source_memory_id=None, profile=profile
            ):
                if child.id not in seen and child.id not in pending:
                    pending.add(child.id)
                    nxt.append(child)
        frontier = nxt
        depth += 1
    return results
