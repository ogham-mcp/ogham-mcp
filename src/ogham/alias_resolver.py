"""AliasResolver -- thin service composing over EntityGraph.

Extracted per the design-rules critique so both store_triple and query_join
share the same resolution logic (DRY).
"""

from __future__ import annotations

from ogham.entity_graph import Entity, EntityGraph


class AliasResolver:
    """Resolve a name or id to an Entity through an EntityGraph backend."""

    def __init__(self, graph: EntityGraph):
        self._graph = graph

    def resolve(self, name_or_id: str | int, profile: str) -> Entity | None:
        return self._graph.resolve_alias(name_or_id, profile)
