"""Entity-graph domain module.

Pure Python types + Protocol. NO SQL, NO MCP wrapping. See the design spec
for the boundary rules: ``docs/superpowers/specs/2026-07-01-typed-edge-context-graph-design.md``.

Exports:
    - Predicate: NewType wrapping str, validated against the vocab table
    - Entity, EntityEdge, JoinResult: frozen dataclasses
    - EntityGraph: runtime_checkable Protocol implemented by backend classes
    - make_predicate: construction-site validator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, NewType, Protocol, runtime_checkable
from uuid import UUID

Predicate = NewType("Predicate", str)

# v1 predicate vocabulary -- mirrors the 16 seed rows in
# sql/migrations/042_entity_edge_predicates.sql exactly. Hard-coded here
# (rather than queried from `entity_edge_predicates` at server start)
# because the seed rows are effectively part of the v0.16 contract: they
# only change via a new migration, and a DB round-trip on every server
# boot adds a failure mode with no upside. Two sources of truth (this
# constant + the SQL seed) is an accepted tradeoff -- if they drift,
# `tests/test_schema_parity.py::test_predicate_vocab_seed_present` catches
# the SQL side, and this constant is the single Python-side source
# `ogham.database.get_entity_graph_and_vocab` loads from.
V1_PREDICATES: frozenset[str] = frozenset(
    {
        "DEPENDS_ON",
        "DEPENDED_ON_BY",
        "OWNS",
        "OWNED_BY",
        "ASSIGNED_TO",
        "HAS_ASSIGNEE",
        "DECIDED",
        "MENTIONS",
        "BLOCKS",
        "BLOCKED_BY",
        "PART_OF",
        "CONTAINS",
        "SUPPORTS",
        "CONTRADICTS",
        "EVOLVED_INTO",
        "RELATED_TO",
    }
)


def make_predicate(value: str, allowed: Iterable[str]) -> Predicate:
    """Construct a Predicate after validating against the vocabulary.

    Args:
        value: predicate string, e.g. "DEPENDS_ON".
        allowed: iterable of allowed predicate strings, typically the set
            of predicates loaded from ``entity_edge_predicates``.

    Raises:
        ValueError: if ``value`` is not in ``allowed``.
    """
    allowed_set = set(allowed)
    if value not in allowed_set:
        raise ValueError(f"predicate {value!r} not in vocabulary ({len(allowed_set)} allowed)")
    return Predicate(value)


@dataclass(frozen=True)
class Entity:
    id: int
    canonical_name: str
    entity_type: str


@dataclass(frozen=True)
class EntityEdge:
    id: int
    subject_id: int
    predicate: Predicate
    object_id: int
    profile: str
    fact_id: UUID | None
    strength: float
    metadata: dict
    valid_from: datetime
    valid_to: datetime | None


@dataclass(frozen=True)
class JoinResult:
    """Result of an EntityGraph.query_join traversal.

    entities: Entity list in BFS insertion order -- start entity first, then
        each hop's discovered entities appended in the order the traversal
        encountered them. Deterministic given a deterministic backend seed
        order. Callers wanting the exact path from start to end should read
        `edges` instead (edges are in hop order, one per traversed hop).
    edges: EntityEdges in hop order (edges[0] is hop 1, edges[1] is hop 2, ...).
    citations: fact_ids collected from every edge that had one, in edge order.
    """

    entities: list[Entity]
    edges: list[EntityEdge]
    citations: list[UUID] = field(default_factory=list)


@runtime_checkable
class EntityGraph(Protocol):
    """Contract implemented by concrete backends (PostgresEntityGraph, SupabaseEntityGraph)."""

    def store_triple(
        self,
        subject: str | int,
        predicate: Predicate,
        object_: str | int,
        source_memory_id: UUID | None,
        profile: str,
        metadata: dict | None = None,
    ) -> int:
        """Insert a new edge, superseding any current (subject, predicate, *, profile).

        Returns the new edge id.
        """
        ...

    def query_join(
        self,
        start_entity: str | int,
        predicate_path: list[Predicate],
        profile: str,
        hop_limit: int,
        direction: str = "outgoing",
    ) -> JoinResult | None:
        """Walk the graph along ``predicate_path`` starting at ``start_entity``.

        Returns ``None`` if no path resolves; otherwise a JoinResult.
        ``hop_limit`` is required (no default) per TBU-109.
        """
        ...

    def add_alias(self, entity_id: int, alias: str, profile: str) -> None:
        """Record an additional surface form pointing at ``entity_id``."""
        ...

    def resolve_alias(self, name_or_id: str | int, profile: str) -> Entity | None:
        """Look up an Entity by canonical name, alias, or id. Returns None if unresolvable."""
        ...
