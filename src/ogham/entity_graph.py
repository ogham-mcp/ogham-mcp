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

# Portable URIs for the v1 predicates (TBU-129). Mirrors the seed in
# sql/migrations/045_predicate_uris.sql + the three schema files. Every
# predicate has a stable ogham_uri identity; schema_org_uri is populated ONLY
# where a Schema.org property is a genuine equivalent (WebFetch-verified
# 2026-07-06); iirds_uri is reserved for TBU-128 (all None here). Kept in
# Python (not queried from the DB) for the same reason V1_PREDICATES is:
# no server-boot round-trip. Drift is caught by
# tests/test_predicate_uris.py + tests/test_schema_parity.py.
_OGHAM_VOCAB_NS = "https://ogham-mcp.dev/vocab#"

_SCHEMA_ORG_URIS: dict[str, str] = {
    "OWNS": "https://schema.org/owns",
    "OWNED_BY": "https://schema.org/owner",
    "MENTIONS": "https://schema.org/mentions",
    "PART_OF": "https://schema.org/isPartOf",
    "CONTAINS": "https://schema.org/hasPart",
}

PREDICATE_URIS: dict[str, dict[str, str | None]] = {
    pred: {
        "ogham": f"{_OGHAM_VOCAB_NS}{pred}",
        "schema_org": _SCHEMA_ORG_URIS.get(pred),
        "iirds": None,
    }
    for pred in V1_PREDICATES
}


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


# Entity types produced by ``extract_entities``. Used to recognise a qualified
# reference like "error:KeyError" so it can be resolved on the exact natural key
# rather than by name alone -- ``entities`` is UNIQUE (canonical_name,
# entity_type), so a bare name is not a key.
KNOWN_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "entity",
        "file",
        "error",
        "quantity",
        "location",
        "event",
        "activity",
        "emotion",
        "relationship",
        "preference",
        "person",
    }
)


def split_entity_ref(ref: str) -> tuple[str | None, str]:
    """Split ``"error:KeyError"`` into ``("error", "KeyError")``.

    Returns ``(None, ref)`` when ``ref`` carries no recognised type prefix, so
    callers can fall back to a name-only lookup.

    Only splits on a KNOWN type prefix. A name that merely contains a colon --
    a URL, a namespaced id, a Windows path -- is left whole, which is why this
    does not simply ``split(":", 1)``.
    """
    prefix, sep, rest = ref.partition(":")
    if sep and rest and prefix in KNOWN_ENTITY_TYPES:
        return prefix, rest
    return None, ref


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
    derived_from: list[dict] = field(default_factory=list)


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


@dataclass(frozen=True)
class ProvenanceTree:
    """Result of trace_provenance -- the derivation lineage of an edge.

    nodes: every edge visited (start edge first, then ancestors in BFS order).
    links: derivation links followed -- {"from_edge_id": int, "to_edge_id": int,
        "reasoning": str | None}.
    root_memories: deduped source_memory_ids + fact_ids reached at the leaves (uuid strings).
    """

    nodes: list[EntityEdge]
    links: list[dict]
    root_memories: list[str] = field(default_factory=list)


def validate_derived_from(value: list[dict] | None) -> list[dict]:
    """Validate + normalize a derived_from array. None -> []. Each element must
    have at least one of source_edge_id (int) / source_memory_id (str);
    reasoning (str) is optional. Shape-only -- no FK / existence check."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("derived_from must be a list of objects")
    out: list[dict] = []
    for i, el in enumerate(value):
        if not isinstance(el, dict):
            raise ValueError(f"derived_from[{i}] must be an object")
        se = el.get("source_edge_id")
        sm = el.get("source_memory_id")
        if se is None and sm is None:
            raise ValueError(f"derived_from[{i}] needs source_edge_id or source_memory_id")
        if se is not None and (isinstance(se, bool) or not isinstance(se, int)):
            raise ValueError(f"derived_from[{i}].source_edge_id must be an int")
        if sm is not None and not isinstance(sm, str):
            raise ValueError(f"derived_from[{i}].source_memory_id must be a uuid string")
        norm: dict = {}
        if se is not None:
            norm["source_edge_id"] = se
        if sm is not None:
            norm["source_memory_id"] = sm
        if el.get("reasoning") is not None:
            norm["reasoning"] = str(el["reasoning"])
        out.append(norm)
    return out


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
        derived_from: list[dict] | None = None,
    ) -> int:
        """Insert a new edge, superseding any current (subject, predicate, object, profile).

        The supersession key includes the OBJECT, matching the
        ``entity_edges_current_uq`` partial unique index -- so one subject may
        hold several current objects under the same predicate. (This docstring
        previously claimed a wildcard on the object, which contradicted both
        backends and the index, and would have made list-valued predicates
        impossible to round-trip through an OKF bundle.)

        Returns the new edge id.
        """
        ...

    def fetch_edge(self, edge_id: int, profile: str) -> EntityEdge | None:
        """Fetch an edge by id, IGNORING valid_to (provenance is historical). None if absent."""
        ...

    def find_citing_edges(
        self, *, source_edge_id: int | None, source_memory_id: str | None, profile: str
    ) -> list[EntityEdge]:
        """Edges whose derived_from cites the given source (containment). Current + historical."""
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

    # -- enumeration (TBU-130) ---------------------------------------
    # query_join walks and fetch_edge fetches one; nothing could list a whole
    # profile. OKF export needs that.

    def find_entity(self, canonical_name: str, entity_type: str) -> int | None:
        """Read-only lookup on the exact natural key. None if absent.

        Distinct from ``resolve_alias`` on a qualified ``type:name`` string,
        which can only split on a type it RECOGNISES -- and the recognised set
        is the 11 types ``extract_entities`` produces. An imported bundle may
        carry any type at all, so import needs a lookup that takes the two parts
        separately rather than re-parsing them out of a string.

        Distinct from ``upsert_entity`` because a dry run must not create.
        """
        ...

    def upsert_entity(self, canonical_name: str, entity_type: str) -> int:
        """Get or create an entity by its natural key, returning its id.

        The natural key is (canonical_name, entity_type) -- the table's UNIQUE
        constraint. A bare name is not a key (TBU-274).

        There was no Python-level way to create an entity before this: the only
        creation path was the ``link_memory_entities`` RPC, which needs a memory
        to hang the entity off. Graph import has no memory, so it needed the
        primitive that was missing rather than a memory-shaped detour.

        Deliberately NOT profile-scoped: ``entities`` is global, and profiles are
        a convenience namespace rather than a trust boundary (decided
        2026-08-20). ``mention_count`` is left alone -- an import is not a
        mention, and inflating it would shift ranking in every profile.
        """
        ...

    def list_entities(self, profile: str) -> list[Entity]:
        """Every entity reachable in ``profile``, ordered by id.

        ``entities`` has no profile column -- it is global, scoped only through
        ``memory_entities`` and ``entity_edges``. So this is the union of both,
        not a table scan. The union is load-bearing for OKF export: because it
        includes every edge subject AND object, every endpoint of every listed
        edge is guaranteed to be a listed entity, which is what makes dangling
        links structurally impossible in an exported bundle.
        """
        ...

    def list_edges(self, profile: str, *, current_only: bool = True) -> list[EntityEdge]:
        """Every edge in ``profile``, ordered by id.

        ``current_only`` keeps to ``valid_to IS NULL``, matching
        ``entity_edges_current_uq``. Pass False to include superseded history.
        """
        ...

    def list_aliases(self, profile: str) -> dict[int, list[str]]:
        """entity_id -> its aliases in ``profile``. Entities with none are absent."""
        ...
