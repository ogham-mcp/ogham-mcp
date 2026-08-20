"""Unit tests for PostgresEntityGraph -- mocked at the psycopg connection
boundary (no real DB). Integration coverage against a live Postgres lives
in TBU-122 (tests/test_entity_graph_integration_store_triple.py, out of
scope here).

Test double shape: a fake pool/connection/cursor where ``cursor.execute``
routes on the SQL text + bound params to a small in-memory model (entities,
aliases, edges). This lets tests assert on real branching behaviour
(supersession, hop traversal, alias fallback) without hard-coding the
exact call order psycopg would see.
"""

from __future__ import annotations

from typing import cast

import pytest
from psycopg_pool import ConnectionPool

from ogham.entity_graph import Predicate
from ogham.postgres.entity_graph import PostgresEntityGraph


class _Router:
    """Routes ``cursor.execute(query, params)`` calls to canned data.

    Mirrors the four query shapes PostgresEntityGraph actually issues
    (see src/ogham/postgres/entity_graph.py): entity/alias resolution,
    entity fetch, edge supersession + insert, and hop traversal selects.

    Returns dict rows (not tuples) -- production builds the psycopg pool
    with ``row_factory=dict_row`` (src/ogham/backends/postgres.py), so this
    double must match that shape or it can't catch by-name-vs-positional
    regressions (TBU-164).
    """

    def __init__(self):
        self.entities: dict[str, int] = {}
        self.entities_by_id: dict[int, dict] = {}
        self.aliases: dict[tuple[str, str], int] = {}
        self.edges_by_subject: dict[tuple[int, str, str], list[dict]] = {}
        self.edges_by_object: dict[tuple[int, str, str], list[dict]] = {}
        self._next_edge_id = 1000
        self.calls: list[tuple[str, tuple | None]] = []

    def add_entity(self, entity_id: int, name: str, entity_type: str = "person") -> None:
        self.entities[name] = entity_id
        self.entities_by_id[entity_id] = {
            "id": entity_id,
            "canonical_name": name,
            "entity_type": entity_type,
        }

    def __call__(self, query: str, params):
        q = " ".join(query.split())
        self.calls.append((q, params))

        # Qualified natural-key lookup: canonical_name + entity_type. Added with
        # the fix for ambiguous entity refs -- `entities` is
        # UNIQUE (canonical_name, entity_type), so a bare name is not a key.
        if "SELECT id FROM entities WHERE canonical_name" in q and "entity_type = %s" in q:
            name, etype = params
            for eid, row in self.entities_by_id.items():
                if row["canonical_name"] == name and row["entity_type"] == etype:
                    return {"id": eid}
            return None

        # Unqualified lookup now returns EVERY matching row, ordered, so the
        # caller can detect ambiguity instead of silently taking LIMIT 1.
        if "SELECT id, entity_type FROM entities WHERE canonical_name" in q:
            (name,) = params
            return [
                {"id": eid, "entity_type": row["entity_type"]}
                for eid, row in sorted(self.entities_by_id.items())
                if row["canonical_name"] == name
            ]

        if "SELECT id FROM entities WHERE canonical_name" in q:
            (name,) = params
            eid = self.entities.get(name)
            return {"id": eid} if eid is not None else None

        if "SELECT entity_id FROM entity_aliases" in q:
            alias, profile = params
            eid = self.aliases.get((alias, profile))
            return {"entity_id": eid} if eid is not None else None

        if q.startswith("UPDATE entity_edges") and "valid_to = now()" in q:
            return None

        if q.startswith("UPDATE entity_edges") and "superseded_by = %s" in q:
            return None

        if q.startswith("INSERT INTO entity_edges("):
            new_id = self._next_edge_id
            self._next_edge_id += 1
            return {"id": new_id}

        if q.startswith("INSERT INTO entity_aliases"):
            return None

        if "SELECT id, canonical_name, entity_type FROM entities WHERE id" in q:
            (eid,) = params
            return self.entities_by_id.get(eid)

        if "FROM entity_edges" in q and "WHERE subject_id" in q:
            subj_id, predicate, profile = params
            return list(self.edges_by_subject.get((subj_id, predicate, profile), []))

        if "FROM entity_edges" in q and "WHERE object_id" in q:
            obj_id, predicate, profile = params
            return list(self.edges_by_object.get((obj_id, predicate, profile), []))

        raise AssertionError(f"unrouted query: {q!r} params={params!r}")


class _FakeCursor:
    def __init__(self, router: _Router):
        self._router = router
        self._last = None

    def execute(self, query, params=None):
        self._last = self._router(query, params)

    def fetchone(self):
        r = self._last
        if r is None:
            return None
        return r[0] if isinstance(r, list) else r

    def fetchall(self):
        r = self._last
        if r is None:
            return []
        return r if isinstance(r, list) else [r]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePool:
    """Stands in for ``psycopg_pool.ConnectionPool`` -- ``.connection()``
    always yields the same fake connection (fine for unit tests; we're
    not testing real pooling)."""

    def __init__(self, router: _Router):
        self.router = router
        self._cursor = _FakeCursor(router)
        self._conn = _FakeConnection(self._cursor)

    def connection(self):
        return self._conn


def _edge_row(
    id: int,
    subject_id: int,
    predicate: str,
    object_id: int,
    profile: str,
    fact_id,
    strength: float,
    metadata: dict,
    valid_from: str,
    valid_to,
    derived_from: list | None = None,
) -> dict:
    """Builds an entity_edges dict row matching the dict_row column shape.

    Column order mirrors the query_join edge SELECT in
    src/ogham/postgres/entity_graph.py: id, subject_id, predicate,
    object_id, profile, fact_id, strength, metadata, derived_from,
    valid_from, valid_to.
    """
    return {
        "id": id,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "profile": profile,
        "fact_id": fact_id,
        "strength": strength,
        "metadata": metadata,
        "derived_from": derived_from or [],
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


@pytest.fixture
def router() -> _Router:
    r = _Router()
    r.add_entity(1, "Alice")
    r.add_entity(2, "Bob")
    r.add_entity(3, "Carol")
    return r


@pytest.fixture
def fake_pool(router: _Router) -> _FakePool:
    return _FakePool(router)


@pytest.fixture
def graph(fake_pool: _FakePool) -> PostgresEntityGraph:
    # _FakePool duck-types psycopg_pool.ConnectionPool's `.connection()`
    # surface; cast so pyright accepts the constructor call without
    # PostgresEntityGraph having to widen to a Protocol just for tests.
    return PostgresEntityGraph(
        cast(ConnectionPool, fake_pool), allowed_predicates={"KNOWS", "WORKS_WITH"}
    )


# ── store_triple ─────────────────────────────────────────────────────


def test_store_triple_new_edge_returns_id_and_commits(router, fake_pool, graph):
    new_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    assert new_id == 1000
    assert fake_pool.connection().commits == 1
    # INSERT + both UPDATE (supersede + wire superseded_by) all issued.
    insert_calls = [c for c in router.calls if c[0].startswith("INSERT INTO entity_edges(")]
    assert len(insert_calls) == 1
    (subj_id, predicate, obj_id, profile, fact_id, strength, _md, _df) = insert_calls[0][1]
    assert (subj_id, predicate, obj_id, profile) == (1, "KNOWS", 2, "work")
    assert fact_id is None
    assert strength == 1.0


def test_store_triple_issues_supersede_before_insert(router, graph):
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    supersede_calls = [c for c in router.calls if "valid_to = now()" in c[0]]
    wire_calls = [c for c in router.calls if "superseded_by = %s" in c[0]]
    assert len(supersede_calls) == 1
    assert supersede_calls[0][1] == (1, "KNOWS", 2, "work")
    assert len(wire_calls) == 1
    assert wire_calls[0][1] == (1000, 1, "KNOWS", 2, "work")


def test_store_triple_self_referential_raises(router, graph):
    with pytest.raises(ValueError, match="self-referential"):
        graph.store_triple("Alice", Predicate("KNOWS"), "Alice", None, "work")


def test_store_triple_unresolvable_subject_raises(router, graph):
    with pytest.raises(ValueError, match="cannot resolve"):
        graph.store_triple("Ghost", Predicate("KNOWS"), "Bob", None, "work")


def test_store_triple_unresolvable_object_raises(router, graph):
    with pytest.raises(ValueError, match="cannot resolve"):
        graph.store_triple("Alice", Predicate("KNOWS"), "Ghost", None, "work")


def test_store_triple_accepts_int_ids_directly(router, graph):
    new_id = graph.store_triple(1, Predicate("KNOWS"), 2, None, "work")
    assert new_id == 1000


# ── query_join ───────────────────────────────────────────────────────


def test_query_join_single_hop_outgoing_success(router, graph):
    router.edges_by_subject[(1, "KNOWS", "work")] = [
        _edge_row(5001, 1, "KNOWS", 2, "work", None, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)

    assert result is not None
    assert {e.canonical_name for e in result.entities} == {"Alice", "Bob"}
    assert len(result.edges) == 1
    assert result.edges[0].id == 5001
    assert result.citations == []


def test_query_join_no_matching_edges_returns_none(router, graph):
    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)
    assert result is None


def test_query_join_hop_limit_below_path_length_raises(router, graph):
    with pytest.raises(ValueError, match="hop_limit"):
        graph.query_join(
            "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=1
        )


def test_query_join_unresolvable_start_returns_none(router, graph):
    result = graph.query_join("Ghost", [Predicate("KNOWS")], "work", hop_limit=1)
    assert result is None


def test_query_join_incoming_direction(router, graph):
    router.edges_by_object[(2, "KNOWS", "work")] = [
        _edge_row(5002, 1, "KNOWS", 2, "work", None, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join(
        "Bob", [Predicate("KNOWS")], "work", hop_limit=1, direction="incoming"
    )

    assert result is not None
    assert {e.canonical_name for e in result.entities} == {"Alice", "Bob"}


def test_query_join_collects_citations_from_fact_id(router, graph):
    from uuid import uuid4

    fact_id = uuid4()
    router.edges_by_subject[(1, "KNOWS", "work")] = [
        _edge_row(5003, 1, "KNOWS", 2, "work", fact_id, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)

    assert result is not None
    assert result.citations == [fact_id]


def test_query_join_two_hop_traversal_succeeds(router, graph):
    """Alice -KNOWS-> Bob -WORKS_WITH-> Carol: the BFS loop must
    accumulate both hops' edges/entities/citations across successive
    ``current_ids``, not just the first hop."""
    from uuid import uuid4

    fact_1, fact_2 = uuid4(), uuid4()
    router.edges_by_subject[(1, "KNOWS", "work")] = [
        _edge_row(5010, 1, "KNOWS", 2, "work", fact_1, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]
    router.edges_by_subject[(2, "WORKS_WITH", "work")] = [
        _edge_row(5011, 2, "WORKS_WITH", 3, "work", fact_2, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join(
        "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is not None
    assert {e.id for e in result.entities} == {1, 2, 3}
    assert len(result.edges) == 2
    assert {e.predicate for e in result.edges} == {"KNOWS", "WORKS_WITH"}
    assert set(result.citations) == {fact_1, fact_2}


def test_query_join_two_hop_dead_end_mid_path_returns_none(router, graph):
    """Only the first hop exists (Alice -KNOWS-> Bob); no WORKS_WITH edge
    out of Bob. The BFS must return None on the SECOND hop's dead end,
    not just the first."""
    router.edges_by_subject[(1, "KNOWS", "work")] = [
        _edge_row(5012, 1, "KNOWS", 2, "work", None, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join(
        "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is None


def test_query_join_entities_returned_in_bfs_insertion_order(router, graph):
    """Regression guard for TBU-150 -- entities must be in traversal order,
    NOT sorted by id. Carol(id=3) -KNOWS-> Alice(id=1) -WORKS_WITH-> Bob(id=2):
    traversal order [3, 1, 2] does NOT match id-sorted order [1, 2, 3], so a
    lurking ``sorted(entities_by_id)`` would produce [1, 2, 3] and fail this
    test."""
    router.edges_by_subject[(3, "KNOWS", "work")] = [
        _edge_row(5020, 3, "KNOWS", 1, "work", None, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]
    router.edges_by_subject[(1, "WORKS_WITH", "work")] = [
        _edge_row(5021, 1, "WORKS_WITH", 2, "work", None, 1.0, {}, "2026-07-01T00:00:00Z", None)
    ]

    result = graph.query_join(
        "Carol", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is not None
    assert [e.id for e in result.entities] == [3, 1, 2]


# ── aliases ──────────────────────────────────────────────────────────


def test_add_alias_issues_on_conflict_do_nothing_and_commits(router, fake_pool, graph):
    graph.add_alias(1, "Al", "work")

    alias_calls = [c for c in router.calls if c[0].startswith("INSERT INTO entity_aliases")]
    assert len(alias_calls) == 1
    assert alias_calls[0][1] == (1, "Al", "work")
    assert "ON CONFLICT (alias, profile) DO NOTHING" in alias_calls[0][0]
    assert fake_pool.connection().commits == 1


def test_resolve_alias_by_canonical_name(router, graph):
    entity = graph.resolve_alias("Alice", "work")
    assert entity is not None
    assert entity.id == 1
    assert entity.canonical_name == "Alice"


def test_resolve_alias_by_alias_fallback(router, graph):
    router.aliases[("Al", "work")] = 1

    entity = graph.resolve_alias("Al", "work")

    assert entity is not None
    assert entity.id == 1


def test_resolve_alias_not_found_returns_none(router, graph):
    assert graph.resolve_alias("Ghost", "work") is None


def test_resolve_alias_by_int_id_skips_name_lookup(router, graph):
    entity = graph.resolve_alias(2, "work")
    assert entity is not None
    assert entity.canonical_name == "Bob"
