"""list_entities / list_edges / list_aliases -- the read primitives OKF export needs.

The EntityGraph protocol could walk the graph (query_join) and fetch one edge
(fetch_edge), but never ENUMERATE. OKF export needs the whole profile, so
TBU-130 adds three list-by-profile methods to the protocol and both backends.

The subtle part is scoping. `entities` has no profile column -- it is global,
scoped only through `memory_entities` and `entity_edges`. So `list_entities`
is the union of those two, not a table scan, and the union is what guarantees
every edge endpoint is an exported concept (which is why dangling links are
structurally impossible on export).
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from ogham.entity_graph import Entity, EntityGraph
from ogham.postgres.entity_graph import PostgresEntityGraph
from ogham.supabase.entity_graph import SupabaseEntityGraph

# ── protocol ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["list_entities", "list_edges", "list_aliases"])
def test_protocol_declares_the_list_primitives(name):
    assert hasattr(EntityGraph, name), f"EntityGraph protocol missing {name}"


@pytest.mark.parametrize("backend", [PostgresEntityGraph, SupabaseEntityGraph])
@pytest.mark.parametrize("name", ["list_entities", "list_edges", "list_aliases"])
def test_both_backends_implement_them(backend, name):
    assert callable(getattr(backend, name, None)), f"{backend.__name__} missing {name}"


def test_store_triple_docstring_names_the_object_in_the_supersession_key():
    """Both backends key supersession on (subject, predicate, OBJECT, profile),
    matching entity_edges_current_uq. The docstring used to claim a wildcard on
    the object -- which would mean one current object per predicate, and would
    make list-valued predicates (OWNS: [a, b]) impossible to round-trip. The
    code was right; the docstring was wrong.
    """
    doc = EntityGraph.store_triple.__doc__ or ""
    assert "(subject, predicate, *, profile)" not in doc
    assert "object" in doc.lower()


# ── postgres backend ──────────────────────────────────────────────────────


class _FakeCursor:
    """Records SQL and replays canned rows in order."""

    def __init__(self, results: list[list[dict[str, Any]]], log: list[tuple[str, Any]]):
        self._results = results
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._results.pop(0) if self._results else []


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        pass


class _FakePool:
    def __init__(self, results):
        self.log: list[tuple[str, Any]] = []
        self._cursor = _FakeCursor(results, self.log)

    def connection(self):
        return _FakeConn(self._cursor)


def _pg(results) -> tuple[PostgresEntityGraph, _FakePool]:
    """Returns (graph, pool) so assertions read the fake directly -- reaching
    through `graph._pool` would resolve to the declared ConnectionPool type."""
    pool = _FakePool(results)
    return PostgresEntityGraph(cast(Any, pool), allowed_predicates=["OWNS"]), pool


def test_pg_list_entities_scopes_through_the_join_tables_not_a_table_scan():
    graph, pool = _pg([[{"id": 42, "canonical_name": "Ogham", "entity_type": "project"}]])
    out = graph.list_entities("work")

    assert out == [Entity(id=42, canonical_name="Ogham", entity_type="project")]
    sql, params = pool.log[0]
    assert "memory_entities" in sql
    assert "subject_id" in sql and "object_id" in sql
    assert params == {"p": "work"}


def test_pg_list_entities_orders_by_id_for_deterministic_export():
    graph, pool = _pg([[]])
    graph.list_entities("work")
    sql, _ = pool.log[0]
    assert "ORDER BY" in sql.upper()


def test_pg_list_edges_filters_to_current_by_default():
    graph, pool = _pg([[]])
    graph.list_edges("work")
    sql, _ = pool.log[0]
    assert "valid_to IS NULL" in sql


def test_pg_list_edges_and_aliases_are_ordered_too():
    """Determinism is not just a list_entities concern.

    Mutation-checked: dropping ORDER BY from list_edges and list_aliases on BOTH
    backends left the whole suite green at 1630 passed. Unordered edges make the
    exported bundle non-deterministic, which breaks the byte-identical-export
    guarantee (D9) -- and the fixture-based export tests cannot catch it,
    because a fixture list is already in a fixed order.
    """
    graph, pool = _pg([[]])
    graph.list_edges("work")
    assert "ORDER BY id" in pool.log[0][0]

    graph, pool = _pg([[]])
    graph.list_aliases("work")
    assert "ORDER BY entity_id, alias" in pool.log[0][0]


def test_pg_list_edges_can_include_history():
    graph, pool = _pg([[]])
    graph.list_edges("work", current_only=False)
    sql, _ = pool.log[0]
    assert "valid_to IS NULL" not in sql


def test_pg_list_aliases_groups_by_entity():
    graph, _pool = _pg(
        [
            [
                {"entity_id": 42, "alias": "OpenBrain"},
                {"entity_id": 42, "alias": "ogham-mcp"},
                {"entity_id": 7, "alias": "sb"},
            ]
        ]
    )
    assert graph.list_aliases("work") == {42: ["OpenBrain", "ogham-mcp"], 7: ["sb"]}


# ── supabase backend ──────────────────────────────────────────────────────


class _FakeQuery:
    def __init__(self, rows, log, table):
        self._rows = rows
        self._log = log
        self._table = table

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def order(self, col, **k):
        self._log.append(("order", self._table, col))
        return self

    def is_(self, col, val):
        self._log.append(("is_", self._table, col, val))
        return self

    def in_(self, col, values):
        self._log.append(("in_", self._table, col, list(values)))
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeClient:
    def __init__(self, tables: dict[str, list[dict[str, Any]]]):
        self.tables = tables
        self.log: list[tuple] = []

    def table(self, name):
        self.log.append(("table", name))
        return _FakeQuery(self.tables.get(name, []), self.log, name)


def _sb(tables) -> tuple[SupabaseEntityGraph, _FakeClient]:
    client = _FakeClient(tables)
    return SupabaseEntityGraph(cast(Any, client), allowed_predicates=["OWNS"]), client


def test_sb_list_entities_unions_edge_endpoints_with_memory_links():
    """PostgREST has no IN-subquery, so the union happens in Python. Every
    endpoint of every edge must come back, or export produces dangling links."""
    graph, client = _sb(
        {
            "memory_entities": [{"entity_id": 7}],
            "entity_edges": [{"subject_id": 42, "object_id": 88}],
            "entities": [
                {"id": 7, "canonical_name": "Supabase", "entity_type": "service"},
                {"id": 42, "canonical_name": "Ogham", "entity_type": "project"},
                {"id": 88, "canonical_name": "Graph", "entity_type": "component"},
            ],
        }
    )
    out = graph.list_entities("work")

    assert {e.id for e in out} == {7, 42, 88}
    in_calls = [c for c in client.log if c[0] == "in_"]
    assert in_calls, "expected a single .in_ lookup against entities"
    assert sorted(in_calls[0][3]) == [7, 42, 88]


def test_sb_list_entities_returns_empty_without_calling_in_with_an_empty_list():
    """`.in_("id", [])` is a PostgREST error, not an empty result. A profile
    with no entities must short-circuit before the query is built."""
    graph, client = _sb({"memory_entities": [], "entity_edges": [], "entities": []})

    assert graph.list_entities("empty") == []
    assert not [c for c in client.log if c[0] == "in_"]


def test_sb_list_entities_asks_the_server_to_order_by_id():
    """A stub cannot prove PostgREST sorts -- only that we asked it to. That the
    rows actually come back ordered is asserted against a real database in the
    postgres_integration export test."""
    graph, client = _sb(
        {
            "memory_entities": [{"entity_id": 88}, {"entity_id": 7}],
            "entity_edges": [],
            "entities": [
                {"id": 7, "canonical_name": "Supabase", "entity_type": "service"},
                {"id": 88, "canonical_name": "Graph", "entity_type": "component"},
            ],
        }
    )
    graph.list_entities("work")
    assert ("order", "entities", "id") in client.log


def test_sb_list_edges_filters_to_current_by_default():
    graph, client = _sb({"entity_edges": []})
    graph.list_edges("work")
    assert ("is_", "entity_edges", "valid_to", "null") in client.log


def test_sb_list_edges_and_aliases_ask_for_an_order():
    """Mirror of the postgres ordering test -- see its docstring for why.

    Supabase also orders aliases by entity_id ALONE, where postgres orders by
    (entity_id, alias). Within an entity the two backends can therefore disagree
    on alias order. Harmless today because aliases land in a YAML list nothing
    compares across backends, but it is a real cross-backend divergence in a
    Protocol whose whole point is behavioural parity -- noted rather than
    silently accepted.
    """
    graph, client = _sb({"entity_edges": []})
    graph.list_edges("work")
    assert ("order", "entity_edges", "id") in client.log

    graph, client = _sb({"entity_aliases": []})
    graph.list_aliases("work")
    assert ("order", "entity_aliases", "entity_id") in client.log


def test_sb_list_edges_can_include_history():
    graph, client = _sb({"entity_edges": []})
    graph.list_edges("work", current_only=False)
    assert ("is_", "entity_edges", "valid_to", "null") not in client.log


def test_sb_list_aliases_groups_by_entity():
    graph, _client = _sb(
        {
            "entity_aliases": [
                {"entity_id": 42, "alias": "OpenBrain"},
                {"entity_id": 42, "alias": "ogham-mcp"},
            ]
        }
    )
    assert graph.list_aliases("work") == {42: ["OpenBrain", "ogham-mcp"]}
