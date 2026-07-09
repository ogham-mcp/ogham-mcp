"""Integration tests -- provenance chains (trace_provenance / find_derivatives)
against a real Postgres backend (TBU-127).

Requires a Postgres instance with migrations 041-046 applied (``entities``,
``entity_edges`` incl. ``derived_from``, ``entity_edge_predicates``,
``entity_aliases``). Run against the shared ``postgres-scratch`` Docker
container:

    DATABASE_URL="postgresql://ogham:ogham@localhost:5433/ogham_scratch" \
        .venv/bin/python -m pytest \
        tests/test_edge_provenance_integration.py -v \
        -m postgres_integration

Skipped automatically when ``DATABASE_URL`` does not contain "scratch" --
see ``_postgres_integration_db_safe`` in ``tests/conftest.py``. This module
does NOT use the ``pg_fresh_db`` fixture (destructive teardown); it opens
its own pool against the shared scratch DB, applies migration 046
idempotently in setup (a fresh scratch DB already has ``derived_from`` via
``schema_postgres.sql``; an older scratch DB predating this slice needs the
``ADD COLUMN IF NOT EXISTS`` applied), and uses uuid-prefixed entity names +
profiles so runs never collide, regardless of order. Mirrors
``tests/test_entity_graph_integration_store_triple.py``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from ogham.entity_graph import Predicate
from ogham.postgres.entity_graph import PostgresEntityGraph
from ogham.provenance import find_derivatives, trace_provenance

pytestmark = pytest.mark.postgres_integration

V1_VOCAB = {
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

_MIGRATION_046 = Path(__file__).parent.parent / "sql" / "migrations" / "046_edge_provenance.sql"


def _uid() -> str:
    """8-char uuid slug so entity names never collide across test runs."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
def graph(pg_url):
    """PostgresEntityGraph bound to the shared scratch database.

    Applies migration 046 idempotently first -- a fresh scratch DB already
    has ``derived_from`` (baked into ``schema_postgres.sql``), but a scratch
    DB left over from before this slice would not.
    """
    with psycopg.connect(pg_url, autocommit=True) as conn:
        # .encode() -- psycopg3's Query type requires a LiteralString for the
        # no-params execute() overload; migration text loaded from disk is a
        # plain str, not a literal, so bytes sidesteps that check (same
        # pattern as tests/test_migration_rls_non_superuser.py).
        conn.execute(_MIGRATION_046.read_text().encode())

    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        pg_url, min_size=1, max_size=2, open=True, kwargs={"row_factory": dict_row}
    )
    yield PostgresEntityGraph(pool, V1_VOCAB)
    pool.close()


def _seed_entities(pool, names_and_types):
    from psycopg.rows import tuple_row

    with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        ids = []
        for name, type_ in names_and_types:
            cur.execute(
                "INSERT INTO entities(canonical_name, entity_type) VALUES (%s, %s) RETURNING id",
                (name, type_),
            )
            ids.append(int(cur.fetchone()[0]))
        conn.commit()
        return ids


def test_scenario_1_no_provenance(graph):
    """Store an edge with no provenance -> derived_from=[]; trace returns
    just that edge (no fact_id set here, so no root_memories either)."""
    uid = _uid()
    a, b = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s")])
    profile = f"work-{uid}"

    edge_id = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)

    tree = trace_provenance(graph, edge_id, profile)
    assert [n.id for n in tree.nodes] == [edge_id]
    assert tree.nodes[0].derived_from == []
    assert tree.root_memories == []


def test_scenario_2_two_parents(graph):
    """An edge citing 2 parent edges -> trace_provenance returns both."""
    uid = _uid()
    a, b, c = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s"), (f"C-{uid}", "s")])
    profile = f"work-{uid}"

    p1 = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)
    p2 = graph.store_triple(b, Predicate("DEPENDS_ON"), c, None, profile)
    child = graph.store_triple(
        a,
        Predicate("OWNS"),
        c,
        None,
        profile,
        derived_from=[{"source_edge_id": p1}, {"source_edge_id": p2}],
    )

    tree = trace_provenance(graph, child, profile)
    assert {n.id for n in tree.nodes} == {child, p1, p2}
    assert {(link["from_edge_id"], link["to_edge_id"]) for link in tree.links} == {
        (child, p1),
        (child, p2),
    }


def test_scenario_3_three_hop_tree(graph):
    """A 3-hop chain (e1 <- e2 <- e3 <- e4) -- full tree at max_depth=3;
    max_depth=1 truncates to the start edge + its direct parent."""
    uid = _uid()
    a, b, c, d, e = _seed_entities(
        graph._pool,
        [
            (f"A-{uid}", "s"),
            (f"B-{uid}", "s"),
            (f"C-{uid}", "s"),
            (f"D-{uid}", "s"),
            (f"E-{uid}", "s"),
        ],
    )
    profile = f"work-{uid}"

    e4 = graph.store_triple(d, Predicate("DEPENDS_ON"), e, None, profile)
    e3 = graph.store_triple(
        c, Predicate("DEPENDS_ON"), d, None, profile, derived_from=[{"source_edge_id": e4}]
    )
    e2 = graph.store_triple(
        b, Predicate("DEPENDS_ON"), c, None, profile, derived_from=[{"source_edge_id": e3}]
    )
    e1 = graph.store_triple(
        a, Predicate("DEPENDS_ON"), b, None, profile, derived_from=[{"source_edge_id": e2}]
    )

    full_tree = trace_provenance(graph, e1, profile, max_depth=3)
    assert {n.id for n in full_tree.nodes} == {e1, e2, e3, e4}

    truncated_tree = trace_provenance(graph, e1, profile, max_depth=1)
    assert {n.id for n in truncated_tree.nodes} == {e1, e2}


def test_scenario_4_find_derivatives_transitive(graph):
    """find_derivatives on a root edge returns all direct + transitive
    derivatives (impact analysis: "what depends on this fact?")."""
    uid = _uid()
    a, b, c = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s"), (f"C-{uid}", "s")])
    profile = f"work-{uid}"

    root = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)
    direct = graph.store_triple(
        b, Predicate("OWNS"), c, None, profile, derived_from=[{"source_edge_id": root}]
    )
    transitive = graph.store_triple(
        c, Predicate("PART_OF"), a, None, profile, derived_from=[{"source_edge_id": direct}]
    )

    derivatives = find_derivatives(graph, root, profile)
    assert {e.id for e in derivatives} == {direct, transitive}


def test_scenario_5_provenance_survives_supersession(graph):
    """Provenance is historical: superseding a parent edge must not break
    trace_provenance on a child that cites it (walk reads regardless of
    valid_to)."""
    uid = _uid()
    a, b, c = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s"), (f"C-{uid}", "s")])
    profile = f"work-{uid}"

    parent = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)
    child = graph.store_triple(
        a, Predicate("OWNS"), c, None, profile, derived_from=[{"source_edge_id": parent}]
    )
    # Re-storing the same (a, DEPENDS_ON, b, profile) triple supersedes `parent`.
    graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)

    tree = trace_provenance(graph, child, profile)
    assert parent in {n.id for n in tree.nodes}
    parent_node = next(n for n in tree.nodes if n.id == parent)
    assert parent_node.valid_to is not None  # confirms it's the superseded row
