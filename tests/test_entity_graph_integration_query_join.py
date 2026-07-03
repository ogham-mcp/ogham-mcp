"""Integration tests -- query_join against a real Postgres backend.

Requires a Postgres instance with migrations 041-043 applied (``entities``,
``entity_edges``, ``entity_edge_predicates``, ``entity_aliases``). Run
against the shared ``postgres-scratch`` Docker container:

    DATABASE_URL="postgresql://ogham:ogham@localhost:5433/ogham_scratch" \
        .venv/bin/python -m pytest \
        tests/test_entity_graph_integration_query_join.py -v \
        -m postgres_integration

Skipped automatically when ``DATABASE_URL`` does not contain "scratch" --
see ``_postgres_integration_db_safe`` in ``tests/conftest.py``, which the
module-level ``postgres_integration`` marker wires in via the autouse
``_isolated_unit_environment`` fixture. Like
``test_entity_graph_integration_store_triple.py``, this module does NOT use
the ``pg_fresh_db`` fixture (a migration harness that DROPs
entities/memory_entities on setup and teardown -- see its docstring), and
reuses the shared ``pg_url`` fixture from ``tests/conftest.py``. Each test
opens its own pool against the same shared scratch DB and never
drops/truncates; tests use uuid-prefixed entity names + profiles so runs
never collide, regardless of order.
"""

from __future__ import annotations

import uuid

import pytest
from psycopg import Connection
from psycopg.rows import DictRow, dict_row, tuple_row
from psycopg_pool import ConnectionPool

from ogham.entity_graph import Predicate
from ogham.postgres.entity_graph import PostgresEntityGraph

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


def _uid() -> str:
    """8-char uuid slug so entity names never collide across test runs.

    ``entities`` has UNIQUE(canonical_name, entity_type); the shared scratch
    DB already carries rows from earlier validation, and this module's own
    tests must be independently re-runnable in any order.
    """
    return uuid.uuid4().hex[:8]


@pytest.fixture
def graph(pg_url):
    """Provides a PostgresEntityGraph bound to the shared scratch database.

    Mirrors production (``src/ogham/backends/postgres.py``): the pool is
    built with ``row_factory=dict_row`` so this test suite exercises
    ``PostgresEntityGraph`` against the same row shape it sees in prod
    (TBU-164 -- tuple rows here masked the by-name-access bug).
    """
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        pg_url, min_size=1, max_size=2, open=True, kwargs={"row_factory": dict_row}
    )
    yield PostgresEntityGraph(pool, V1_VOCAB)
    pool.close()


def _seed_entities(pool, names_and_types):
    # This is the test's own bookkeeping SQL, not PostgresEntityGraph -- use
    # an explicit tuple_row cursor so positional access below keeps working
    # even though the pool itself is dict_row (see graph fixture docstring).
    with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        ids = {}
        for name, type_ in names_and_types:
            cur.execute(
                "INSERT INTO entities(canonical_name, entity_type) VALUES (%s, %s) RETURNING id",
                (name, type_),
            )
            ids[name] = int(cur.fetchone()[0])
        conn.commit()
        return ids


def test_single_hop_path(graph):
    uid = _uid()
    a_name, b_name = f"A-{uid}", f"B-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("DEPENDS_ON"), ids[b_name], None, profile)

    result = graph.query_join(
        start_entity=ids[a_name],
        predicate_path=[Predicate("DEPENDS_ON")],
        profile=profile,
        hop_limit=1,
    )
    assert result is not None
    assert len(result.edges) == 1
    assert {e.canonical_name for e in result.entities} == {a_name, b_name}


def test_multi_hop_path(graph):
    uid = _uid()
    a_name, b_name, c_name = f"A-{uid}", f"B-{uid}", f"C-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s"), (c_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("DEPENDS_ON"), ids[b_name], None, profile)
    graph.store_triple(ids[b_name], Predicate("OWNS"), ids[c_name], None, profile)

    result = graph.query_join(
        start_entity=ids[a_name],
        predicate_path=[Predicate("DEPENDS_ON"), Predicate("OWNS")],
        profile=profile,
        hop_limit=2,
    )
    assert result is not None
    assert len(result.edges) == 2
    assert {e.canonical_name for e in result.entities} == {a_name, b_name, c_name}


def test_multi_hop_entities_bfs_order(graph):
    """entities is a path signal (TBU-150): insertion order, not id-sorted.

    JoinResult.entities docstring (src/ogham/entity_graph.py) says the list
    is in BFS insertion order -- start entity first, then each hop's
    discovered entities in the order the traversal encountered them. A
    set-based membership check (as in test_multi_hop_path above) can't
    catch a regression that silently reorders/sorts the list, so this test
    asserts the exact sequence for a linear A -> B -> C path.
    """
    uid = _uid()
    a_name, b_name, c_name = f"A-{uid}", f"B-{uid}", f"C-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s"), (c_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("DEPENDS_ON"), ids[b_name], None, profile)
    graph.store_triple(ids[b_name], Predicate("OWNS"), ids[c_name], None, profile)

    result = graph.query_join(
        start_entity=ids[a_name],
        predicate_path=[Predicate("DEPENDS_ON"), Predicate("OWNS")],
        profile=profile,
        hop_limit=2,
    )
    assert result is not None
    assert [e.canonical_name for e in result.entities] == [a_name, b_name, c_name]


def test_no_path_returns_none(graph):
    uid = _uid()
    a_name, b_name = f"A-{uid}", f"B-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("DEPENDS_ON"), ids[b_name], None, profile)

    result = graph.query_join(
        start_entity=ids[a_name],
        predicate_path=[Predicate("OWNS")],
        profile=profile,
        hop_limit=1,
    )
    assert result is None


def test_cycle_detection_terminates(graph):
    uid = _uid()
    a_name, b_name = f"A-{uid}", f"B-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("RELATED_TO"), ids[b_name], None, profile)
    graph.store_triple(ids[b_name], Predicate("RELATED_TO"), ids[a_name], None, profile)

    # Walk A -[RELATED_TO]-> B -[RELATED_TO]-> ??
    # A is already visited by hop 2, so the traversal has nowhere new to go
    # and terminates without looping back to A.
    result = graph.query_join(
        start_entity=ids[a_name],
        predicate_path=[Predicate("RELATED_TO"), Predicate("RELATED_TO")],
        profile=profile,
        hop_limit=2,
    )
    assert result is None


def test_canonical_name_start_entity(graph):
    """query_join with start_entity given as a canonical name (not an int id,
    not an alias) that resolves to a real entity via the FIRST branch of
    ``_resolve_to_id`` (``SELECT id FROM entities WHERE canonical_name``).

    This is the exact shape of the live TBU-164 repro
    (``query_join("OpenBrain")``): a real, resolvable entity name. None of
    the other scenarios in this module exercise this branch -- they either
    pass int ids directly (skip resolution entirely) or resolve via the
    alias fallback branch (test_alias_start_entity below) -- so without
    this test the canonical-name KeyError regression class could ship
    again undetected.
    """
    uid = _uid()
    a_name, b_name = f"A-{uid}", f"B-{uid}"
    ids = _seed_entities(graph._pool, [(a_name, "s"), (b_name, "s")])
    profile = f"work-{uid}"
    graph.store_triple(ids[a_name], Predicate("DEPENDS_ON"), ids[b_name], None, profile)

    result = graph.query_join(
        start_entity=a_name,
        predicate_path=[Predicate("DEPENDS_ON")],
        profile=profile,
        hop_limit=1,
    )
    assert result is not None
    assert {e.canonical_name for e in result.entities} == {a_name, b_name}


def test_alias_start_entity(graph):
    uid = _uid()
    auth_name, login_name = f"AuthService-{uid}", f"LoginModule-{uid}"
    ids = _seed_entities(graph._pool, [(auth_name, "s"), (login_name, "s")])
    profile = f"work-{uid}"
    alias = f"auth-{uid}"
    graph.add_alias(ids[auth_name], alias, profile)
    graph.store_triple(ids[auth_name], Predicate("DEPENDS_ON"), ids[login_name], None, profile)

    result = graph.query_join(
        start_entity=alias,
        predicate_path=[Predicate("DEPENDS_ON")],
        profile=profile,
        hop_limit=1,
    )
    assert result is not None
    assert {e.canonical_name for e in result.entities} == {auth_name, login_name}


def test_unresolvable_start_entity_returns_none(graph):
    """query_join with a start_entity that resolves to no entity/alias returns None.

    Not one of the plan's original 5 scenarios -- added because the mock-
    level unit tests can't exercise real resolver-miss behaviour there.
    Confirmed against the actual source (src/ogham/postgres/entity_graph.py
    query_join): ``_resolve_to_id`` returning None short-circuits to
    ``return None`` immediately, the same "no path resolves" contract as
    test_no_path_returns_none above -- it does NOT raise. This differs from
    store_triple, which raises ValueError on an unresolvable subject/object
    (see test_scenario_6_unresolvable_subject_raises_value_error in
    test_entity_graph_integration_store_triple.py): a read-only traversal
    with no match is a legitimate "no result", not a caller error.
    """
    uid = _uid()
    result = graph.query_join(
        start_entity=f"ghost-{uid}",
        predicate_path=[Predicate("DEPENDS_ON")],
        profile=f"work-{uid}",
        hop_limit=1,
    )
    assert result is None
