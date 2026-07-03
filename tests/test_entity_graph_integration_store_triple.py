"""Integration tests -- store_triple against a real Postgres backend.

Requires a Postgres instance with migrations 041-043 applied (``entities``,
``entity_edges``, ``entity_edge_predicates``, ``entity_aliases``). Run
against the shared ``postgres-scratch`` Docker container:

    DATABASE_URL="postgresql://ogham:ogham@localhost:5433/ogham_scratch" \
        .venv/bin/python -m pytest \
        tests/test_entity_graph_integration_store_triple.py -v \
        -m postgres_integration

Skipped automatically when ``DATABASE_URL`` does not contain "scratch" --
see ``_postgres_integration_db_safe`` in ``tests/conftest.py``, which the
module-level ``postgres_integration`` marker wires in via the autouse
``_isolated_unit_environment`` fixture. This module does NOT use the
``pg_fresh_db`` fixture from conftest.py: that fixture is a migration
harness that DROPs the ``entities``/``memory_entities`` tables on setup and
teardown (see its docstring), which would destroy the 041-043 schema these
tests depend on. Instead each test opens its own pool against the same
shared scratch DB and never drops/truncates -- tests use uuid-prefixed
entity names + profiles so runs never collide, regardless of order.
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
        ids = []
        for name, type_ in names_and_types:
            cur.execute(
                "INSERT INTO entities(canonical_name, entity_type) VALUES (%s, %s) RETURNING id",
                (name, type_),
            )
            ids.append(int(cur.fetchone()[0]))
        conn.commit()
        return ids


def test_scenario_1_store_new_triple(graph):
    """Store a new triple; asserted present with valid_to=NULL."""
    uid = _uid()
    a, b = _seed_entities(
        graph._pool,
        [(f"AuthService-{uid}", "service"), (f"LoginModule-{uid}", "module")],
    )
    edge_id = graph.store_triple(
        subject=a,
        predicate=Predicate("DEPENDS_ON"),
        object_=b,
        source_memory_id=None,
        profile=f"work-{uid}",
    )
    with graph._pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT COUNT(*) FROM entity_edges WHERE id=%s AND valid_to IS NULL",
            (edge_id,),
        )
        assert cur.fetchone()[0] == 1


def test_scenario_2_duplicate_triple_supersedes_prior(graph):
    """Storing the same triple twice is write-time supersession, not a no-op.

    Named "duplicate_noop" in the original plan draft, which is misleading:
    the old row is NOT silently ignored -- its valid_to is stamped and a new
    current row is inserted (see PostgresEntityGraph.store_triple). The
    assertion below (exactly one current row) is still the correct
    observable outcome: only one row is ever current for a given
    (subject, predicate, object, profile).
    """
    uid = _uid()
    a, b = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s")])
    profile = f"work-{uid}"
    e1 = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)
    e2 = graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile)
    assert e1 != e2
    with graph._pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT COUNT(*) FROM entity_edges "
            "WHERE subject_id=%s AND object_id=%s AND profile=%s AND valid_to IS NULL",
            (a, b, profile),
        )
        assert cur.fetchone()[0] == 1
        # The first row should now be superseded and point at the second.
        cur.execute("SELECT valid_to IS NULL, superseded_by FROM entity_edges WHERE id=%s", (e1,))
        valid_to_is_null, superseded_by = cur.fetchone()
        assert valid_to_is_null is False
        assert superseded_by == e2


def test_scenario_3_different_objects_do_not_supersede(graph):
    """(A, OWNS, B) and (A, OWNS, C) are BOTH current.

    Named "supersession_chain" in the original plan draft, which implied
    e1 gets superseded when e2 is stored. That's not the design:
    supersession keys on (subject, predicate, object, profile) -- not just
    (subject, predicate) -- so different objects never collide. Both edges
    stay current simultaneously.
    """
    uid = _uid()
    a, b, c = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s"), (f"C-{uid}", "s")])
    profile = f"work-{uid}"
    e1 = graph.store_triple(a, Predicate("OWNS"), b, None, profile)
    graph.store_triple(a, Predicate("OWNS"), c, None, profile)
    with graph._pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT valid_to IS NULL, superseded_by FROM entity_edges WHERE id=%s",
            (e1,),
        )
        row = cur.fetchone()
        assert row[0] is True  # e1 still current
        assert row[1] is None  # never superseded
        cur.execute(
            "SELECT COUNT(*) FROM entity_edges "
            "WHERE subject_id=%s AND predicate='OWNS' AND profile=%s AND valid_to IS NULL",
            (a, profile),
        )
        assert cur.fetchone()[0] == 2


def test_scenario_4_alias_resolution(graph):
    """Subject given as alias resolves to canonical entity_id."""
    uid = _uid()
    a, b = _seed_entities(graph._pool, [(f"AuthService-{uid}", "s"), (f"LoginModule-{uid}", "s")])
    profile = f"work-{uid}"
    alias = f"auth-{uid}"
    graph.add_alias(a, alias, profile)
    edge_id = graph.store_triple(alias, Predicate("DEPENDS_ON"), b, None, profile)
    with graph._pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SELECT subject_id FROM entity_edges WHERE id=%s", (edge_id,))
        assert cur.fetchone()[0] == a


def test_scenario_5_profile_isolation(graph):
    """Same (subject, predicate, object) in two profiles -- one current row each."""
    uid = _uid()
    a, b = _seed_entities(graph._pool, [(f"A-{uid}", "s"), (f"B-{uid}", "s")])
    profile_work = f"work-{uid}"
    profile_personal = f"personal-{uid}"
    graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile_work)
    graph.store_triple(a, Predicate("DEPENDS_ON"), b, None, profile_personal)
    with graph._pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT profile FROM entity_edges "
            "WHERE subject_id=%s AND object_id=%s AND valid_to IS NULL ORDER BY profile",
            (a, b),
        )
        profiles = [r[0] for r in cur.fetchall()]
        assert profiles == sorted([profile_personal, profile_work])


def test_scenario_6_unresolvable_subject_raises_value_error(graph):
    """store_triple with a subject that resolves to no entity/alias raises cleanly.

    Not one of the plan's original 5 scenarios -- added because the mock-
    level unit tests (tests/test_postgres_entity_graph.py) can't exercise
    real resolver-miss behaviour, the resolver is mocked there. Confirmed
    against the actual source (src/ogham/postgres/entity_graph.py
    store_triple): a None from _resolve_to_id raises ValueError, not
    LookupError -- there is no INSERT attempt with a NULL subject_id (the
    column is NOT NULL, so that path would otherwise surface as an opaque
    IntegrityError instead of a clean domain error).
    """
    uid = _uid()
    (b,) = _seed_entities(graph._pool, [(f"B-{uid}", "s")])
    with pytest.raises(ValueError, match="cannot resolve"):
        graph.store_triple(f"ghost-{uid}", Predicate("DEPENDS_ON"), b, None, f"work-{uid}")
