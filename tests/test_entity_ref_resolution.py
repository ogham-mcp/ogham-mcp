"""Entity references resolve on the natural key, not on the name alone.

``entities`` is UNIQUE (canonical_name, entity_type), so a bare name is NOT a
key. Ordinary extraction produces the same name under two types routinely --
every ``*Error`` matches both the CamelCase rule and the Error-suffix rule, so
``ValueError`` lands as ``entity:ValueError`` AND ``error:ValueError``. There
were 18 such names on the live store on 2026-08-20.

Before this fix ``_resolve_to_id`` ran ``WHERE canonical_name = %s LIMIT 1``
with no ORDER BY and no entity_type, so ``store_triple`` and ``walk_knowledge``
picked between them arbitrarily -- an edge meant for the error node could land
on the entity node, and nothing reported it.
"""

from __future__ import annotations

import pytest
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from ogham.entity_graph import KNOWN_ENTITY_TYPES, split_entity_ref
from ogham.postgres.entity_graph import PostgresEntityGraph


class TestSplitEntityRef:
    def test_splits_a_known_type_prefix(self):
        assert split_entity_ref("error:KeyError") == ("error", "KeyError")
        assert split_entity_ref("file:src/a.py") == ("file", "src/a.py")

    def test_bare_name_is_left_whole(self):
        assert split_entity_ref("KeyError") == (None, "KeyError")

    def test_unknown_prefix_is_not_a_type(self):
        # "weird" is not an entity type, so this is a name that has a colon in
        # it, not a qualified reference.
        assert split_entity_ref("weird:thing") == (None, "weird:thing")

    @pytest.mark.parametrize(
        "ref",
        [
            "https://example.com/a",
            "postgresql://ogham@10.10.14.200:5432/ogham",
            "C:/Users/app.toml",
        ],
    )
    def test_names_that_merely_contain_a_colon_survive_intact(self, ref):
        """The reason this is not ``split(':', 1)``.

        A URL, a DSN and a Windows path all contain colons and none of them is
        a qualified reference. Splitting naively would mangle them into the
        wrong entity silently.
        """
        assert split_entity_ref(ref) == (None, ref)

    def test_every_extraction_type_is_recognised(self):
        # If extract_entities gains a type and this set is not updated, that
        # type's qualified refs would silently fall back to name-only lookup.
        for t in ("entity", "file", "error", "quantity", "location"):
            assert t in KNOWN_ENTITY_TYPES


V1_VOCAB = ["OWNS", "MENTIONS"]


@pytest.fixture
def graph_and_pool(pg_url):
    """Same construction as tests/test_entity_graph_integration_query_join.py.

    dict_row deliberately, so this exercises the row shape production uses
    (TBU-164 -- tuple rows here once masked a by-name-access bug).
    """
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        pg_url, min_size=1, max_size=2, open=True, kwargs={"row_factory": dict_row}
    )
    yield PostgresEntityGraph(pool, V1_VOCAB), pool
    pool.close()


def _seed_collision(pool) -> tuple[int, int]:
    """Create the exact collision ordinary extraction produces for ``*Error``."""
    ids = {}
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM entities WHERE canonical_name = 'ValueError'")
        for etype in ("entity", "error"):
            cur.execute(
                "INSERT INTO entities (canonical_name, entity_type) "
                "VALUES ('ValueError', %s) RETURNING id",
                (etype,),
            )
            row = cur.fetchone()
            assert row is not None
            ids[etype] = int(row["id"])
        conn.commit()
    return ids["entity"], ids["error"]


@pytest.mark.postgres_integration
class TestResolveToIdIsUnambiguous:
    def test_qualified_reference_resolves_to_the_right_type(self, graph_and_pool):
        graph, pool = graph_and_pool
        entity_id, error_id = _seed_collision(pool)
        assert graph._resolve_to_id("entity:ValueError", "t") == entity_id
        assert graph._resolve_to_id("error:ValueError", "t") == error_id
        assert entity_id != error_id

    def test_unqualified_reference_is_deterministic(self, graph_and_pool):
        """Ambiguity is unavoidable here; arbitrariness is not."""
        graph, pool = graph_and_pool
        entity_id, error_id = _seed_collision(pool)
        first = graph._resolve_to_id("ValueError", "t")
        assert first == graph._resolve_to_id("ValueError", "t")
        assert first == min(entity_id, error_id), "lowest id wins, by ORDER BY id"

    def test_unqualified_ambiguity_is_reported(self, graph_and_pool, caplog):
        graph, pool = graph_and_pool
        _seed_collision(pool)
        with caplog.at_level("WARNING"):
            graph._resolve_to_id("ValueError", "t")
        assert any("ambiguous" in r.getMessage().lower() for r in caplog.records), (
            "an ambiguous resolution must not be silent"
        )
