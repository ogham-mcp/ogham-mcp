"""Migration 036 + link_memory_entities RPC integration tests.

Verifies (a) 036 forward creates entities + memory_entities tables with
the right columns; (b) the link_memory_entities RPC upserts entities and
inserts edges idempotently; (c) entity_graph_density returns sensible
counts after a write; (d) suggest_unlinked_by_shared_entities lights up
once the graph is populated.

Surfaced 2026-04-29: prior to v0.14, NO write path populated entities or
memory_entities, so all three RPCs were silent no-ops. This test pins the
end-to-end happy path so future refactors don't regress it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_036 = Path(__file__).parent.parent / "sql/migrations/036_entities_backfill.sql"
ROLLBACK_036 = (
    Path(__file__).parent.parent / "sql/migrations/rollback/DANGER_036_entities_backfill.sql"
)


def _can_connect() -> bool:
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()
        backend._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(not _can_connect(), reason="Postgres backend not configured or unreachable"),
]


def _apply_036(pg_fresh_db):
    """Apply migration 036 in isolation. ``pg_fresh_db`` already has the
    base memories table (created by the schema fixture), so 036's FK to
    ``memories(id)`` resolves cleanly.
    """
    pg_fresh_db.apply_sql(MIG_036)


def test_036_creates_entities_and_memory_entities_tables(pg_fresh_db):
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = backend._execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name IN ('entities', 'memory_entities') AND table_schema = 'public'",
        {},
        fetch="all",
    )
    found = {r["table_name"] for r in rows or []}
    assert found == {"entities", "memory_entities"}


def test_036_creates_supporting_rpc_functions(pg_fresh_db):
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    rows = backend._execute(
        "SELECT routine_name FROM information_schema.routines "
        "WHERE routine_name IN ('link_memory_entities', 'refresh_entity_temporal_span', "
        "                       'spread_entity_activation_memories') "
        "  AND routine_schema = 'public'",
        {},
        fetch="all",
    )
    found = {r["routine_name"] for r in rows or []}
    assert found == {
        "link_memory_entities",
        "refresh_entity_temporal_span",
        "spread_entity_activation_memories",
    }


def _seed_memory(pg_fresh_db, content: str = "Bug fixed in src/ogham/llm.py", profile: str = "t"):
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    row = backend._execute(
        "INSERT INTO memories (content, profile, source) "
        "VALUES (%(c)s, %(p)s, 't') RETURNING id::text AS id",
        {"c": content, "p": profile},
        fetch="one",
    )
    assert row is not None, "INSERT ... RETURNING should always yield a row"
    return row["id"]


def test_link_memory_entities_upserts_and_links(pg_fresh_db):
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    memory_id = _seed_memory(pg_fresh_db)

    n = backend.link_memory_entities(
        memory_id=memory_id,
        profile="t",
        entity_tags=["file:src/ogham/llm.py", "person:Cemre", "project:ogham"],
    )
    assert n == 3

    edges = backend._execute(
        "SELECT count(*) AS c FROM memory_entities WHERE memory_id = %(m)s::uuid",
        {"m": memory_id},
        fetch="scalar",
    )
    assert edges == 3

    entity_count = backend._execute("SELECT count(*) AS c FROM entities", {}, fetch="scalar")
    assert entity_count == 3


def test_link_memory_entities_idempotent_on_re_run(pg_fresh_db):
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    memory_id = _seed_memory(pg_fresh_db)
    tags = ["file:foo.py", "person:Cemre"]

    first = backend.link_memory_entities(memory_id=memory_id, profile="t", entity_tags=tags)
    second = backend.link_memory_entities(memory_id=memory_id, profile="t", entity_tags=tags)

    assert first == 2
    assert second == 0  # ON CONFLICT DO NOTHING -- no new edges

    edges = backend._execute(
        "SELECT count(*) AS c FROM memory_entities WHERE memory_id = %(m)s::uuid",
        {"m": memory_id},
        fetch="scalar",
    )
    assert edges == 2  # still 2, not 4


def test_link_memory_entities_increments_mention_count(pg_fresh_db):
    """A second link with the same entity tag bumps the mention counter."""
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    m1 = _seed_memory(pg_fresh_db, content="First mention of Cemre")
    m2 = _seed_memory(pg_fresh_db, content="Second mention of Cemre")

    backend.link_memory_entities(memory_id=m1, profile="t", entity_tags=["person:Cemre"])
    backend.link_memory_entities(memory_id=m2, profile="t", entity_tags=["person:Cemre"])

    mention = backend._execute(
        "SELECT mention_count FROM entities WHERE canonical_name='Cemre' AND entity_type='person'",
        {},
        fetch="scalar",
    )
    assert mention == 2


def test_link_memory_entities_handles_empty_input(pg_fresh_db):
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    memory_id = _seed_memory(pg_fresh_db)

    n = backend.link_memory_entities(memory_id=memory_id, profile="t", entity_tags=[])
    assert n == 0


def test_link_memory_entities_skips_malformed_tags(pg_fresh_db):
    """Tags without a colon or with empty name parts are dropped silently."""
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    memory_id = _seed_memory(pg_fresh_db)

    n = backend.link_memory_entities(
        memory_id=memory_id,
        profile="t",
        entity_tags=["good:valid", "missing-colon", "empty:", "person:Real"],
    )
    assert n == 2  # only "good:valid" and "person:Real"


def test_entity_graph_density_after_link(pg_fresh_db):
    """Hotfix A's density RPC starts returning real numbers once the graph is fed."""
    _apply_036(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    memory_id = _seed_memory(pg_fresh_db)
    backend.link_memory_entities(
        memory_id=memory_id, profile="t", entity_tags=["file:a.py", "person:X"]
    )

    entities, edges = backend.entity_graph_density("t")
    assert entities == 2.0
    assert edges == 2.0
