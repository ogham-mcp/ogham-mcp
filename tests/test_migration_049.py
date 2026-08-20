"""Migration 049 -- entity evidence class (TBU-261).

Pins the three things that are easy to regress and impossible to notice:

(a) the class is DERIVED from the tag prefix inside ``link_memory_entities``,
    so no client change is needed and Python/Go parity is not exposed;
(b) evidence is MONOTONE -- an entity established from adapter provenance
    must not be demoted by a later syntactic sighting;
(c) the column is an enumerated class, not a float, and the CHECK constraint
    is what stops it drifting into being treated as a probability.

The reason (b) matters: ``person:`` became load-bearing in the ranking term
because a value that looked like signal was consumed by code written by
someone who was not in the conversation where it was defined. A silently
demoted evidence class would fail the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SQL_DIR = Path(__file__).parent.parent / "sql/migrations"
MIG_036 = SQL_DIR / "036_entities_backfill.sql"
MIG_049 = SQL_DIR / "049_entity_evidence_class.sql"


def _can_connect() -> bool:
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        PostgresBackend()._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(not _can_connect(), reason="Postgres backend not configured or unreachable"),
]


def _apply(pg_fresh_db):
    pg_fresh_db.apply_sql(MIG_036)
    pg_fresh_db.apply_sql(MIG_049)


def _seed_memory(profile: str = "t") -> str:
    from ogham.backends.postgres import PostgresBackend

    row = PostgresBackend()._execute(
        "INSERT INTO memories (content, profile, source) "
        "VALUES ('probe', %(p)s, 't') RETURNING id::text AS id",
        {"p": profile},
        fetch="one",
    )
    assert row is not None
    return row["id"]


def _classes() -> dict[str, str]:
    from ogham.backends.postgres import PostgresBackend

    rows = PostgresBackend()._execute(
        "SELECT entity_type, canonical_name, evidence_class FROM entities", {}, fetch="all"
    )
    return {f"{r['entity_type']}:{r['canonical_name']}": r["evidence_class"] for r in rows or []}


# The four types that survived the v0.18.0 person: deletion are exactly the
# four with an unambiguous syntactic marker. If a new entity type is added
# without a decision about its class, this test is where it should surface.
SYNTACTIC_TAGS = ["entity:PostgreSQL", "file:src/app.py", "error:ValueError", "quantity:5 gb"]
INFERRED_TAGS = ["location:Paris", "event:launch", "emotion:happy", "preference:dark"]


def test_049_adds_evidence_class_column(pg_fresh_db):
    _apply(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    row = PostgresBackend()._execute(
        "SELECT data_type, column_default, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'entities' AND column_name = 'evidence_class'",
        {},
        fetch="one",
    )
    assert row is not None, "049 must add entities.evidence_class"
    assert row["data_type"] == "text", "must be an enumerated class, never a float"
    assert row["is_nullable"] == "NO"
    assert "inferred" in (row["column_default"] or ""), "unclassified must default to the weakest"


def test_link_memory_entities_derives_class_from_prefix(pg_fresh_db):
    _apply(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    memory_id = _seed_memory()
    n = PostgresBackend().link_memory_entities(
        memory_id=memory_id, profile="t", entity_tags=SYNTACTIC_TAGS + INFERRED_TAGS
    )
    assert n == len(SYNTACTIC_TAGS + INFERRED_TAGS)

    classes = _classes()
    for tag in SYNTACTIC_TAGS:
        assert classes[tag] == "syntactic", f"{tag} has an unambiguous marker"
    for tag in INFERRED_TAGS:
        assert classes[tag] == "inferred", f"{tag} is a dictionary lookup, not a marker"


def test_structured_evidence_is_not_demoted_by_a_later_syntactic_write(pg_fresh_db):
    """Evidence is monotone. This is the one that is awkward to reverse."""
    _apply(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    backend.link_memory_entities(
        memory_id=_seed_memory(), profile="t", entity_tags=["entity:PostgreSQL"]
    )
    # simulate an adapter establishing it from provenance (TBU-268)
    backend._execute(
        "UPDATE entities SET evidence_class = 'structured' WHERE canonical_name = 'PostgreSQL'",
        {},
        fetch="none",
    )

    backend.link_memory_entities(
        memory_id=_seed_memory(), profile="t", entity_tags=["entity:PostgreSQL"]
    )

    row = backend._execute(
        "SELECT evidence_class, mention_count FROM entities WHERE canonical_name = 'PostgreSQL'",
        {},
        fetch="one",
    )
    assert row is not None
    assert row["evidence_class"] == "structured", "a syntactic sighting must not demote provenance"
    assert row["mention_count"] == 2, "the mention count must still increment"


def test_evidence_class_rejects_values_outside_the_enumeration(pg_fresh_db):
    _apply(pg_fresh_db)
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    backend.link_memory_entities(
        memory_id=_seed_memory(), profile="t", entity_tags=["location:Paris"]
    )
    with pytest.raises(Exception, match="evidence_class"):
        backend._execute(
            "UPDATE entities SET evidence_class = '0.7' WHERE canonical_name = 'Paris'",
            {},
            fetch="none",
        )
