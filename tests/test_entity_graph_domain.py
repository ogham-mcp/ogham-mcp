"""Domain module tests -- Predicate + dataclasses + Protocol shape."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest


def test_predicate_newtype_accepts_valid_string():
    from ogham.entity_graph import make_predicate

    p = make_predicate("DEPENDS_ON", ["DEPENDS_ON", "OWNS"])
    assert p == "DEPENDS_ON"
    assert isinstance(p, str)


def test_make_predicate_rejects_unknown():
    from ogham.entity_graph import make_predicate

    with pytest.raises(ValueError, match="not in vocabulary"):
        make_predicate("BOGUS_PREDICATE", ["DEPENDS_ON", "OWNS"])


def test_entity_dataclass_frozen():
    from ogham.entity_graph import Entity

    e = Entity(id=1, canonical_name="AuthService", entity_type="service")
    with pytest.raises(Exception):
        e.canonical_name = "changed"  # pyright: ignore[reportAttributeAccessIssue] -- test intent
    assert e.canonical_name == "AuthService"


def test_entity_edge_dataclass_shape():
    from ogham.entity_graph import EntityEdge, make_predicate

    p = make_predicate("DEPENDS_ON", ["DEPENDS_ON"])
    now = datetime.now(timezone.utc)
    fid = uuid4()
    edge = EntityEdge(
        id=42,
        subject_id=1,
        predicate=p,
        object_id=2,
        profile="work",
        fact_id=fid,
        strength=1.0,
        metadata={},
        valid_from=now,
        valid_to=None,
    )
    assert edge.valid_to is None
    assert edge.predicate == "DEPENDS_ON"


def test_join_result_shape():
    from ogham.entity_graph import Entity, EntityEdge, JoinResult, make_predicate

    p = make_predicate("DEPENDS_ON", ["DEPENDS_ON"])
    now = datetime.now(timezone.utc)
    a = Entity(id=1, canonical_name="A", entity_type="service")
    b = Entity(id=2, canonical_name="B", entity_type="service")
    e = EntityEdge(
        id=1,
        subject_id=1,
        predicate=p,
        object_id=2,
        profile="w",
        fact_id=None,
        strength=1.0,
        metadata={},
        valid_from=now,
        valid_to=None,
    )
    r = JoinResult(entities=[a, b], edges=[e], citations=[])
    assert len(r.entities) == 2
    assert len(r.edges) == 1
    assert r.citations == []


def test_entity_graph_protocol_signature():
    """Any class implementing every protocol method should satisfy the Protocol.

    Was "the six methods" until TBU-130 added the three enumeration primitives.
    Note what this test costs: because ``EntityGraph`` is ``runtime_checkable``,
    widening it silently un-conforms every existing implementer. That is fine
    here -- the only implementers are our two backends, both updated in the same
    change -- but it is the reason to think twice before adding a seventh method
    if anything outside this repo ever implements the protocol.
    """
    from ogham.entity_graph import EntityGraph

    class Dummy:
        def store_triple(
            self,
            subject,
            predicate,
            object_,
            source_memory_id,
            profile,
            metadata=None,
            derived_from=None,
        ):
            return 1

        def query_join(
            self, start_entity, predicate_path, profile, hop_limit, direction="outgoing"
        ):
            return None

        def add_alias(self, entity_id, alias, profile):
            return None

        def resolve_alias(self, name_or_id, profile):
            return None

        def fetch_edge(self, edge_id, profile):
            return None

        def find_citing_edges(self, *, source_edge_id, source_memory_id, profile):
            return []

        def list_entities(self, profile):
            return []

        def list_edges(self, profile, *, current_only=True):
            return []

        def list_aliases(self, profile):
            return {}

        def upsert_entity(self, canonical_name, entity_type):
            return 1

        def find_entity(self, canonical_name, entity_type):
            return None

    # runtime_checkable Protocol lets us assert this cheaply
    assert isinstance(Dummy(), EntityGraph)
