"""query_join MCP tool -- unit tests with a mocked backend."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID

from ogham.entity_graph import Entity, EntityEdge, JoinResult, Predicate


def _make_graph_returning(entities, edges, citations):
    graph = MagicMock()
    graph.query_join.return_value = JoinResult(entities=entities, edges=edges, citations=citations)
    return graph


def test_query_join_serializes_join_result():
    from ogham.tools.entity_graph import query_join_impl

    now = datetime.now(timezone.utc)
    a = Entity(id=1, canonical_name="A", entity_type="s")
    b = Entity(id=2, canonical_name="B", entity_type="s")
    edge = EntityEdge(
        id=100,
        subject_id=1,
        predicate=Predicate("DEPENDS_ON"),
        object_id=2,
        profile="w",
        fact_id=None,
        strength=1.0,
        metadata={},
        valid_from=now,
        valid_to=None,
    )
    graph = _make_graph_returning([a, b], [edge], [])

    result = query_join_impl(
        graph=graph,
        allowed_predicates={"DEPENDS_ON"},
        start_entity="A",
        predicate_path=["DEPENDS_ON"],
        profile="w",
        hop_limit=2,
    )
    assert len(result["entities"]) == 2
    assert len(result["edges"]) == 1
    assert result["edges"][0]["predicate"] == "DEPENDS_ON"


def test_query_join_returns_empty_on_none():
    from ogham.tools.entity_graph import query_join_impl

    graph = MagicMock()
    graph.query_join.return_value = None

    result = query_join_impl(
        graph=graph,
        allowed_predicates={"DEPENDS_ON"},
        start_entity="A",
        predicate_path=["DEPENDS_ON"],
        profile="w",
        hop_limit=2,
    )
    assert result == {"entities": [], "edges": [], "citations": []}


def test_query_join_rejects_unknown_predicate():
    import pytest

    from ogham.tools.entity_graph import query_join_impl

    graph = MagicMock()
    with pytest.raises(ValueError, match="not in vocabulary"):
        query_join_impl(
            graph=graph,
            allowed_predicates={"OWNS"},
            start_entity="A",
            predicate_path=["BOGUS"],
            profile="w",
            hop_limit=1,
        )


def test_query_join_rejects_missing_hop_limit():
    """Defense-in-depth: hop_limit is required per TBU-109, even at the impl layer."""
    import pytest

    from ogham.tools.entity_graph import query_join_impl

    graph = MagicMock()
    with pytest.raises(ValueError, match="hop_limit is required"):
        query_join_impl(
            graph=graph,
            allowed_predicates={"DEPENDS_ON"},
            start_entity="A",
            predicate_path=["DEPENDS_ON"],
            profile="w",
            hop_limit=0,
        )


def test_query_join_serializes_uuid_fields():
    """fact_id (UUID) and valid_to (datetime) must round-trip to JSON-safe strings."""
    from ogham.tools.entity_graph import query_join_impl

    valid_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
    valid_to = datetime(2026, 6, 1, tzinfo=timezone.utc)
    fact_id = UUID("00000000-0000-0000-0000-000000000001")
    a = Entity(id=1, canonical_name="A", entity_type="s")
    b = Entity(id=2, canonical_name="B", entity_type="s")
    edge = EntityEdge(
        id=100,
        subject_id=1,
        predicate=Predicate("DEPENDS_ON"),
        object_id=2,
        profile="w",
        fact_id=fact_id,
        strength=0.5,
        metadata={"note": "x"},
        valid_from=valid_from,
        valid_to=valid_to,
    )
    graph = _make_graph_returning([a, b], [edge], [fact_id])

    result = query_join_impl(
        graph=graph,
        allowed_predicates={"DEPENDS_ON"},
        start_entity="A",
        predicate_path=["DEPENDS_ON"],
        profile="w",
        hop_limit=1,
    )

    serialized_edge = result["edges"][0]
    assert serialized_edge["fact_id"] == "00000000-0000-0000-0000-000000000001"
    assert serialized_edge["valid_from"] == valid_from.isoformat()
    assert serialized_edge["valid_to"] == valid_to.isoformat()
    assert result["citations"] == ["00000000-0000-0000-0000-000000000001"]


def test_query_join_uses_active_profile_when_not_specified():
    """Regression guard: omitting `profile` must not silently query 'default'.

    Mirrors ogham.tools.entity_graph.store_triple's active-profile resolution.
    """
    from unittest.mock import patch

    from ogham.tools import entity_graph as et

    graph = MagicMock()
    graph.query_join.return_value = None
    with (
        patch.object(et, "get_entity_graph_and_vocab", return_value=(graph, {"DEPENDS_ON"})),
        patch.object(et, "get_active_profile", return_value="work"),
    ):
        et.query_join(start_entity="A", predicate_path=["DEPENDS_ON"], hop_limit=2)

    assert graph.query_join.call_args.kwargs["profile"] == "work"


def test_query_join_explicit_profile_overrides_active_profile():
    """An explicitly-passed profile kwarg must win over get_active_profile()."""
    from unittest.mock import patch

    from ogham.tools import entity_graph as et

    graph = MagicMock()
    graph.query_join.return_value = None
    with (
        patch.object(et, "get_entity_graph_and_vocab", return_value=(graph, {"DEPENDS_ON"})),
        patch.object(et, "get_active_profile", return_value="work"),
    ):
        et.query_join(
            start_entity="A",
            predicate_path=["DEPENDS_ON"],
            hop_limit=2,
            profile="personal",
        )

    assert graph.query_join.call_args.kwargs["profile"] == "personal"


def test_query_join_predicate_path_param_uses_coercion():
    """Regression guard for FastMCP JSON-string coercion (-32602 Invalid params).

    `predicate_path` must use the same BeforeValidator(_coerce_list) idiom as
    the optional `ListStr` fields in ogham.tools.memory, but stay required
    (no default) per the tool's signature.
    """
    import inspect
    from typing import get_args

    from pydantic import BeforeValidator

    from ogham.tools.entity_graph import query_join
    from ogham.tools.memory import _coerce_list

    sig = inspect.signature(query_join, eval_str=True)
    param = sig.parameters["predicate_path"]
    assert param.default is inspect.Parameter.empty, "predicate_path must be required (no default)"
    validators = [a for a in get_args(param.annotation) if isinstance(a, BeforeValidator)]
    assert validators, "predicate_path annotation must use a BeforeValidator for FastMCP coercion"
    assert validators[0].func is _coerce_list
