"""store_triple MCP tool -- unit tests with a mocked backend."""

from unittest.mock import MagicMock

from ogham.entity_graph import Predicate


def _make_mock_graph():
    graph = MagicMock()
    graph.store_triple.return_value = 42
    return graph


def test_store_triple_tool_returns_edge_id():
    from ogham.tools.entity_graph import store_triple_impl

    graph = _make_mock_graph()
    result = store_triple_impl(
        graph=graph,
        allowed_predicates={"DEPENDS_ON"},
        subject="AuthService",
        predicate="DEPENDS_ON",
        object_="LoginModule",
        profile="work",
        source_memory_id=None,
        metadata=None,
    )
    assert result == {"edge_id": 42}
    graph.store_triple.assert_called_once()
    call = graph.store_triple.call_args
    assert call.kwargs["subject"] == "AuthService"
    assert call.kwargs["object_"] == "LoginModule"
    assert call.kwargs["predicate"] == Predicate("DEPENDS_ON")
    assert call.kwargs["profile"] == "work"


def test_store_triple_tool_rejects_unknown_predicate():
    import pytest

    from ogham.tools.entity_graph import store_triple_impl

    graph = _make_mock_graph()
    with pytest.raises(ValueError, match="not in vocabulary"):
        store_triple_impl(
            graph=graph,
            allowed_predicates={"OWNS"},
            subject="A",
            predicate="BOGUS",
            object_="B",
            profile="w",
            source_memory_id=None,
            metadata=None,
        )


def test_store_triple_tool_passes_metadata_through():
    from ogham.tools.entity_graph import store_triple_impl

    graph = _make_mock_graph()
    store_triple_impl(
        graph=graph,
        allowed_predicates={"MENTIONS"},
        subject="AuthService",
        predicate="MENTIONS",
        object_="LoginModule",
        profile="work",
        source_memory_id=None,
        metadata={"weight": 0.7, "source": "manual"},
    )
    assert graph.store_triple.call_args.kwargs["metadata"] == {"weight": 0.7, "source": "manual"}


def test_store_triple_tool_parses_source_memory_id_to_uuid():
    from uuid import UUID

    from ogham.tools.entity_graph import store_triple_impl

    graph = _make_mock_graph()
    store_triple_impl(
        graph=graph,
        allowed_predicates={"DEPENDS_ON"},
        subject="AuthService",
        predicate="DEPENDS_ON",
        object_="LoginModule",
        profile="work",
        source_memory_id="00000000-0000-0000-0000-000000000001",
        metadata=None,
    )
    passed_id = graph.store_triple.call_args.kwargs["source_memory_id"]
    assert isinstance(passed_id, UUID)
    assert str(passed_id) == "00000000-0000-0000-0000-000000000001"


def test_store_triple_tool_raises_on_malformed_source_memory_id():
    import pytest

    from ogham.tools.entity_graph import store_triple_impl

    graph = _make_mock_graph()
    with pytest.raises(ValueError):
        store_triple_impl(
            graph=graph,
            allowed_predicates={"DEPENDS_ON"},
            subject="A",
            predicate="DEPENDS_ON",
            object_="B",
            profile="w",
            source_memory_id="not-a-uuid",
            metadata=None,
        )


def test_store_triple_metadata_param_uses_dict_any_coercion():
    """Regression guard for FastMCP JSON-string coercion (-32602 Invalid params).

    FastMCP clients sometimes serialise dict[str, Any] tool params as JSON
    strings before the transport layer; plain `dict | None` annotations
    reject that shape. `metadata` must use the same `DictAny` /
    BeforeValidator(_coerce_dict) idiom as store_memory / switch_profile in
    ogham.tools.memory. The actual JSON-string coercion path is only
    exercisable via a live FastMCP transport (integration territory,
    TBU-122/123) -- this is a shape-level guard.
    """
    import inspect
    from typing import get_args

    from pydantic import BeforeValidator

    from ogham.tools.entity_graph import store_triple
    from ogham.tools.memory import _coerce_dict

    sig = inspect.signature(store_triple, eval_str=True)
    annotation = sig.parameters["metadata"].annotation
    validators = [a for a in get_args(annotation) if isinstance(a, BeforeValidator)]
    assert validators, (
        "metadata annotation must use a BeforeValidator (DictAny) for FastMCP JSON-string coercion"
    )
    assert validators[0].func is _coerce_dict


def test_store_triple_uses_active_profile_when_not_specified():
    """Regression guard: omitting `profile` must not silently write to 'default'.

    Every other tool in ogham.tools.memory resolves an unset profile via
    get_active_profile() (see advance_lifecycle, backfill_entities_tool).
    store_triple must match -- a caller who switch_profile('work')'d and
    then calls store_triple() without an explicit profile kwarg should
    write into 'work', not silently fall back to 'default'.
    """
    from unittest.mock import patch

    from ogham.tools import entity_graph as et

    graph = _make_mock_graph()
    with (
        patch.object(et, "get_entity_graph_and_vocab", return_value=(graph, {"DEPENDS_ON"})),
        patch.object(et, "get_active_profile", return_value="work"),
    ):
        et.store_triple(subject="AuthService", predicate="DEPENDS_ON", object_="LoginModule")

    assert graph.store_triple.call_args.kwargs["profile"] == "work"


def test_store_triple_explicit_profile_overrides_active_profile():
    """An explicitly-passed profile kwarg must win over get_active_profile()."""
    from unittest.mock import patch

    from ogham.tools import entity_graph as et

    graph = _make_mock_graph()
    with (
        patch.object(et, "get_entity_graph_and_vocab", return_value=(graph, {"DEPENDS_ON"})),
        patch.object(et, "get_active_profile", return_value="work"),
    ):
        et.store_triple(
            subject="AuthService",
            predicate="DEPENDS_ON",
            object_="LoginModule",
            profile="personal",
        )

    assert graph.store_triple.call_args.kwargs["profile"] == "personal"
