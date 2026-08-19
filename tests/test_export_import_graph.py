"""The OKF export path fetches the graph and threads it into the bundle.

Fails OPEN by design: an install that never ran the entity migrations must
still produce a valid memories-only bundle rather than an exception. Same
posture as _demote_superseded in service.py.
"""

from unittest.mock import MagicMock, patch

from ogham.entity_graph import Entity
from ogham.export_import import export_memories

ENTITY = Entity(id=42, canonical_name="Ogham", entity_type="project")
EDGE = object()  # opaque -- this file asserts wiring, not marshalling
ALIASES = {42: ["OpenBrain"]}
BRIDGE = {"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": [42]}


def _graph(entities=None, edges=None, aliases=None):
    g = MagicMock()
    g.list_entities.return_value = entities or []
    g.list_edges.return_value = edges or []
    g.list_aliases.return_value = aliases or {}
    return g


def _export_with_writer(tmp_path, monkeypatch, graph, mem_entities):
    monkeypatch.chdir(tmp_path)
    with (
        patch("ogham.export_import._list_all_memories", return_value=[]),
        patch("ogham.database.get_entity_graph_and_vocab", return_value=(graph, frozenset())),
        patch("ogham.database.get_memory_entities", return_value=mem_entities),
        patch("ogham.okf.bundle.export_okf_bundle") as writer,
    ):
        export_memories("default", format="okf")
    return writer.call_args.kwargs


def test_every_leg_of_the_graph_reaches_the_bundle_writer(tmp_path, monkeypatch):
    """All four kwargs, not just `entities`.

    Mutation-checked: deleting `edges=`, `aliases=` and `memory_entities=` from
    the export_okf_bundle(...) call left the whole default suite green, because
    the original test asserted only `entities` and separately that list_edges
    had been *called*. Every real export would then have shipped bundles with no
    MENTIONS on any memory and no aliases on any entity -- the entire D7
    Schema.org bridge, silently gone, with CI green.

    All four are keyword-only with None defaults on export_okf_bundle, so
    dropping one raises nothing. Only an assertion catches it.
    """
    graph = _graph([ENTITY], [EDGE], ALIASES)
    kwargs = _export_with_writer(tmp_path, monkeypatch, graph, BRIDGE)

    assert kwargs["entities"] == [ENTITY]
    assert kwargs["edges"] == [EDGE]
    assert kwargs["aliases"] == ALIASES
    assert kwargs["memory_entities"] == BRIDGE


def test_each_read_is_scoped_to_the_requested_profile(tmp_path, monkeypatch):
    graph = _graph([ENTITY], [EDGE], ALIASES)
    _export_with_writer(tmp_path, monkeypatch, graph, BRIDGE)

    graph.list_entities.assert_called_once_with("default")
    graph.list_edges.assert_called_once_with("default")
    graph.list_aliases.assert_called_once_with("default")


def test_export_survives_a_backend_with_no_entities_layer(tmp_path, monkeypatch):
    """Pre-036 installs have no entities tables. That must degrade to a
    memories-only bundle, not raise."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("ogham.export_import._list_all_memories", return_value=[]),
        patch("ogham.database.get_entity_graph_and_vocab", side_effect=RuntimeError("no table")),
    ):
        path = export_memories("default", format="okf")

    assert path  # a bundle was still written


def test_a_read_failing_midway_yields_no_graph_at_all(tmp_path, monkeypatch):
    """The fetch is all-or-nothing, and the earlier reads must not survive.

    Deterministic on a partially-migrated install, not just a transient error:
    `derived_from` arrives by ALTER TABLE in migration 046 and only list_edges
    names it, so an install at 041-045 succeeds on list_entities and fails on
    list_edges every time. Leaking the entities through would stamp
    ogham_graph_version into the manifest and write an entities/ tree in which
    every single triple is missing -- indistinguishable, to a consumer, from a
    graph that genuinely has no edges.
    """
    graph = _graph([ENTITY], [EDGE], ALIASES)
    graph.list_edges.side_effect = RuntimeError('column "derived_from" does not exist')

    kwargs = _export_with_writer(tmp_path, monkeypatch, graph, BRIDGE)

    assert kwargs["entities"] == []
    assert kwargs["edges"] == []
    assert kwargs["aliases"] == {}
    assert kwargs["memory_entities"] == {}


def test_the_bridge_read_failing_last_also_yields_no_graph(tmp_path, monkeypatch):
    """The 4th read is the easiest one to forget: entities and edges have both
    already succeeded by then."""
    graph = _graph([ENTITY], [EDGE], ALIASES)
    monkeypatch.chdir(tmp_path)
    with (
        patch("ogham.export_import._list_all_memories", return_value=[]),
        patch("ogham.database.get_entity_graph_and_vocab", return_value=(graph, frozenset())),
        patch("ogham.database.get_memory_entities", side_effect=RuntimeError("no join table")),
        patch("ogham.okf.bundle.export_okf_bundle") as writer,
    ):
        export_memories("default", format="okf")

    assert writer.call_args.kwargs["entities"] == []


def test_non_okf_formats_do_not_touch_the_graph(tmp_path, monkeypatch):
    """json/markdown export must not pay for a graph fetch it never uses."""
    monkeypatch.chdir(tmp_path)
    graph = _graph()
    with (
        patch("ogham.export_import._list_all_memories", return_value=[]),
        patch("ogham.database.get_entity_graph_and_vocab", return_value=(graph, frozenset())),
    ):
        export_memories("default", format="json")

    graph.list_entities.assert_not_called()
