import pytest

from ogham.entity_graph import EntityEdge, Predicate, ProvenanceTree, validate_derived_from
from ogham.provenance import find_derivatives, trace_provenance
from ogham.tools.entity_graph import find_derivatives_impl, trace_provenance_impl


def test_validate_none_is_empty():
    assert validate_derived_from(None) == []


def test_validate_normalizes_elements():
    out = validate_derived_from(
        [
            {"source_edge_id": 5},
            {"source_memory_id": "abc", "reasoning": "because"},
            {"source_edge_id": 7, "source_memory_id": "def"},
        ]
    )
    assert out == [
        {"source_edge_id": 5},
        {"source_memory_id": "abc", "reasoning": "because"},
        {"source_edge_id": 7, "source_memory_id": "def"},
    ]


def test_validate_rejects_bad_shapes():
    for bad in (
        "notalist",
        [1],
        [{}],
        [{"reasoning": "x"}],
        [{"source_edge_id": "5"}],
        [{"source_memory_id": 9}],
        [{"source_edge_id": True}],  # isinstance(True, int) footgun -- bool must be rejected
        [{"source_edge_id": False}],
    ):
        with pytest.raises(ValueError):
            validate_derived_from(bad)  # type: ignore[arg-type]


def test_provenance_tree_constructs():
    t = ProvenanceTree(nodes=[], links=[], root_memories=["u1"])
    assert t.root_memories == ["u1"]


def _edge(eid, derived_from=None, fact_id=None):
    from datetime import datetime, timezone

    return EntityEdge(
        id=eid,
        subject_id=1,
        predicate=Predicate("DEPENDS_ON"),
        object_id=2,
        profile="work",
        fact_id=fact_id,
        strength=1.0,
        metadata={},
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
        derived_from=derived_from or [],
    )


class _FakeGraph:
    def __init__(self, edges):
        self._by_id = {e.id: e for e in edges}

    def fetch_edge(self, edge_id, profile):
        return self._by_id.get(edge_id)

    def find_citing_edges(self, *, source_edge_id, source_memory_id, profile):
        out = []
        for e in self._by_id.values():
            for el in e.derived_from:
                if source_edge_id is not None and el.get("source_edge_id") == source_edge_id:
                    out.append(e)
                elif (
                    source_memory_id is not None and el.get("source_memory_id") == source_memory_id
                ):
                    out.append(e)
        return out


def test_trace_no_provenance_returns_start_plus_factid_root():
    from uuid import UUID

    fid = UUID("11111111-1111-1111-1111-111111111111")
    g = _FakeGraph([_edge(1, fact_id=fid)])
    t = trace_provenance(g, 1, "work")
    assert [n.id for n in t.nodes] == [1]
    assert t.root_memories == [str(fid)]


def test_trace_two_parents():
    child = _edge(3, derived_from=[{"source_edge_id": 1}, {"source_edge_id": 2}])
    g = _FakeGraph([child, _edge(1), _edge(2)])
    t = trace_provenance(g, 3, "work")
    assert set(n.id for n in t.nodes) == {1, 2, 3}
    assert {(link["from_edge_id"], link["to_edge_id"]) for link in t.links} == {(3, 1), (3, 2)}


def test_trace_three_hops_and_max_depth():
    g = _FakeGraph(
        [
            _edge(1, derived_from=[{"source_edge_id": 2}]),
            _edge(2, derived_from=[{"source_edge_id": 3}]),
            _edge(3, derived_from=[{"source_edge_id": 4}]),
            _edge(4),
        ]
    )
    assert {n.id for n in trace_provenance(g, 1, "work", max_depth=3).nodes} == {1, 2, 3, 4}
    assert {n.id for n in trace_provenance(g, 1, "work", max_depth=1).nodes} == {1, 2}


def test_trace_source_memory_becomes_root():
    g = _FakeGraph([_edge(1, derived_from=[{"source_memory_id": "mem-9"}])])
    assert trace_provenance(g, 1, "work").root_memories == ["mem-9"]


def test_trace_cycle_guard_and_dangling():
    # edge 2 cites both edge 1 (cycle) and edge 99 (dangling -- unresolvable).
    g = _FakeGraph(
        [
            _edge(1, derived_from=[{"source_edge_id": 2}]),
            _edge(2, derived_from=[{"source_edge_id": 1}, {"source_edge_id": 99}]),
        ]
    )
    t = trace_provenance(g, 1, "work")
    assert {n.id for n in t.nodes} == {1, 2}  # 99 dangling -> skipped, no infinite loop


def test_trace_missing_start_edge_empty():
    t = trace_provenance(_FakeGraph([]), 42, "work")
    assert t.nodes == [] and t.root_memories == []


def test_find_derivatives_transitive():
    g = _FakeGraph(
        [
            _edge(10),
            _edge(11, derived_from=[{"source_edge_id": 10}]),
            _edge(12, derived_from=[{"source_edge_id": 11}]),
        ]
    )
    ids = {e.id for e in find_derivatives(g, 10, "work")}
    assert ids == {11, 12}


def test_find_derivatives_by_memory():
    g = _FakeGraph([_edge(20, derived_from=[{"source_memory_id": "mem-1"}])])
    assert {e.id for e in find_derivatives(g, "mem-1", "work")} == {20}


def test_trace_impl_serializes_tree():
    child = _edge(3, derived_from=[{"source_edge_id": 1, "reasoning": "r"}])
    g = _FakeGraph([child, _edge(1)])
    out = trace_provenance_impl(graph=g, edge_id=3, profile="work")
    assert {n["id"] for n in out["nodes"]} == {1, 3}
    assert out["links"] == [{"from_edge_id": 3, "to_edge_id": 1, "reasoning": "r"}]
    assert "derived_from" in out["nodes"][0]  # edges serialize their lineage
    assert out["root_memories"] == []


def test_find_derivatives_impl_serializes_list():
    g = _FakeGraph([_edge(10), _edge(11, derived_from=[{"source_edge_id": 10}])])
    out = find_derivatives_impl(graph=g, source_id=10, profile="work")
    assert [e["id"] for e in out["edges"]] == [11]
