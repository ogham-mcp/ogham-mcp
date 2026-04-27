"""Unit tests for walk_knowledge MCP tool + walk_memory_graph helper.

Covers the direction knob (outgoing / incoming / both), depth bounds,
SQL shape (right join clause per direction), parameter pass-through,
and the tool's response wrapping (start_id echo, node count, node
shape). Postgres integration coverage is left for the live-DB suite
since the recursive CTE relies on memory_relationships rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------- #
# walk_memory_graph (database.py helper)
# --------------------------------------------------------------------- #


def test_walk_memory_graph_rejects_invalid_direction():
    from ogham.database import walk_memory_graph

    with pytest.raises(ValueError, match="direction must be"):
        walk_memory_graph("abc", direction="sideways")


def test_walk_memory_graph_rejects_negative_depth():
    from ogham.database import walk_memory_graph

    with pytest.raises(ValueError, match="depth must be"):
        walk_memory_graph("abc", depth=-1)


def test_walk_memory_graph_rejects_excessive_depth():
    from ogham.database import walk_memory_graph

    with pytest.raises(ValueError, match="depth must be <= 5"):
        walk_memory_graph("abc", depth=6)


def _patch_backend(rows: list[dict]):
    """Patch get_backend so backend.wiki_walk_graph(...) returns `rows`.

    The SQL shape is tested in tests/test_wiki_integration.py against a
    real DB; here we only want to assert the database.walk_memory_graph
    facade's parameter passthrough and result shape.
    """
    fake = MagicMock()
    fake.wiki_walk_graph.return_value = rows
    return patch("ogham.database.get_backend", return_value=fake), fake


def test_walk_memory_graph_passes_outgoing_to_backend():
    from ogham import database

    ctx, fake = _patch_backend([])
    with ctx:
        database.walk_memory_graph("abc", depth=1, direction="outgoing")

    fake.wiki_walk_graph.assert_called_once()
    kwargs = fake.wiki_walk_graph.call_args.kwargs
    assert kwargs["start_id"] == "abc"
    assert kwargs["max_depth"] == 1
    assert kwargs["direction"] == "outgoing"


def test_walk_memory_graph_passes_incoming_to_backend():
    from ogham import database

    ctx, fake = _patch_backend([])
    with ctx:
        database.walk_memory_graph("abc", depth=1, direction="incoming")

    kwargs = fake.wiki_walk_graph.call_args.kwargs
    assert kwargs["direction"] == "incoming"


def test_walk_memory_graph_passes_both_to_backend():
    from ogham import database

    ctx, fake = _patch_backend([])
    with ctx:
        database.walk_memory_graph("abc", depth=2, direction="both")

    kwargs = fake.wiki_walk_graph.call_args.kwargs
    assert kwargs["direction"] == "both"
    assert kwargs["max_depth"] == 2


def test_walk_memory_graph_passes_relationship_types_filter():
    from ogham import database

    ctx, fake = _patch_backend([])
    with ctx:
        database.walk_memory_graph(
            "abc",
            depth=1,
            direction="both",
            relationship_types=["similar_to", "contradicts"],
        )

    kwargs = fake.wiki_walk_graph.call_args.kwargs
    assert kwargs["relationship_types"] == ["similar_to", "contradicts"]


def test_walk_memory_graph_passes_no_types_filter_when_none():
    from ogham import database

    ctx, fake = _patch_backend([])
    with ctx:
        database.walk_memory_graph("abc", depth=1, direction="both")

    kwargs = fake.wiki_walk_graph.call_args.kwargs
    assert kwargs["relationship_types"] is None


def test_walk_memory_graph_returns_list_of_dicts():
    from ogham import database

    fake_row = {
        "id": "node-1",
        "content": "neighbouring memory",
        "metadata": {},
        "source": "claude",
        "tags": ["foo"],
        "confidence": 0.9,
        "depth": 1,
        "relationship": "similar_to",
        "edge_strength": 0.85,
        "connected_from": "abc",
        "direction_used": "outgoing",
    }
    ctx, fake = _patch_backend([fake_row])
    with ctx:
        out = database.walk_memory_graph("abc", depth=1, direction="outgoing")

    assert len(out) == 1
    assert out[0]["id"] == "node-1"
    assert out[0]["depth"] == 1
    assert out[0]["direction_used"] == "outgoing"


# --------------------------------------------------------------------- #
# walk_knowledge MCP tool
# --------------------------------------------------------------------- #


def test_walk_knowledge_invalid_direction_returns_error_dict():
    from ogham.tools import wiki

    out = wiki.walk_knowledge(start_id="abc", direction="sideways")
    assert out["status"] == "error"
    assert "direction" in out["message"]


def test_walk_knowledge_propagates_validation_error_from_helper():
    """ValueError from the helper (depth out of bounds) becomes a dict, not a raise."""
    from ogham.tools import wiki

    with patch.object(wiki, "walk_memory_graph", side_effect=ValueError("depth must be <= 5")):
        out = wiki.walk_knowledge(start_id="abc", depth=10)

    assert out["status"] == "error"
    assert "depth must be <= 5" in out["message"]


def test_walk_knowledge_returns_wrapped_response():
    from ogham.tools import wiki

    fake_rows = [
        {
            "id": "node-1",
            "content": "first hop",
            "tags": ["foo"],
            "source": "claude",
            "confidence": 0.9,
            "depth": 1,
            "relationship": "similar_to",
            "edge_strength": 0.85,
            "connected_from": "abc",
            "direction_used": "outgoing",
        },
        {
            "id": "node-2",
            "content": "second hop",
            "tags": [],
            "source": "cursor",
            "confidence": 0.7,
            "depth": 2,
            "relationship": "similar_to",
            "edge_strength": 0.6,
            "connected_from": "node-1",
            "direction_used": "outgoing",
        },
    ]
    with patch.object(wiki, "walk_memory_graph", return_value=fake_rows):
        out = wiki.walk_knowledge(start_id="abc", depth=2, direction="outgoing")

    assert out["start_id"] == "abc"
    assert out["depth"] == 2
    assert out["direction"] == "outgoing"
    assert out["node_count"] == 2
    assert out["nodes"][0]["id"] == "node-1"
    assert out["nodes"][0]["connected_from"] == "abc"
    assert out["nodes"][1]["depth"] == 2


def test_walk_knowledge_passes_args_to_helper():
    from ogham.tools import wiki

    with patch.object(wiki, "walk_memory_graph", return_value=[]) as mock_walk:
        wiki.walk_knowledge(
            start_id="abc",
            depth=3,
            direction="incoming",
            min_strength=0.4,
            relationship_types=["contradicts"],
            limit=20,
        )

    mock_walk.assert_called_once_with(
        start_id="abc",
        depth=3,
        direction="incoming",
        min_strength=0.4,
        relationship_types=["contradicts"],
        limit=20,
    )


def test_walk_knowledge_empty_graph_returns_zero_nodes():
    from ogham.tools import wiki

    with patch.object(wiki, "walk_memory_graph", return_value=[]):
        out = wiki.walk_knowledge(start_id="abc")

    assert out["node_count"] == 0
    assert out["nodes"] == []
    assert out["start_id"] == "abc"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
