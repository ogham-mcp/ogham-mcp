"""Unit tests for wiki Tier 1 context injection (v0.12).

Two layers under test:
  * `topic_summaries.search_summaries` -- the SQL helper that hits the
    partial HNSW index. Mocked at the backend level; we're checking
    the SQL shape and parameter pass-through, not real similarity.
  * `service._wiki_injection_results` -- the orchestrator the search
    pipeline calls. Tests cover the feature flag (must default off),
    the failure-soft behaviour (search errors don't sink the search
    path), and the result shape (tagged result_type + provenance
    metadata).

Postgres integration coverage will be added against a live DB after
the BEAM/LME regression check confirms injection doesn't degrade
retrieval. For now the unit layer protects the wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------- #
# search_summaries
# --------------------------------------------------------------------- #


def test_search_summaries_empty_embedding_returns_empty():
    from ogham.topic_summaries import search_summaries

    assert search_summaries("work", [], top_k=3) == []


def test_search_summaries_zero_top_k_returns_empty():
    from ogham.topic_summaries import search_summaries

    assert search_summaries("work", [0.1] * 512, top_k=0) == []


def test_search_summaries_passes_correct_params_to_backend():
    """search_summaries now dispatches to backend.wiki_topic_search RPC.

    The SQL shape itself (status='fresh' filter, embedding <=> ordering,
    LIMIT, threshold) lives in migration 031's wiki_topic_search function
    and is verified end-to-end in tests/test_wiki_integration.py.
    """
    from ogham import topic_summaries

    fake_backend = MagicMock()
    fake_backend.wiki_topic_search.return_value = []

    with patch.object(topic_summaries, "get_backend", return_value=fake_backend):
        topic_summaries.search_summaries(
            profile="work",
            query_embedding=[0.5] * 512,
            top_k=5,
            min_similarity=0.6,
        )

    fake_backend.wiki_topic_search.assert_called_once()
    kwargs = fake_backend.wiki_topic_search.call_args.kwargs
    assert kwargs["profile"] == "work"
    assert kwargs["top_k"] == 5
    assert kwargs["min_similarity"] == 0.6
    assert len(kwargs["query_embedding"]) == 512


def test_search_summaries_returns_dict_per_row():
    from ogham import topic_summaries

    fake_backend = MagicMock()
    fake_row = {
        "id": "abc",
        "topic_key": "quantum",
        "profile_id": "work",
        "content": "compiled body",
        "source_count": 4,
        "source_cursor": "uuid",
        "source_hash": b"\x00" * 32,
        "model_used": "openai/gpt-4o-mini",
        "version": 1,
        "status": "fresh",
        "updated_at": datetime(2026, 4, 27, tzinfo=timezone.utc),
        "similarity": 0.82,
    }
    fake_backend.wiki_topic_search.return_value = [fake_row]

    with patch.object(topic_summaries, "get_backend", return_value=fake_backend):
        out = topic_summaries.search_summaries("work", [0.1] * 512, top_k=3)

    assert len(out) == 1
    assert out[0]["topic_key"] == "quantum"
    assert out[0]["similarity"] == 0.82
    assert out[0]["status"] == "fresh"


# --------------------------------------------------------------------- #
# _wiki_injection_results -- the orchestrator
# --------------------------------------------------------------------- #


def test_wiki_injection_default_state():
    """v0.12.1: injection defaults ON because the layering refactor moved
    preamble out of service.search_memories_enriched and into the MCP tool
    layer. Benchmarks calling the service directly are not affected by
    this flag at all. If a future refactor accidentally moves injection
    back into service.py, this assertion plus
    test_service_search_memories_enriched_does_not_inject below will both
    fail and force a deliberate decision."""
    from ogham.config import settings

    assert settings.wiki_injection_enabled is True


def test_wiki_injection_returns_empty_when_flag_off():
    from ogham import service

    with patch("ogham.config.settings.wiki_injection_enabled", False):
        with patch.object(service, "search_summaries") as searched:
            out = service._wiki_injection_results("work", [0.1] * 512)

    assert out == []
    searched.assert_not_called()  # short-circuit before SQL call


def test_wiki_injection_returns_empty_when_embedding_missing():
    from ogham import service

    with patch("ogham.config.settings.wiki_injection_enabled", True):
        with patch.object(service, "search_summaries") as searched:
            out = service._wiki_injection_results("work", [])

    assert out == []
    searched.assert_not_called()


def test_wiki_injection_swallows_search_errors():
    """Wiki layer outage must not take the search path down with it."""
    from ogham import service

    with patch("ogham.config.settings.wiki_injection_enabled", True):
        with patch.object(service, "search_summaries", side_effect=RuntimeError("DB down")):
            out = service._wiki_injection_results("work", [0.1] * 512)

    assert out == []  # logged + suppressed


def test_wiki_injection_shapes_result_with_wiki_summary_tag():
    from ogham import service

    fake_row = {
        "id": "summary-1",
        "topic_key": "quantum",
        "content": "## Overview\n\nbody",
        "source_count": 4,
        "model_used": "openai/gpt-4o-mini",
        "version": 2,
        "similarity": 0.78,
    }
    with patch("ogham.config.settings.wiki_injection_enabled", True):
        with patch("ogham.config.settings.wiki_injection_top_k", 3):
            with patch("ogham.config.settings.wiki_injection_min_similarity", 0.4):
                with patch.object(service, "search_summaries", return_value=[fake_row]):
                    out = service._wiki_injection_results("work", [0.1] * 512)

    assert len(out) == 1
    r = out[0]
    assert r["result_type"] == "wiki_summary"
    assert r["topic_key"] == "quantum"
    assert r["content"] == "## Overview\n\nbody"
    assert r["similarity"] == 0.78
    assert r["version"] == 2
    assert r["tags"] == ["wiki:quantum"]
    assert r["metadata"]["wiki_summary_id"] == "summary-1"
    assert r["metadata"]["topic_key"] == "quantum"


def test_wiki_injection_passes_settings_to_search_helper():
    """Top-K and min-similarity from settings flow through to the SQL call."""
    from ogham import service

    with patch("ogham.config.settings.wiki_injection_enabled", True):
        with patch("ogham.config.settings.wiki_injection_top_k", 7):
            with patch("ogham.config.settings.wiki_injection_min_similarity", 0.55):
                with patch.object(service, "search_summaries", return_value=[]) as mock_search:
                    service._wiki_injection_results("personal", [0.1] * 512)

    mock_search.assert_called_once_with(
        profile="personal",
        query_embedding=[0.1] * 512,
        top_k=7,
        min_similarity=0.55,
    )


# --------------------------------------------------------------------- #
# v0.12.1 layering invariant: service.search_memories_enriched is a pure
# retrieval engine; injection happens at the MCP tool layer
# (tools/memory.py::hybrid_search) so benchmarks calling the service
# directly never see preamble pollution. These tests lock the boundary.
# --------------------------------------------------------------------- #


def test_service_search_memories_enriched_does_not_inject():
    """search_memories_enriched must NOT prepend wiki preamble.

    BEAM/LME and any retrieval-only caller depend on this; v0.12.0 had
    the injection here and benchmark scoring crashed (-38pp MRR) because
    injected wiki_summary IDs occupy the top ranks the scorer compares
    against gold. The fix: lift injection up to the MCP-tool layer.
    """
    from ogham import service

    fake_results = [
        {"id": "mem-1", "content": "real memory 1", "result_type": "memory"},
        {"id": "mem-2", "content": "real memory 2", "result_type": "memory"},
    ]
    with (
        patch.object(service, "_search_memories_raw", return_value=fake_results),
        patch.object(service, "_maybe_rerank", side_effect=lambda q, r, lim: r),
        patch.object(service, "_reorder_for_attention", side_effect=lambda r: r),
        patch.object(service, "record_access"),
        patch.object(service, "emit_audit_event"),
        patch.object(service, "_lifecycle_submit"),
        patch.object(service, "generate_embedding", return_value=[0.1] * 512),
        # Injection helper would still trigger if called -- assert it isn't.
        patch.object(service, "_wiki_injection_results") as mock_inject,
        patch("ogham.config.settings.wiki_injection_enabled", True),
    ):
        out = service.search_memories_enriched(
            query="quantum",
            profile="work",
            limit=10,
        )

    mock_inject.assert_not_called()
    assert all(r.get("result_type") != "wiki_summary" for r in out)
    assert [r["id"] for r in out] == ["mem-1", "mem-2"]


def test_hybrid_search_tool_prepends_preamble_when_flag_on():
    """The MCP tool layer is where preamble gets attached."""
    from ogham.tools import memory as memory_tools

    fake_results = [
        {"id": "mem-1", "content": "real 1", "result_type": "memory"},
    ]
    fake_preamble = [
        {"id": "summary-1", "content": "## body", "result_type": "wiki_summary"},
    ]
    with (
        patch.object(memory_tools, "get_active_profile", return_value="work"),
        patch("ogham.config.settings.wiki_injection_enabled", True),
        patch("ogham.service.search_memories_enriched", return_value=fake_results),
        patch("ogham.service._wiki_injection_results", return_value=fake_preamble),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        out = memory_tools.hybrid_search(query="quantum", limit=10)

    assert len(out) == 2
    assert out[0]["result_type"] == "wiki_summary"  # preamble first
    assert out[1]["result_type"] == "memory"


def test_hybrid_search_tool_does_not_inject_when_flag_off():
    """Default behaviour: tool-layer mirrors service-layer (clean retrieval)."""
    from ogham.tools import memory as memory_tools

    fake_results = [{"id": "mem-1", "content": "real", "result_type": "memory"}]
    with (
        patch.object(memory_tools, "get_active_profile", return_value="work"),
        patch("ogham.config.settings.wiki_injection_enabled", False),
        patch("ogham.service.search_memories_enriched", return_value=fake_results),
        patch("ogham.service._wiki_injection_results") as mock_inject,
    ):
        out = memory_tools.hybrid_search(query="quantum", limit=10)

    mock_inject.assert_not_called()
    assert out == fake_results


def test_hybrid_search_tool_skips_injection_under_extract_facts():
    """extract_facts mode pipes raw memories to an LLM extractor; preamble
    would confuse the extractor. The tool layer must skip injection."""
    from ogham.tools import memory as memory_tools

    fake_results = [{"id": "mem-1", "content": "fact", "result_type": "memory"}]
    with (
        patch.object(memory_tools, "get_active_profile", return_value="work"),
        patch("ogham.config.settings.wiki_injection_enabled", True),
        patch("ogham.service.search_memories_enriched", return_value=fake_results),
        patch("ogham.service._wiki_injection_results") as mock_inject,
    ):
        out = memory_tools.hybrid_search(query="quantum", limit=10, extract_facts=True)

    mock_inject.assert_not_called()
    assert out == fake_results


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
