from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")


@pytest.fixture(autouse=True)
def reset_profile():
    import ogham.tools.memory as mem

    mem._active_profile = "default"
    yield
    mem._active_profile = "default"


def test_summarize_recent_prompt():
    """summarize-recent prompt should return messages with memory content"""
    from ogham.prompts import summarize_recent

    with patch("ogham.prompts.list_recent_memories") as mock_list:
        mock_list.return_value = [
            {"content": "memory one", "tags": ["tag1"], "created_at": "2026-01-01"},
            {"content": "memory two", "tags": [], "created_at": "2026-01-02"},
        ]
        result = summarize_recent(limit=2)

    assert isinstance(result, str)
    assert "memory one" in result
    assert "memory two" in result


def test_summarize_recent_empty():
    """summarize-recent should handle no memories"""
    from ogham.prompts import summarize_recent

    with patch("ogham.prompts.list_recent_memories") as mock_list:
        mock_list.return_value = []
        result = summarize_recent()

    assert "No memories" in result


def test_find_decisions_prompt():
    """find-decisions prompt should search with decision tag"""
    from ogham.prompts import find_decisions

    with (
        patch("ogham.prompts.generate_embedding") as mock_embed,
        patch("ogham.prompts.hybrid_search_memories") as mock_search,
    ):
        mock_embed.return_value = [0.1] * 1024
        mock_search.return_value = [
            {"content": "chose postgres", "relevance": 0.05, "tags": ["type:decision"]},
        ]
        result = find_decisions(topic="database")

    assert isinstance(result, str)
    assert "chose postgres" in result


def test_profile_overview_prompt():
    """profile-overview prompt should include stats"""
    from ogham.prompts import profile_overview

    with (
        patch("ogham.prompts.get_memory_stats") as mock_stats,
        patch("ogham.prompts.list_recent_memories") as mock_list,
    ):
        mock_stats.return_value = {
            "profile": "default",
            "total": 42,
            "sources": {"claude-code": 30},
            "top_tags": [{"tag": "project:foo", "count": 10}],
        }
        mock_list.return_value = []
        result = profile_overview()

    assert isinstance(result, str)
    assert "42" in result


def test_cleanup_check_prompt():
    """cleanup-check prompt should report expired count"""
    from ogham.prompts import cleanup_check

    with patch("ogham.prompts.db_count_expired") as mock_count:
        mock_count.return_value = 7
        result = cleanup_check()

    assert isinstance(result, str)
    assert "7" in result


def test_cleanup_check_none_expired():
    """cleanup-check should say nothing to clean when no expired"""
    from ogham.prompts import cleanup_check

    with patch("ogham.prompts.db_count_expired") as mock_count:
        mock_count.return_value = 0
        result = cleanup_check()

    assert "No expired" in result
