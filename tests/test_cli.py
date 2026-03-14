from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")


def test_cli_health():
    """ogham health should show connection status"""
    from ogham.cli import app

    with patch("ogham.health.full_health_check") as mock_health:
        mock_health.return_value = {
            "supabase": {"status": "ok", "connected": True},
            "embedding": {"status": "ok", "provider": "ollama"},
            "config": {"status": "ok", "issues": []},
        }
        result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "ok" in result.output.lower()


def test_cli_profiles():
    """ogham profiles should list profiles"""
    from ogham.cli import app

    with patch("ogham.database.list_profiles") as mock_profiles:
        mock_profiles.return_value = [
            {"profile": "default", "count": 5},
            {"profile": "work", "count": 3},
        ]
        result = runner.invoke(app, ["profiles"])

    assert result.exit_code == 0
    assert "default" in result.output
    assert "work" in result.output


def test_cli_stats():
    """ogham stats should show profile stats"""
    from ogham.cli import app

    with patch("ogham.database.get_memory_stats") as mock_stats:
        mock_stats.return_value = {
            "profile": "default",
            "total": 42,
            "sources": {"claude-code": 30},
            "top_tags": [{"tag": "project:foo", "count": 10}],
        }
        result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    assert "42" in result.output


def test_cli_search():
    """ogham search should perform hybrid search"""
    from ogham.cli import app

    with (
        patch("ogham.embeddings.generate_embedding") as mock_embed,
        patch("ogham.database.hybrid_search_memories") as mock_search,
    ):
        mock_embed.return_value = [0.1] * 1024
        mock_search.return_value = [
            {
                "id": "abc123",
                "content": "test memory",
                "relevance": 0.052,
                "tags": ["tag1"],
                "source": "test",
            }
        ]
        result = runner.invoke(app, ["search", "test query"])

    assert result.exit_code == 0
    assert "test memory" in result.output


def test_cli_list():
    """ogham list should show recent memories"""
    from ogham.cli import app

    with patch("ogham.database.list_recent_memories") as mock_list:
        mock_list.return_value = [
            {
                "id": "abc123",
                "content": "recent memory",
                "tags": [],
                "source": "test",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
        result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "recent memory" in result.output


def test_cli_cleanup():
    """ogham cleanup should remove expired memories"""
    from ogham.cli import app

    with (
        patch("ogham.database.count_expired") as mock_count,
        patch("ogham.database.cleanup_expired") as mock_cleanup,
    ):
        mock_count.return_value = 3
        mock_cleanup.return_value = 3
        result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 0
    assert "3" in result.output


def test_cli_cleanup_nothing():
    """ogham cleanup with nothing expired"""
    from ogham.cli import app

    with patch("ogham.database.count_expired") as mock_count:
        mock_count.return_value = 0
        result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0
    assert "No expired" in result.output


def test_cli_openapi(tmp_path):
    """ogham openapi should generate a spec file"""
    from ogham.cli import app

    output_file = tmp_path / "openapi.json"
    result = runner.invoke(app, ["openapi", "--output", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    import json

    content = json.loads(output_file.read_text())
    assert "openapi" in content
    assert "store_memory" in str(content)


def test_cli_store():
    """ogham store should store a memory"""
    from ogham.cli import app

    with (
        patch("ogham.embeddings.generate_embedding") as mock_embed,
        patch("ogham.database.store_memory") as mock_store,
        patch("ogham.database.get_profile_ttl") as mock_ttl,
    ):
        mock_embed.return_value = [0.1] * 1024
        mock_store.return_value = {"id": "abc123", "created_at": "2026-01-01T00:00:00Z"}
        mock_ttl.return_value = None
        result = runner.invoke(app, ["store", "test memory", "--tag", "tag1"])

    assert result.exit_code == 0
    assert "abc123" in result.output
    mock_store.assert_called_once()
    call_kwargs = mock_store.call_args[1]
    assert call_kwargs["content"] == "test memory"
    assert call_kwargs["source"] == "cli"


def test_cli_store_with_ttl():
    """ogham store should set expires_at when profile has TTL"""
    from ogham.cli import app

    with (
        patch("ogham.embeddings.generate_embedding") as mock_embed,
        patch("ogham.database.store_memory") as mock_store,
        patch("ogham.database.get_profile_ttl") as mock_ttl,
    ):
        mock_embed.return_value = [0.1] * 1024
        mock_store.return_value = {"id": "abc123", "created_at": "2026-01-01T00:00:00Z"}
        mock_ttl.return_value = 90
        result = runner.invoke(app, ["store", "work note", "--profile", "work"])

    assert result.exit_code == 0
    call_kwargs = mock_store.call_args[1]
    assert call_kwargs["expires_at"] is not None
    assert "Expires" in result.output


def test_cli_serve():
    """ogham serve should call server main with no transport args"""
    from ogham.cli import app

    with patch("ogham.server.main") as mock_main:
        result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    mock_main.assert_called_once_with(transport=None, host=None, port=None)


def test_cli_serve_sse():
    """ogham serve --transport sse should pass args to server_main"""
    from ogham.cli import app

    with patch("ogham.server.main") as mock_main:
        result = runner.invoke(app, ["serve", "--transport", "sse", "--port", "9000"])

    assert result.exit_code == 0
    mock_main.assert_called_once_with(transport="sse", host=None, port=9000)
