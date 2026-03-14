"""Tests for server transport branching (stdio vs SSE)."""

from unittest.mock import patch


def test_server_stdio_default(monkeypatch):
    """Default transport calls mcp.run() with no args."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp") as mock_mcp,
    ):
        from ogham.server import main

        main()
        mock_mcp.run.assert_called_once_with()


def test_server_sse_transport(monkeypatch):
    """SSE transport calls mcp.run() with transport, host, port."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp") as mock_mcp,
    ):
        from ogham.server import main

        main(transport="sse", host="0.0.0.0", port=9000)
        mock_mcp.run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9000)


def test_server_cli_overrides_env(monkeypatch):
    """CLI args take precedence over OGHAM_ env vars."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OGHAM_TRANSPORT", "stdio")
    monkeypatch.setenv("OGHAM_PORT", "8742")

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp") as mock_mcp,
    ):
        from ogham.server import main

        main(transport="sse", host="127.0.0.1", port=9000)
        mock_mcp.run.assert_called_once_with(transport="sse", host="127.0.0.1", port=9000)
