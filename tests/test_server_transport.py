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


# --- streamable-http (ogham-mcp#71) --------------------------------------
#
# HTTP+SSE is the deprecated MCP transport and has no session-recovery path:
# once a session loses its initialized state, the client wedges on -32602
# forever. Streamable HTTP defines recovery (session gone -> HTTP 404 ->
# client re-initializes), so remote deployments belong on it.


def _env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")


def test_server_streamable_http_transport(monkeypatch):
    """streamable-http reaches mcp.run with host and port."""
    _env(monkeypatch)

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp") as mock_mcp,
    ):
        from ogham.server import main

        main(transport="streamable-http", host="0.0.0.0", port=9001)
        mock_mcp.run.assert_called_once_with(transport="streamable-http", host="0.0.0.0", port=9001)


def test_server_http_alias_transport(monkeypatch):
    """FastMCP's `http` alias is accepted and passed through unchanged."""
    _env(monkeypatch)

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp") as mock_mcp,
    ):
        from ogham.server import main

        main(transport="http", host="127.0.0.1", port=9002)
        mock_mcp.run.assert_called_once_with(transport="http", host="127.0.0.1", port=9002)


def test_sse_logs_deprecation_warning(monkeypatch, caplog):
    """Choosing SSE warns, and names the transport to use instead."""
    import logging

    _env(monkeypatch)

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp"),
        caplog.at_level(logging.WARNING, logger="ogham.server"),
    ):
        from ogham.server import main

        main(transport="sse", host="127.0.0.1", port=9003)

    assert any("streamable-http" in r.message for r in caplog.records), caplog.text


def test_streamable_http_does_not_warn(monkeypatch, caplog):
    """The recommended transport is not nagged about."""
    import logging

    _env(monkeypatch)

    with (
        patch("ogham.server.validate_startup"),
        patch("ogham.server.mcp"),
        caplog.at_level(logging.WARNING, logger="ogham.server"),
    ):
        from ogham.server import main

        main(transport="streamable-http", host="127.0.0.1", port=9004)

    assert not any("deprecated" in r.message.lower() for r in caplog.records)
