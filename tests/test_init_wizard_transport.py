"""Wizard transport selection (TBU-189).

`ogham init` could only configure stdio or SSE, and handed back a /sse URL.
Since HTTP+SSE is the deprecated MCP transport with no session-recovery path
(ogham-mcp#71), the wizard was steering every new multi-agent user onto it.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")


# --- endpoint per transport ----------------------------------------------


def test_streamable_http_entry_uses_mcp_endpoint():
    from ogham.init_wizard import _build_mcp_entry

    entry = _build_mcp_entry({}, "uvx", "streamable-http", "127.0.0.1", 8742)
    assert entry == {"url": "http://127.0.0.1:8742/mcp"}


def test_http_alias_entry_uses_mcp_endpoint():
    from ogham.init_wizard import _build_mcp_entry

    entry = _build_mcp_entry({}, "uvx", "http", "0.0.0.0", 9000)
    assert entry == {"url": "http://0.0.0.0:9000/mcp"}


def test_sse_entry_still_uses_sse_endpoint():
    """Legacy choice must keep working for existing deployments."""
    from ogham.init_wizard import _build_mcp_entry

    entry = _build_mcp_entry({}, "uvx", "sse", "127.0.0.1", 8742)
    assert entry == {"url": "http://127.0.0.1:8742/sse"}


def test_stdio_entry_is_not_a_url():
    from ogham.init_wizard import _build_mcp_entry

    entry = _build_mcp_entry({"FOO": "bar"}, "uvx", "stdio", "127.0.0.1", 8742)
    assert "url" not in entry
    assert "command" in entry


# --- the prompt ------------------------------------------------------------


def test_prompt_offers_streamable_http():
    """The recommended transport must be selectable."""
    from ogham.init_wizard import _prompt_transport

    with patch("ogham.init_wizard.Prompt.ask", side_effect=["2", "127.0.0.1", "8742"]):
        transport, host, port = _prompt_transport()

    assert transport == "streamable-http"
    assert (host, port) == ("127.0.0.1", 8742)


def test_prompt_still_offers_sse_as_legacy():
    from ogham.init_wizard import _prompt_transport

    with patch("ogham.init_wizard.Prompt.ask", side_effect=["3", "127.0.0.1", "8742"]):
        transport, _, _ = _prompt_transport()

    assert transport == "sse"


def test_prompt_default_is_stdio():
    from ogham.init_wizard import _prompt_transport

    with patch("ogham.init_wizard.Prompt.ask", side_effect=["1"]):
        transport, host, port = _prompt_transport()

    assert transport == "stdio"
    assert (host, port) == ("127.0.0.1", 8742)


def test_prompt_accepts_transport_names_not_just_numbers():
    from ogham.init_wizard import _prompt_transport

    with patch("ogham.init_wizard.Prompt.ask", side_effect=["streamable-http", "0.0.0.0", "9001"]):
        transport, host, port = _prompt_transport()

    assert transport == "streamable-http"
    assert (host, port) == ("0.0.0.0", 9001)


def test_prompt_choices_include_every_supported_transport():
    """Whatever the wizard offers must be what the server accepts."""
    from ogham.config import Settings
    from ogham.init_wizard import _prompt_transport

    captured = {}

    def fake_ask(*args, **kwargs):
        captured.setdefault("choices", kwargs.get("choices"))
        return "1"

    with patch("ogham.init_wizard.Prompt.ask", side_effect=fake_ask):
        _prompt_transport()

    named = [c for c in captured["choices"] if not c.isdigit()]
    assert "streamable-http" in named
    # Every named choice must survive config validation.
    for name in named:
        Settings(server_transport=name)
