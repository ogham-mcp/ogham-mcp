"""Phase 3 tests for src/ogham/llm.py — direct-httpx provider layer.

No real network. Each test patches httpx.Client to capture the outgoing
request and feed back a canned response. This locks in the wire-format
contract for each provider: endpoint URL, headers, body shape, and how
we extract the response text.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(*, status: int = 200, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_body)
    return resp


def _patch_post(response):
    """Patch httpx.Client so .post returns the given response.

    Yields the mock client instance so tests can assert on captured calls.
    """
    client_instance = MagicMock()
    client_instance.post = MagicMock(return_value=response)
    client_instance.__enter__ = MagicMock(return_value=client_instance)
    client_instance.__exit__ = MagicMock(return_value=False)
    return patch("httpx.Client", return_value=client_instance), client_instance


# ---------- OpenAI-compat providers (openai, groq, mistral, gemini, ollama, vllm) -----


def test_synthesize_openai_posts_to_chat_completions(monkeypatch):
    """OpenAI path: POST /v1/chat/completions with Bearer auth and model+messages."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "hello from openai"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        out = synthesize(prompt="summarize this", provider="openai", model="gpt-4o-mini")

    assert out == "hello from openai"
    assert client.post.call_count == 1
    call = client.post.call_args
    assert "api.openai.com" in call[0][0]
    assert call[0][0].endswith("/chat/completions")
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-test-abc"
    body = call.kwargs["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "summarize this"}]


def test_synthesize_openai_with_system_prompt(monkeypatch):
    """System prompt goes in front of user message."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "ok"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(
            prompt="user part",
            provider="openai",
            model="gpt-4o",
            system="you are a wiki compiler",
        )

    body = client.post.call_args.kwargs["json"]
    assert body["messages"] == [
        {"role": "system", "content": "you are a wiki compiler"},
        {"role": "user", "content": "user part"},
    ]


def test_synthesize_ollama_no_auth_header_and_remaps_max_tokens(monkeypatch):
    """Ollama: no API key, uses settings.ollama_url, max_tokens -> num_predict."""
    resp = _mock_response(json_body={"choices": [{"message": {"content": "local llm out"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(
            prompt="hi",
            provider="ollama",
            model="llama3.2",
            max_tokens=256,
        )

    call = client.post.call_args
    # Ollama OpenAI-compat lives at {ollama_url}/v1/chat/completions
    assert "/v1/chat/completions" in call[0][0]
    # No Authorization header -- local endpoint.
    assert "Authorization" not in call.kwargs.get("headers", {})
    body = call.kwargs["json"]
    # Ollama-specific: some models ignore max_tokens, the num_predict override
    # lands in options so we never silently truncate.
    assert body.get("options", {}).get("num_predict") == 256


def test_synthesize_groq_uses_groq_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "g"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(prompt="x", provider="groq", model="llama-3.3-70b-versatile")

    assert "api.groq.com" in client.post.call_args[0][0]


def test_synthesize_gemini_uses_openai_compat_endpoint(monkeypatch):
    """Gemini has a v1beta OpenAI-compat shim we rely on."""
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "g"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(prompt="x", provider="gemini", model="gemini-2.0-flash")

    url = client.post.call_args[0][0]
    assert "generativelanguage.googleapis.com" in url
    assert "/v1beta/openai/" in url


# ---------- Anthropic native branch ----------


def test_synthesize_anthropic_posts_to_v1_messages(monkeypatch):
    """Anthropic is the one non-OpenAI-compat branch -- native /v1/messages.

    The OpenAI-compat shim at /v1/chat/completions silently drops tools +
    tool_choice, so compile_wiki (which may need tool-calls in a later
    phase) always routes through native.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    resp = _mock_response(json_body={"content": [{"type": "text", "text": "claude says hi"}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        out = synthesize(
            prompt="hello",
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=1024,
        )

    assert out == "claude says hi"
    call = client.post.call_args
    assert call[0][0] == "https://api.anthropic.com/v1/messages"
    headers = call.kwargs["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    body = call.kwargs["json"]
    assert body["model"] == "claude-sonnet-4-6"
    assert body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_synthesize_anthropic_system_is_top_level_not_message(monkeypatch):
    """Anthropic /v1/messages puts system prompt outside the messages array."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    resp = _mock_response(json_body={"content": [{"type": "text", "text": "ok"}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(
            prompt="u",
            provider="anthropic",
            model="claude-haiku-4-5",
            system="you compile wikis",
        )

    body = client.post.call_args.kwargs["json"]
    assert body["system"] == "you compile wikis"
    # System must NOT leak into messages -- Anthropic rejects "role: system".
    assert all(m["role"] != "system" for m in body["messages"])


# ---------- error paths ----------


def test_synthesize_unknown_provider_raises():
    from ogham.llm import synthesize

    with pytest.raises(ValueError, match="unknown.*provider"):
        synthesize(prompt="x", provider="claude-desktop-or-something", model="m")


def test_synthesize_missing_api_key_raises(monkeypatch):
    """Provider that requires a key but the env var is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from ogham.llm import synthesize

    with pytest.raises((RuntimeError, ValueError)):
        synthesize(prompt="x", provider="openai", model="gpt-4o-mini")
