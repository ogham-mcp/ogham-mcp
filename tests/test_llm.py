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
    # Pydantic settings loads from .env / ~/.ogham/config.env -- patch the
    # attribute too so the fallback in synthesize() can't pick up a real key.
    from ogham import llm as _llm

    monkeypatch.setattr(_llm.settings, "openai_api_key", None, raising=False)

    from ogham.llm import synthesize

    with pytest.raises((RuntimeError, ValueError)):
        synthesize(prompt="x", provider="openai", model="gpt-4o-mini")


# ---------- response_format plumbing (v0.14 fix from Hotfix C) -----------------


def test_synthesize_passes_response_format_to_openai_compat(monkeypatch):
    """response_format is forwarded into the OpenAI-compat request body."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(
            prompt="x",
            provider="openai",
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
        )

    body = client.post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}


def test_synthesize_ollama_lifts_format_json_top_level(monkeypatch):
    """Ollama path gets `format=json` at the top level when response_format is json_object.

    Older Ollama versions don't honour OpenAI-compat response_format and need
    the top-level `format` instead. Belt-and-braces -- both fields are set.
    """
    resp = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(
            prompt="x",
            provider="ollama",
            model="llama3.2",
            response_format={"type": "json_object"},
        )

    body = client.post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["format"] == "json"


def test_synthesize_no_response_format_omits_field(monkeypatch):
    """When response_format is None it's not sent in the body."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "ok"}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize

    with patcher:
        synthesize(prompt="x", provider="openai", model="gpt-4o-mini")

    body = client.post.call_args.kwargs["json"]
    assert "response_format" not in body
    assert "format" not in body


def test_synthesize_json_auto_sets_response_format(monkeypatch):
    """synthesize_json sets response_format={'type':'json_object'} automatically."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": '{"body": "x"}'}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize_json

    with patcher:
        synthesize_json(
            prompt="x",
            provider="openai",
            model="gpt-4o-mini",
            json_schema={"type": "object", "required": ["body"]},
        )

    body = client.post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}


# ---------- synthesize_json: structured output for v0.13 progressive recall ----


def test_synthesize_json_returns_parsed_dict(monkeypatch):
    """synthesize_json wraps synthesize and parses the JSON response."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_response = (
        '{"body": "full body here", "tldr_short": "paragraph", "tldr_one_line": "one line"}'
    )
    resp = _mock_response(json_body={"choices": [{"message": {"content": fake_response}}]})
    patcher, _client = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {
        "type": "object",
        "required": ["body", "tldr_short", "tldr_one_line"],
        "properties": {
            "body": {"type": "string"},
            "tldr_short": {"type": "string"},
            "tldr_one_line": {"type": "string"},
        },
    }

    with patcher:
        result = synthesize_json(
            prompt="summarize this",
            provider="openai",
            model="gpt-4o-mini",
            json_schema=schema,
        )

    assert result == {
        "body": "full body here",
        "tldr_short": "paragraph",
        "tldr_one_line": "one line",
    }


def test_synthesize_json_strips_markdown_fences(monkeypatch):
    """Models often wrap JSON in ```json ... ``` despite the instruction not to."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fenced = '```json\n{"body": "x", "tldr_short": "y", "tldr_one_line": "z"}\n```'
    resp = _mock_response(json_body={"choices": [{"message": {"content": fenced}}]})
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {
        "type": "object",
        "required": ["body", "tldr_short", "tldr_one_line"],
    }
    with patcher:
        result = synthesize_json(
            prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema
        )

    assert result["body"] == "x"
    assert result["tldr_short"] == "y"
    assert result["tldr_one_line"] == "z"


def test_synthesize_json_strips_bare_fence_without_lang(monkeypatch):
    """Some providers emit ``` ... ``` without a language tag."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fenced = '```\n{"body": "a"}\n```'
    resp = _mock_response(json_body={"choices": [{"message": {"content": fenced}}]})
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {"type": "object", "required": ["body"]}
    with patcher:
        result = synthesize_json(
            prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema
        )

    assert result["body"] == "a"


def test_synthesize_json_tolerates_bare_control_chars_in_string(monkeypatch):
    """Bare \\n/\\t inside string values is recovered via strict=False fallback.

    Gemini occasionally emits raw control chars in long markdown body fields
    despite response_format=json_object. Strict parse fails; non-strict fallback
    recovers the value. Surfaced 2026-04-29 during Hotfix C bulk recompile
    (project:ogham + sql-migration topics).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Bare newline (0x0A) inside the body string. Strict JSON forbids this;
    # strict=False accepts and preserves it.
    bad = '{\n  "body": "line one\nline two",\n  "tldr_short": "x",\n  "tldr_one_line": "y"\n}'
    resp = _mock_response(json_body={"choices": [{"message": {"content": bad}}]})
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {
        "type": "object",
        "required": ["body", "tldr_short", "tldr_one_line"],
    }
    with patcher:
        result = synthesize_json(
            prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema
        )

    assert result["body"] == "line one\nline two"
    assert result["tldr_short"] == "x"
    assert result["tldr_one_line"] == "y"


def test_synthesize_json_missing_required_field_raises(monkeypatch):
    """If a required field is absent the helper raises ValueError, not silently returns partial."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": '{"body": "x"}'}}]})
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {
        "type": "object",
        "required": ["body", "tldr_short", "tldr_one_line"],
    }
    with patcher:
        with pytest.raises(ValueError, match="missing required fields"):
            synthesize_json(prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema)


def test_synthesize_json_invalid_json_raises(monkeypatch):
    """Garbage in -> ValueError out."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": "not json at all"}}]})
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {"type": "object", "required": ["body"]}
    with patcher:
        with pytest.raises(ValueError, match="invalid JSON"):
            synthesize_json(prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema)


def test_synthesize_json_layered_system_prompt(monkeypatch):
    """When the caller passes a system prompt, schema hint is appended (not replaced)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(json_body={"choices": [{"message": {"content": '{"body": "ok"}'}}]})
    patcher, client = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {"type": "object", "required": ["body"], "title": "test_layered"}
    with patcher:
        synthesize_json(
            prompt="x",
            provider="openai",
            model="gpt-4o-mini",
            json_schema=schema,
            system="you are a wiki compiler",
        )

    body = client.post.call_args.kwargs["json"]
    system_msg = body["messages"][0]
    assert system_msg["role"] == "system"
    # Original system prompt preserved.
    assert "you are a wiki compiler" in system_msg["content"]
    # Schema hint appended.
    assert "VALID JSON" in system_msg["content"]
    assert "test_layered" in system_msg["content"]


def test_synthesize_json_non_object_root_raises(monkeypatch):
    """JSON scalars/arrays at root are rejected: schema requires an object."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    resp = _mock_response(
        json_body={"choices": [{"message": {"content": '["just", "an", "array"]'}}]}
    )
    patcher, _ = _patch_post(resp)

    from ogham.llm import synthesize_json

    schema = {"type": "object", "required": ["body"]}
    with patcher:
        with pytest.raises(ValueError, match="not an object"):
            synthesize_json(prompt="x", provider="openai", model="gpt-4o-mini", json_schema=schema)
