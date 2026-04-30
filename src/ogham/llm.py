"""LLM provider layer for the T1.4 wiki-compile path.

Direct httpx, no library. Mirrors the embeddings.py provider-switch
pattern. Decision + rationale in stream-d-llm-abstraction.md:
30-line direct path beats 15-25 MB of any-llm/LiteLLM transitive
deps, and matches our "Sovereign" user story.

Two wire formats:
  * OpenAI-compat (openai, groq, mistral, gemini, ollama, vllm,
    openrouter) -- POST /v1/chat/completions, Bearer auth, choices[0].
  * Anthropic native -- POST /v1/messages. Needed because their
    OpenAI-compat shim silently drops tool_choice/tool_use. If
    compile_wiki ever calls back into Ogham MCP via tool-use we
    need the native branch anyway.

Auth keys are read from env at call time (not stored in settings).
That keeps the module importable in environments that aren't going
to synthesize (benchmarks, tests) without pydantic complaining.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ogham.config import settings

_OPENAI_COMPAT = {
    # provider -> (base_url, env var for API key or None for no-auth)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Local / self-hosted -- no auth.
    "ollama": (None, None),  # base filled at call time from settings.ollama_url
    "vllm": (None, None),  # base filled from OGHAM_VLLM_URL env
}


def synthesize(
    prompt: str,
    *,
    provider: str,
    model: str,
    system: str | None = None,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Synthesize a response from the configured LLM provider.

    Sync on purpose -- the recompute pipeline runs inside the lifecycle
    executor's thread pool, and sync httpx is simpler to reason about
    than wrapping async calls in run_in_executor just to get back to
    a thread.

    `response_format` is forwarded to OpenAI-compatible providers (e.g.
    `{"type": "json_object"}`); Anthropic ignores it -- callers needing
    JSON from Anthropic must rely on the system-prompt schema hint.
    """
    if provider == "anthropic":
        return _anthropic(prompt, model, system, max_tokens, timeout)
    if provider not in _OPENAI_COMPAT:
        raise ValueError(
            f"unknown LLM provider {provider!r}; expected one of "
            f"{sorted([*_OPENAI_COMPAT, 'anthropic'])}"
        )
    return _openai_compat(provider, prompt, model, system, max_tokens, timeout, response_format)


def _openai_compat(
    provider: str,
    prompt: str,
    model: str,
    system: str | None,
    max_tokens: int,
    timeout: float,
    response_format: dict[str, Any] | None = None,
) -> str:
    base, env_key = _OPENAI_COMPAT[provider]
    if provider == "ollama":
        base = f"{settings.ollama_url.rstrip('/')}/v1"
    elif provider == "vllm":
        vllm_url = os.environ.get("OGHAM_VLLM_URL", "http://localhost:8000")
        base = f"{vllm_url.rstrip('/')}/v1"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if env_key:
        # Fall back to settings (loaded from .env / ~/.ogham/config.env via
        # pydantic) when the bare env var isn't exported -- which is the
        # normal case when the MCP server is launched by a client that
        # doesn't propagate the user's shell env.
        key = os.environ.get(env_key) or getattr(settings, env_key.lower(), None)
        if not key:
            raise RuntimeError(f"{provider} requires {env_key} in the environment -- not set")
        headers["Authorization"] = f"Bearer {key}"

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if provider == "ollama":
        # Some Ollama models silently ignore max_tokens and use num_predict
        # instead. Set both -- max_tokens for the OpenAI-compat shim,
        # options.num_predict for the underlying runtime. Belt-and-braces.
        body["options"] = {"num_predict": max_tokens}
        # Ollama maps response_format to the runtime "format" option; lift
        # to top-level "format=json" so older Ollama versions handle it.
        if response_format and response_format.get("type") == "json_object":
            body["format"] = "json"

    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def _anthropic(prompt: str, model: str, system: str | None, max_tokens: int, timeout: float) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", None)
    if not key:
        raise RuntimeError("anthropic requires ANTHROPIC_API_KEY in the environment -- not set")

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        # Anthropic puts system at the top level, NOT as a message with
        # role=system. Submitting role=system returns HTTP 400.
        body["system"] = system

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    # Response shape: {"content": [{"type": "text", "text": "..."}]}
    return data["content"][0]["text"]


def synthesize_json(
    prompt: str,
    *,
    provider: str,
    model: str,
    json_schema: dict[str, Any],
    system: str | None = None,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call synthesize() with JSON-mode hint, parse the result.

    Adds a "respond with valid JSON matching this schema" instruction to
    the system prompt so the model emits structured output instead of
    prose. Strips markdown code fences if the model wraps its JSON in
    ```json ... ```. Validates that every field in the schema's
    ``required`` list is present in the parsed dict.

    Returns the parsed dict. Raises ValueError if the model returns
    invalid JSON or fails the schema's required-fields check.

    Used by recompute.py to generate body + tldr_short + tldr_one_line
    in a single LLM call (v0.13 progressive recall).
    """
    schema_hint = (
        "Respond with VALID JSON matching this schema (no prose, no markdown "
        "fences):\n" + json.dumps(json_schema, indent=2)
    )
    full_system = f"{system}\n\n{schema_hint}" if system else schema_hint

    raw = synthesize(
        prompt=prompt,
        provider=provider,
        model=model,
        system=full_system,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
    )

    # Strip markdown fences the model may have added despite instructions.
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Three-attempt parse chain. Each step is more lenient than the last;
    # we only escalate after the cheaper one fails so clean responses are
    # not silently mutated by the repair pass.
    #
    #   1. json.loads(strict)        -- catches structural errors verbatim
    #   2. JSONDecoder(strict=False) -- tolerates bare control chars
    #   3. json_repair.repair_json   -- heuristic recovery (unescaped
    #      quotes, missing commas, truncated outputs). Last resort:
    #      handles long-body Gemini malformations that #2 can't reach.
    #
    # Required-field validation runs after parsing regardless of which
    # attempt succeeded, so a repaired-but-incomplete response still
    # surfaces as a ValueError downstream.
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = json.JSONDecoder(strict=False).decode(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                repaired = repair_json(raw, return_objects=False)
                parsed = json.loads(repaired)
            except (json.JSONDecodeError, ValueError) as e3:
                title = json_schema.get("title", "<unnamed>")
                raise ValueError(
                    f"LLM returned invalid JSON for json_schema={title}: {e3}\n"
                    f"Raw response (first 500 chars): {raw[:500]}"
                ) from e3

    # json-repair sometimes wraps a malformed object in a single-item list
    # when it can't determine the right top-level shape. If that's what we
    # got AND the wrapped item is a dict with the schema's required fields,
    # unwrap it. Otherwise treat as a real shape mismatch.
    required = json_schema.get("required", [])
    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], dict)
        and all(f in parsed[0] for f in required)
    ):
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM returned JSON that is not an object (got {type(parsed).__name__}): {raw[:200]}"
        )

    missing = [f for f in required if f not in parsed]
    if missing:
        raise ValueError(
            f"LLM returned JSON missing required fields {missing}. Got keys: {list(parsed.keys())}"
        )

    return parsed
