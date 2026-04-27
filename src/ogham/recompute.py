"""Topic-summary recompute orchestrator (T1.4 Phase 4).

Stitches topic_summaries.py + llm.py + embeddings.py into a single
callable that:
  1. Fetches source memories for the topic (by profile + tag).
  2. Computes the current source_hash.
  3. Short-circuits if the hash matches the stored fresh row (cheap
     path -- no LLM call, no embedding, no DB write).
  4. Otherwise renders a compile prompt, calls synthesize, embeds the
     result, and upserts atomically.

Letta #3270 guard: the upsert only runs after a successful synthesize.
If the LLM call fails, the exception propagates -- the previous fresh
row is untouched because we never got to the upsert step. The executor
wrapping this call logs the failure and moves on.
"""

from __future__ import annotations

import logging
from typing import Any

from ogham.config import settings
from ogham.data.loader import get_wiki_compile
from ogham.database import get_backend
from ogham.embeddings import generate_embedding
from ogham.llm import synthesize
from ogham.topic_summaries import compute_source_hash, get_summary_by_topic, upsert_summary

logger = logging.getLogger(__name__)

# Token-budget warning threshold. We don't block -- Ogham is infrastructure,
# we don't pay the LLM bill -- but oversized prompts usually mean the
# operator hasn't tuned max_sources or a single tag has gone unbounded.
# 1 token ~= 4 chars for English; threshold is estimated-tokens.
_PROMPT_TOKEN_WARN = 10_000

# Max composed output length in characters before we emit a warning. Rows
# are still stored (TOAST handles arbitrary sizes) but runaway outputs
# usually signal a compile prompt that didn't anchor the LLM tightly
# enough, or a source memory that was itself massive and got echoed.
_OUTPUT_LEN_WARN_CHARS = 25_000


def recompute_topic_summary(
    profile: str,
    topic_key: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Recompute the topic summary for (profile, topic_key) if sources changed.

    Returns a small dict describing what happened so the executor can
    log useful telemetry without a second DB round-trip:
      * {"action": "no_sources"} -- no memories with this tag
      * {"action": "skipped", "reason": "source_hash_match", "summary_id": ...}
      * {"action": "recomputed", "summary_id": ..., "source_count": ...}

    Caller passes provider + model explicitly, or omits them to fall back
    to settings.llm_provider + settings.llm_model. Phase 6 hooks will use
    the settings path so ingest paths don't need to know the LLM config.
    """
    backend = get_backend()

    # 1. Pull source memory ids for this topic via the migration 031
    #    function (works on both PostgresBackend and SupabaseBackend).
    source_ids = backend.wiki_recompute_get_source_ids(profile, topic_key)
    if not source_ids:
        return {"action": "no_sources", "profile": profile, "topic_key": topic_key}

    # 2. Hash check -- short-circuit if nothing has moved.
    new_hash = compute_source_hash(source_ids)
    existing = get_summary_by_topic(profile, topic_key)
    if existing and existing.get("status") == "fresh":
        # Stored hash may come back as bytes (psycopg) or as a hex
        # string `\xDEADBEEF` (PostgREST). Normalise either to bytes.
        stored_raw = existing.get("source_hash")
        if isinstance(stored_raw, str):
            stored_bytes = bytes.fromhex(stored_raw.removeprefix("\\x"))
        elif isinstance(stored_raw, (bytes, bytearray, memoryview)):
            stored_bytes = bytes(stored_raw)
        else:
            stored_bytes = b""
        if stored_bytes == new_hash:
            return {
                "action": "skipped",
                "reason": "source_hash_match",
                "summary_id": str(existing["id"]),
            }

    # 3. Fetch source content for the prompt.
    content_rows = backend.wiki_recompute_get_source_content(source_ids)
    prompt = _render_compile_prompt(topic_key, content_rows or [])

    # 3b. Token-budget signal. Char/4 is a rough English-token estimate --
    # close enough for a warn threshold without pulling in tiktoken.
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > _PROMPT_TOKEN_WARN:
        logger.warning(
            "compile prompt is ~%d tokens (threshold %d) for %s/%s -- "
            "this is a bit excessive, consider narrowing the source set",
            estimated_tokens,
            _PROMPT_TOKEN_WARN,
            profile,
            topic_key,
        )

    # 4. Synthesize. Failures propagate -- previous fresh row stays intact.
    used_provider = provider or getattr(settings, "llm_provider", "ollama")
    used_model = model or getattr(settings, "llm_model", "llama3.2")
    composed = synthesize(
        prompt=prompt,
        provider=used_provider,
        model=used_model,
        system=_compile_system_prompt(),
    )

    # 4b. Output sanity. Empty = hard fail (better to keep the old row
    # than cache a blank page). Excessive length = soft warn.
    _validate_synthesize_output(composed, profile, topic_key)

    # 5. Embed the composed summary + atomic upsert.
    embedding = generate_embedding(composed)
    summary = upsert_summary(
        profile=profile,
        topic_key=topic_key,
        content=composed,
        embedding=embedding,
        source_memory_ids=source_ids,
        model_used=f"{used_provider}/{used_model}",
    )
    return {
        "action": "recomputed",
        "summary_id": str(summary["id"]),
        "source_count": len(source_ids),
    }


def _compile_system_prompt() -> str:
    """Return the system prompt used for compile_wiki synthesis.

    Exposed as a named function so tests can assert the anti-injection
    instruction is present and prompt-tuning changes can't drop it by
    accident. Loaded from `wiki_compile.system_prompt` in the active
    locale's YAML (src/ogham/data/languages/<lang>.yaml), falling back
    to English when a locale doesn't override.
    """
    locale = getattr(settings, "locale", "en")
    return get_wiki_compile("system_prompt", lang=locale)


def _wrap_source(source_id: str, content: str) -> str:
    """Wrap a source memory in an XML-ish tag, escaping close-tags in the
    body so adversarial content can't terminate the wrapper early.

    The escape rewrites ``</source>`` inside content to ``</source_ESC>``.
    Operators who inspect the prompt will see the marker; the LLM is
    instructed (via the system prompt) to treat everything inside the
    outer wrapper as data regardless.
    """
    safe = content.replace("</source>", "</source_ESC>")
    return f'<source id="{source_id}">\n{safe}\n</source>'


def _render_compile_prompt(topic_key: str, source_rows: list[dict[str, Any]]) -> str:
    """Render the compile-wiki prompt from source memories.

    Template loaded from `wiki_compile.prompt_template` in the active
    locale's YAML, with `{topic_key}` and `{sources}` placeholders.
    """
    locale = getattr(settings, "locale", "en")
    template = get_wiki_compile("prompt_template", lang=locale)
    sources = "\n\n".join(_wrap_source(row["id"], row["content"]) for row in source_rows)
    return template.format(topic_key=topic_key, sources=sources)


def _validate_synthesize_output(composed: str, profile: str, topic_key: str) -> None:
    """Raise on empty/whitespace output; log warn on excessive length.

    Empty is hard-fail because writing a blank cache row is worse than
    keeping the previous fresh row via the Letta #3270 guard path.
    Length is soft because TOAST handles arbitrary size on disk -- the
    operator signal just matters.
    """
    if not composed or not composed.strip():
        raise ValueError(
            f"synthesize returned empty output for {profile}/{topic_key} "
            "-- refusing to cache a blank summary"
        )
    if len(composed) > _OUTPUT_LEN_WARN_CHARS:
        logger.warning(
            "composed summary is %d chars (threshold %d) for %s/%s -- "
            "this is excessive, compile prompt may be too permissive",
            len(composed),
            _OUTPUT_LEN_WARN_CHARS,
            profile,
            topic_key,
        )
