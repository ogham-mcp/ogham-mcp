"""Embedding generation plus optional usage sidecar capture.

Callers that only need vectors use `generate_embedding()` /
`generate_embeddings_batch()` exactly as before.

Callers that also want provider usage pass a mutable `usage_out` dict.
When provided, the function mutates it in place with best-effort fields
such as `model`, `input_tokens`, and `cache_hit`.
"""

import hashlib
import logging
import math
import os
import re
from collections.abc import Callable
from typing import Any, cast

from ogham.config import settings
from ogham.embedding_cache import EmbeddingCache
from ogham.retry import with_retry

logger = logging.getLogger(__name__)

_cache: EmbeddingCache | None = None


def _get_cache() -> EmbeddingCache:
    """Create the embedding cache on demand to avoid import-time settings validation."""
    global _cache
    if _cache is None:
        _cache = EmbeddingCache(
            cache_dir=settings.embedding_cache_dir,
            max_size=settings.embedding_cache_max_size,
        )
    return _cache


EmbeddingUsage = dict[str, Any]


def get_cache_stats() -> dict:
    """Return cache statistics."""
    return _get_cache().stats()


def _cache_key(text: str) -> str:
    """Build a cache key scoped to the current provider, model, and dimension.

    Switching providers, models, or dimensions automatically invalidates cached
    vectors because the key prefix changes.
    """
    prefix = f"{_current_embedding_model()}:{settings.embedding_dim}:"
    return hashlib.sha256((prefix + text).encode()).hexdigest()


def _current_embedding_model(provider: str | None = None) -> str:
    """Return the normalized provider:model identifier used in audit rows."""
    provider = provider or settings.embedding_provider
    match provider:
        case "ollama":
            model = settings.ollama_embed_model
        case "openai":
            model = "text-embedding-3-small"
        case "mistral":
            model = settings.mistral_embed_model
        case "voyage":
            model = settings.voyage_embed_model
        case "gemini":
            model = settings.gemini_embed_model
        case "onnx":
            model = "local"
        case _:
            model = "unknown"
    return f"{provider}:{model}"


def _cached_embedding_usage() -> EmbeddingUsage:
    """Return the synthetic usage payload for a cache hit."""
    return {
        "model": _current_embedding_model(),
        "input_tokens": 0,
        "cache_hit": True,
    }


def _usage_dict(
    *,
    model: str,
    input_tokens: int | None = None,
    cache_hit: bool | None = None,
) -> EmbeddingUsage:
    """Build a compact usage payload, skipping unknown fields."""
    usage: EmbeddingUsage = {"model": model}
    if input_tokens is not None:
        usage["input_tokens"] = int(input_tokens)
    if cache_hit is not None:
        usage["cache_hit"] = cache_hit
    return usage


def _set_usage_out(usage_out: EmbeddingUsage | None, usage: EmbeddingUsage | None) -> None:
    """Replace the caller-provided usage sidecar in place when present."""
    if usage_out is None or usage is None:
        return
    usage_out.clear()
    usage_out.update(usage)


def _merge_usage(
    total: EmbeddingUsage | None,
    current: EmbeddingUsage | None,
) -> EmbeddingUsage | None:
    """Accumulate usage across multiple provider calls in one logical request."""
    if current is None:
        return total
    if total is None:
        return dict(current)

    merged: EmbeddingUsage = dict(total)
    if not merged.get("model"):
        merged["model"] = current.get("model", "")
    if "input_tokens" in current:
        merged["input_tokens"] = merged.get("input_tokens", 0) + current["input_tokens"]
    merged["cache_hit"] = merged.get("cache_hit", False) and current.get("cache_hit", False)
    return merged


def _model_only_usage(provider: str) -> EmbeddingUsage:
    """Return model provenance for providers that do not expose token usage."""
    return _usage_dict(model=_current_embedding_model(provider))


def generate_embedding(
    text: str,
    usage_out: EmbeddingUsage | None = None,
) -> list[float]:
    """Generate one embedding vector, optionally populating `usage_out`.

    Uses persistent SQLite cache keyed by provider + model + dimension +
    SHA256 of text to avoid re-embedding identical content. Switching
    providers, models, or dimensions automatically invalidates cached vectors.

    If `usage_out` is provided, it is mutated in place with best-effort usage
    metadata for this request. Cache hits report `input_tokens=0`.
    """
    cache_key = _cache_key(text)

    cache = _get_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Embedding cache hit for text hash %s", cache_key[:8])
        _set_usage_out(usage_out, _cached_embedding_usage())
        return cached

    if usage_out is None:
        embedding = _generate_uncached(text)
    else:
        embedding = _generate_uncached(text, usage_out=usage_out)
    cache.put(cache_key, embedding)
    return embedding


@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
def _generate_uncached(
    text: str,
    usage_out: EmbeddingUsage | None = None,
) -> list[float]:
    """Generate one embedding without cache lookup, forwarding `usage_out`."""
    provider = settings.embedding_provider

    match provider:
        case "ollama":
            return _embed_ollama(text, usage_out=usage_out)
        case "openai":
            return _embed_openai(text, usage_out=usage_out)
        case "mistral":
            return _embed_mistral(text, usage_out=usage_out)
        case "voyage":
            return _embed_voyage(text, usage_out=usage_out)
        case "gemini":
            return _embed_gemini(text, usage_out=usage_out)
        case "onnx":
            return _embed_onnx(text, usage_out=usage_out)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")


def _embed_onnx(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    from ogham.onnx_embedder import encode

    result = encode(text, settings.onnx_model_path or None)
    embedding = result.dense
    _validate_dim(embedding)
    _set_usage_out(usage_out, _model_only_usage("onnx"))
    return embedding


_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        _ollama_client = ollama.Client(host=settings.ollama_url, timeout=settings.ollama_timeout)
    return _ollama_client


def _embed_ollama(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    client = _get_ollama_client()
    kwargs: dict = {"model": settings.ollama_embed_model, "input": text}
    if settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim
    response = client.embed(**kwargs)
    embedding = response["embeddings"][0]
    _validate_dim(embedding)
    _set_usage_out(usage_out, _model_only_usage("ollama"))
    return embedding


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _extract_openai_usage(response) -> EmbeddingUsage:
    """Extract best-effort token usage from an OpenAI embeddings response."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "total_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)
    return _usage_dict(model=_current_embedding_model("openai"), input_tokens=input_tokens)


def _embed_openai(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required when embedding_provider=openai")

    client = _get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=settings.embedding_dim,
    )
    embedding = response.data[0].embedding
    _validate_dim(embedding)
    _set_usage_out(usage_out, _extract_openai_usage(response))
    return embedding


_mistral_client = None


def _get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        from mistralai import Mistral

        _mistral_client = Mistral(api_key=settings.mistral_api_key)
    return _mistral_client


def _embed_mistral(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY required when embedding_provider=mistral")
    client = _get_mistral_client()
    response = client.embeddings.create(
        model=settings.mistral_embed_model,
        inputs=[text],
    )
    embedding = response.data[0].embedding
    _validate_dim(embedding)
    _set_usage_out(usage_out, _model_only_usage("mistral"))
    return embedding


_voyage_client = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        _voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    return _voyage_client


def _extract_voyage_usage(response) -> EmbeddingUsage:
    """Extract best-effort token usage from a Voyage embeddings response."""
    return _usage_dict(
        model=_current_embedding_model("voyage"),
        input_tokens=getattr(response, "total_tokens", None),
    )


def _embed_voyage(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY required when embedding_provider=voyage")
    client = _get_voyage_client()
    response = client.embed(
        texts=[text],
        model=settings.voyage_embed_model,
        output_dimension=settings.embedding_dim,
    )
    embedding = response.embeddings[0]
    _validate_dim(embedding)
    _set_usage_out(usage_out, _extract_voyage_usage(response))
    return embedding


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # pyright: ignore[reportAttributeAccessIssue]

        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


_EMBED_MAX_CHARS = 20000  # ~6-7K tokens at typical 3-4 chars/token, safe for 8191 token limit


def _extract_gemini_usage(response) -> EmbeddingUsage:
    """Extract best-effort token usage from a Gemini embeddings response."""
    metadata = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    prompt_tokens = getattr(metadata, "prompt_token_count", None)
    if prompt_tokens is None and isinstance(metadata, dict):
        prompt_tokens = metadata.get("prompt_token_count")
    return _usage_dict(model=_current_embedding_model("gemini"), input_tokens=prompt_tokens)


def _embed_gemini(text: str, usage_out: EmbeddingUsage | None = None) -> list[float]:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY required when embedding_provider=gemini")
    client = _get_gemini_client()
    response = client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=text,
        config={"output_dimensionality": settings.embedding_dim},
    )
    embedding = response.embeddings[0].values
    _validate_dim(embedding)
    if settings.embedding_dim < 3072 and not _gemini_model_pre_normalizes():
        embedding = _l2_normalize(embedding)
    _set_usage_out(usage_out, _extract_gemini_usage(response))
    return embedding


def _gemini_model_pre_normalizes() -> bool:
    """`gemini-embedding-2` (GA) returns pre-normalized vectors at every
    output dim -- verified empirically 2026-07-02 at 512/768/1536/3072,
    ||v|| within 2e-7 of 1.0. Older / preview aliases are treated as
    not-pre-normalized so we keep the defensive client-side normalize.
    """
    return settings.gemini_embed_model == "gemini-embedding-2"


def _validate_dim(embedding: list[float]) -> None:
    if len(embedding) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.embedding_dim}"
        )


def _l2_normalize(embedding: list[float]) -> list[float]:
    """Rescale `embedding` to unit length. Zero vectors pass through unchanged
    (normalizing would divide by zero).

    Historical note: pre-GA Gemini embedding models only pre-normalized at
    the model's native 3072 dim; sub-3072 outputs needed client-side
    normalize or cosine similarity became magnitude-weighted. `gemini-embedding-2`
    GA changed this -- it now pre-normalizes at all output dims
    (verified 2026-07-02 across 512/768/1536/3072). See `_gemini_model_pre_normalizes`
    for the gate; this function stays as the defensive path for older aliases and
    for providers other than Gemini.
    """
    sum_sq = sum(x * x for x in embedding)
    if sum_sq == 0:
        return embedding
    norm = math.sqrt(sum_sq)
    return [x / norm for x in embedding]


def generate_embeddings_batch(
    texts: list[str],
    batch_size: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts, batched for efficiency.
    Optionally populating `usage_out`.

    Checks cache first, batches uncached texts through the provider,
    and returns results in original order.

    Args:
        on_progress: Optional callback(embedded_so_far, total) called after each batch.
        usage_out: Optional dict mutated in place with aggregated usage for
            uncached provider calls only. Cache-hit items contribute zero spend.
    """
    effective_batch_size = (
        batch_size if batch_size is not None else settings.embedding_batch_size or 32
    )
    total = len(texts)
    results: list[list[float] | None] = [None] * total
    uncached: list[tuple[int, str, str]] = []  # (index, cache_key, text)
    total_usage: EmbeddingUsage | None = None

    for i, text in enumerate(texts):
        cache_key = _cache_key(text)
        cached = _get_cache().get(cache_key)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append((i, cache_key, text))

    cached_count = total - len(uncached)
    embedded = cached_count
    if on_progress and cached_count > 0:
        on_progress(embedded, total)

    # Batch embed uncached texts
    for start in range(0, len(uncached), effective_batch_size):
        batch = uncached[start : start + effective_batch_size]
        batch_texts = [t for _, _, t in batch]
        batch_usage: EmbeddingUsage = {}
        if usage_out is None:
            embeddings = _generate_batch_uncached(batch_texts)
        else:
            embeddings = _generate_batch_uncached(batch_texts, usage_out=batch_usage)
        for (idx, cache_key, _), embedding in zip(batch, embeddings):
            results[idx] = embedding
            _get_cache().put(cache_key, embedding)
        total_usage = _merge_usage(total_usage, batch_usage or None)
        embedded += len(batch)
        if on_progress:
            on_progress(embedded, total)

    _set_usage_out(usage_out, total_usage)
    if any(result is None for result in results):
        raise RuntimeError("Embedding batch completed with missing results")
    return cast(list[list[float]], results)


@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
def _generate_batch_uncached(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts without cache lookup. Forwarding `usage_out`."""
    provider = settings.embedding_provider

    match provider:
        case "ollama":
            return _embed_ollama_batch(texts, usage_out=usage_out)
        case "openai":
            return _embed_openai_batch(texts, usage_out=usage_out)
        case "mistral":
            return _embed_mistral_batch(texts, usage_out=usage_out)
        case "voyage":
            return _embed_voyage_batch(texts, usage_out=usage_out)
        case "gemini":
            return _embed_gemini_batch(texts, usage_out=usage_out)
        case "onnx":
            return _embed_onnx_batch(texts, usage_out=usage_out)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")


def _order_by_index(provider: str, items: list, expected: int) -> list:
    """Return `items` in request order, using each item's `index` when it has one.

    The caller pairs texts to vectors POSITIONALLY, so array order is
    load-bearing. OpenAI models `index` as a REQUIRED field on every embedding
    precisely because position in `data` is not the contract -- the index is.
    We used to read the array in whatever order it arrived, which is fine right
    up until it isn't, and the failure is silent: each memory gets its
    neighbour's vector, cached under its own key, with nothing downstream able
    to notice. (TBU-209)

    Providers whose responses carry no index (ollama and voyage return bare
    lists) fall through unchanged -- there is nothing better available, and
    saying so here beats leaving the assumption unwritten.
    """
    # `isinstance(..., int)` rather than a None check: list, tuple and str all
    # carry an `.index` METHOD, so a bare list of vectors would otherwise look
    # indexed and blow up in sorted(). Caught by the ollama/voyage passthrough
    # test, which is the whole reason it exists.
    indices = [getattr(item, "index", None) for item in items]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
        return items
    if sorted(indices) != list(range(expected)):
        raise RuntimeError(
            f"{provider} returned indices {sorted(indices)} for {expected} inputs -- "
            "cannot establish request order, refusing to guess"
        )
    return [item for _, item in sorted(zip(indices, items), key=lambda pair: pair[0])]


def _assert_full_batch(provider: str, texts: list[str], embeddings: list) -> None:
    """A batch must return exactly one embedding per input, in order.

    The caller pairs the two lists POSITIONALLY with zip() and writes the result
    into the embedding cache under each text's key. A short response therefore
    does not merely lose data -- if a provider drops an item from the middle,
    every text after it is paired with its neighbour's vector and that
    mispairing is cached. Nothing downstream can detect it.

    Until 2026-07-30 only Gemini checked this, because issue #60 forced it. The
    other providers relied on `any(result is None)` further up, which fires
    after the fact, cannot say which provider misbehaved, and catches only the
    short case -- never a wrong-order response of the right length.
    """
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"{provider} returned {len(embeddings)} embeddings for {len(texts)} inputs -- "
            "refusing to pair texts with the wrong vectors"
        )


def _embed_onnx_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    embeddings = [_embed_onnx(t) for t in texts]
    _assert_full_batch("onnx", texts, embeddings)
    _set_usage_out(usage_out, _model_only_usage("onnx"))
    return embeddings


def _embed_ollama_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    client = _get_ollama_client()
    kwargs: dict = {"model": settings.ollama_embed_model, "input": texts}
    if settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim
    response = client.embed(**kwargs)
    # Ollama returns a bare list with no per-item index, so request order is
    # the only signal available. See TBU-209.
    embeddings = response["embeddings"]
    _assert_full_batch("ollama", texts, embeddings)
    for emb in embeddings:
        _validate_dim(emb)
    _set_usage_out(usage_out, _model_only_usage("ollama"))
    return embeddings


def _embed_openai_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required when embedding_provider=openai")

    client = _get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=settings.embedding_dim,
    )
    ordered = _order_by_index("openai", list(response.data), len(texts))
    embeddings = [d.embedding for d in ordered]
    _assert_full_batch("openai", texts, embeddings)
    for emb in embeddings:
        _validate_dim(emb)
    _set_usage_out(usage_out, _extract_openai_usage(response))
    return embeddings


def _embed_mistral_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY required when embedding_provider=mistral")
    client = _get_mistral_client()
    response = client.embeddings.create(
        model=settings.mistral_embed_model,
        inputs=texts,
    )
    # Mistral mirrors the OpenAI response shape; _order_by_index is a no-op
    # if this SDK turns out not to expose `index`.
    ordered = _order_by_index("mistral", list(response.data), len(texts))
    embeddings = [d.embedding for d in ordered]
    _assert_full_batch("mistral", texts, embeddings)
    for emb in embeddings:
        _validate_dim(emb)
    _set_usage_out(usage_out, _model_only_usage("mistral"))
    return embeddings


def _embed_voyage_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY required when embedding_provider=voyage")
    client = _get_voyage_client()
    all_embeddings = []
    total_usage: EmbeddingUsage | None = None
    # Voyage max 1000 per request
    for start in range(0, len(texts), 1000):
        batch = texts[start : start + 1000]
        response = client.embed(
            texts=batch,
            model=settings.voyage_embed_model,
            output_dimension=settings.embedding_dim,
        )
        # Voyage returns a bare list per chunk -- no index to reorder by.
        all_embeddings.extend(response.embeddings)
        total_usage = _merge_usage(total_usage, _extract_voyage_usage(response))
    _assert_full_batch("voyage", texts, all_embeddings)
    for emb in all_embeddings:
        _validate_dim(emb)
    _set_usage_out(usage_out, total_usage)
    return all_embeddings


class _GeminiShortResponseError(RuntimeError):
    """Gemini batchEmbedContents returned fewer embeddings than items submitted.

    Observed as a transient: an immediate retry typically returns a full
    response. See OM-mcp issue #60.
    """


# HTTP statuses where another attempt is genuinely expected to help. Anything
# else from the 4xx range is the caller's problem and will fail identically on
# every retry -- a bad key, a disabled API, a model that does not exist.
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Is another attempt worth making? Classified by status, not by prose.

    `google.genai.errors.APIError` carries `.code` (the HTTP status), so use
    it. The previous version matched substrings against the whole message,
    which was wrong in both directions (TBU-210):

    - `"quota" in msg.lower()` retried PERMANENT failures. Billing disabled and
      a disabled API both return 403 and both say "quota", so an outcome fixed
      from the first call cost six attempts and 93s of backoff before surfacing
      the real cause -- buried under the retry noise.
    - `"429" in msg` matched anywhere in the text, so a request id or a token
      count containing those digits read as a rate limit.

    Bare "quota" is deliberately gone from the fallback. `RESOURCE_EXHAUSTED`
    is the status string that actually accompanies a retryable 429, and it does
    not appear on the terminal 403s.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code in _RETRYABLE_HTTP_STATUS

    # No typed status: a raw transport error, or a provider that does not carry
    # one. Match whole tokens so a number embedded in prose cannot trigger it.
    msg = str(exc)
    if re.search(r"\b(?:408|429|500|502|503|504)\b", msg):
        return True
    return "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg


# Models observed to ignore multi-input batching in this process. Populated at
# runtime, never persisted -- if Google fixes the model, a restart re-probes.
#
# `gemini-embedding-2` began returning exactly ONE embedding for any number of
# inputs (verified 2026-07-30: batch 2, 3 and 5 all returned 1, while
# `gemini-embedding-001` returned the full count). That is not the flaky short
# response of issue #60 -- it is deterministic, so retrying can never win, and
# every batched caller (re_embed_all, adapter ingestion, importers, benchmarks)
# failed hard after exhausting its attempts. Interactive use never noticed
# because storing one memory embeds one input. See TBU-208.
_GEMINI_NO_BATCH: set[str] = set()

# A genuine flake resolves on the next call; anything beyond that is a provider
# that does not batch, and further attempts just add latency before the
# fallback. Rate limits keep the full attempt budget separately.
_GEMINI_SHORT_RESPONSE_ATTEMPTS = 2


def _reset_gemini_batch_support() -> None:
    """Forget which models were observed not to batch. For tests."""
    _GEMINI_NO_BATCH.clear()


def _embed_gemini_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY required when embedding_provider=gemini")
    client = _get_gemini_client()

    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    # OGHAM_RETRY_FAST collapses tenacity backoff for tests: the real call
    # chain stays at the production schedule, only test runs short-circuit it.
    _fast = bool(os.environ.get("OGHAM_RETRY_FAST"))
    _wait = (
        wait_exponential(multiplier=0, min=0, max=0)
        if _fast
        else wait_exponential(multiplier=3, min=3, max=90)
    )

    model = settings.gemini_embed_model

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=_wait,
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call(batch: list[str], sink: EmbeddingUsage | None) -> list[list[float]]:
        """One provider call. Raises _GeminiShortResponseError on a count mismatch.

        The count check is a hard assertion, not a nicety: accepting a response
        with fewer embeddings than inputs would silently pair text with the
        WRONG vector, which is far worse than an error.
        """
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config={"output_dimensionality": settings.embedding_dim},
        )
        returned = len(response.embeddings or [])
        if returned != len(batch):
            raise _GeminiShortResponseError(
                f"Gemini returned {returned} embeddings for {len(batch)} inputs"
            )
        embeddings = [e.values for e in response.embeddings]
        for emb in embeddings:
            _validate_dim(emb)
        if settings.embedding_dim < 3072 and not _gemini_model_pre_normalizes():
            embeddings = [_l2_normalize(emb) for emb in embeddings]
        _set_usage_out(sink, _extract_gemini_usage(response))
        return embeddings

    def _one_at_a_time() -> list[list[float]]:
        """Degrade to single-input calls, which the provider does honour."""
        embeddings: list[list[float]] = []
        total: EmbeddingUsage | None = None
        for text in texts:
            per_call: EmbeddingUsage = {}
            embeddings.extend(_call([text], per_call))
            total = _merge_usage(total, per_call or None)
        _set_usage_out(usage_out, total)
        return embeddings

    if model in _GEMINI_NO_BATCH and len(texts) > 1:
        return _one_at_a_time()

    last: _GeminiShortResponseError | None = None
    for attempt in range(_GEMINI_SHORT_RESPONSE_ATTEMPTS):
        try:
            return _call(texts, usage_out)
        except _GeminiShortResponseError as exc:
            last = exc
            logger.warning(
                "gemini: %s (attempt %d/%d)", exc, attempt + 1, _GEMINI_SHORT_RESPONSE_ATTEMPTS
            )
            if len(texts) == 1:
                # Nothing left to degrade to -- a single input already failed.
                raise

    _GEMINI_NO_BATCH.add(model)
    logger.warning(
        "gemini: model %r does not honour multi-input batching (%s) -- falling back to "
        "one request per input for the rest of this process. See TBU-208.",
        model,
        last,
    )
    return _one_at_a_time()


def clear_embedding_cache() -> int:
    """Clear the embedding cache. Returns number of entries cleared."""
    return _get_cache().clear()
