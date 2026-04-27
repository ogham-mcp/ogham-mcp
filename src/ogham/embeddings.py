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
    if settings.embedding_dim < 3072:
        embedding = _l2_normalize(embedding)
    _set_usage_out(usage_out, _extract_gemini_usage(response))
    return embedding


def _validate_dim(embedding: list[float]) -> None:
    if len(embedding) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.embedding_dim}"
        )


def _l2_normalize(embedding: list[float]) -> list[float]:
    """Rescale `embedding` to unit length. Zero vectors pass through unchanged
    (normalizing would divide by zero).

    Gemini only pre-normalizes vectors at the model's native 3072 dim. At
    512 / 768 / 1536 the magnitude varies, which turns cosine similarity
    into a magnitude-weighted score. Google's docs explicitly say the
    caller must normalize sub-3072 outputs client-side:
    https://ai.google.dev/gemini-api/docs/embeddings
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


def _embed_onnx_batch(
    texts: list[str],
    usage_out: EmbeddingUsage | None = None,
) -> list[list[float]]:
    embeddings = [_embed_onnx(t) for t in texts]
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
    embeddings = response["embeddings"]
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
    embeddings = [d.embedding for d in response.data]
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
    embeddings = [d.embedding for d in response.data]
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
        all_embeddings.extend(response.embeddings)
        total_usage = _merge_usage(total_usage, _extract_voyage_usage(response))
    for emb in all_embeddings:
        _validate_dim(emb)
    _set_usage_out(usage_out, total_usage)
    return all_embeddings


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect Gemini 429/quota/503 errors that are worth retrying."""
    msg = str(exc)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
        or "503" in msg
        or "UNAVAILABLE" in msg
    )


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

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(multiplier=3, min=3, max=90),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        response = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=texts,
            config={"output_dimensionality": settings.embedding_dim},
        )
        embeddings = [e.values for e in response.embeddings]
        for emb in embeddings:
            _validate_dim(emb)
        if settings.embedding_dim < 3072:
            embeddings = [_l2_normalize(emb) for emb in embeddings]
        _set_usage_out(usage_out, _extract_gemini_usage(response))
        return embeddings

    return _call()


def clear_embedding_cache() -> int:
    """Clear the embedding cache. Returns number of entries cleared."""
    return _get_cache().clear()
