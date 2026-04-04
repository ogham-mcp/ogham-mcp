import hashlib
import logging

from ogham.config import settings
from ogham.embedding_cache import EmbeddingCache
from ogham.retry import with_retry

logger = logging.getLogger(__name__)

_cache = EmbeddingCache(
    cache_dir=settings.embedding_cache_dir,
    max_size=settings.embedding_cache_max_size,
)


def get_cache_stats() -> dict:
    """Return cache statistics."""
    return _cache.stats()


def _cache_key(text: str) -> str:
    """Build a cache key scoped to the current provider and dimension.

    Switching providers or dimensions automatically invalidates cached vectors
    because the key prefix changes.
    """
    prefix = f"{settings.embedding_provider}:{settings.embedding_dim}:"
    return hashlib.sha256((prefix + text).encode()).hexdigest()


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses persistent SQLite cache keyed by provider + dimension + SHA256 of text
    to avoid re-embedding identical content. Switching providers or dimensions
    automatically invalidates cached vectors.
    """
    cache_key = _cache_key(text)

    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Embedding cache hit for text hash %s", cache_key[:8])
        return cached

    embedding = _generate_uncached(text)
    _cache.put(cache_key, embedding)
    return embedding


@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
def _generate_uncached(text: str) -> list[float]:
    """Generate embedding without cache lookup."""
    provider = settings.embedding_provider

    match provider:
        case "ollama":
            return _embed_ollama(text)
        case "openai":
            return _embed_openai(text)
        case "mistral":
            return _embed_mistral(text)
        case "voyage":
            return _embed_voyage(text)
        case "gemini":
            return _embed_gemini(text)
        case "onnx":
            return _embed_onnx(text)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")


def _embed_onnx(text: str) -> list[float]:
    from ogham.onnx_embedder import encode

    result = encode(text, settings.onnx_model_path or None)
    embedding = result["dense"]
    _validate_dim(embedding)
    return embedding


_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        _ollama_client = ollama.Client(host=settings.ollama_url, timeout=settings.ollama_timeout)
    return _ollama_client


def _embed_ollama(text: str) -> list[float]:
    client = _get_ollama_client()
    kwargs: dict = {"model": settings.ollama_embed_model, "input": text}
    if settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim
    response = client.embed(**kwargs)
    embedding = response["embeddings"][0]
    _validate_dim(embedding)
    return embedding


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _embed_openai(text: str) -> list[float]:
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
    return embedding


_mistral_client = None


def _get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        from mistralai import Mistral

        _mistral_client = Mistral(api_key=settings.mistral_api_key)
    return _mistral_client


def _embed_mistral(text: str) -> list[float]:
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY required when embedding_provider=mistral")
    client = _get_mistral_client()
    response = client.embeddings.create(
        model=settings.mistral_embed_model,
        inputs=[text],
    )
    embedding = response.data[0].embedding
    _validate_dim(embedding)
    return embedding


_voyage_client = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        _voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    return _voyage_client


def _embed_voyage(text: str) -> list[float]:
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
    return embedding


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _embed_gemini(text: str) -> list[float]:
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
    return embedding


def _validate_dim(embedding: list[float]) -> None:
    if len(embedding) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.embedding_dim}"
        )


def generate_embeddings_batch(
    texts: list[str],
    batch_size: int | None = None,
    on_progress: callable = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts, batched for efficiency.

    Checks cache first, batches uncached texts through the provider,
    and returns results in original order.

    Args:
        on_progress: Optional callback(embedded_so_far, total) called after each batch.
    """
    if batch_size is None:
        batch_size = settings.embedding_batch_size
    total = len(texts)
    results: list[list[float] | None] = [None] * total
    uncached: list[tuple[int, str, str]] = []  # (index, cache_key, text)

    for i, text in enumerate(texts):
        cache_key = _cache_key(text)
        cached = _cache.get(cache_key)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append((i, cache_key, text))

    cached_count = total - len(uncached)
    embedded = cached_count
    if on_progress and cached_count > 0:
        on_progress(embedded, total)

    # Batch embed uncached texts
    for start in range(0, len(uncached), batch_size):
        batch = uncached[start : start + batch_size]
        batch_texts = [t for _, _, t in batch]
        embeddings = _generate_batch_uncached(batch_texts)
        for (idx, cache_key, _), embedding in zip(batch, embeddings):
            results[idx] = embedding
            _cache.put(cache_key, embedding)
        embedded += len(batch)
        if on_progress:
            on_progress(embedded, total)

    return results


@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
def _generate_batch_uncached(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts without cache lookup."""
    provider = settings.embedding_provider

    match provider:
        case "ollama":
            return _embed_ollama_batch(texts)
        case "openai":
            return _embed_openai_batch(texts)
        case "mistral":
            return _embed_mistral_batch(texts)
        case "voyage":
            return _embed_voyage_batch(texts)
        case "gemini":
            return _embed_gemini_batch(texts)
        case "onnx":
            return _embed_onnx_batch(texts)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")


def _embed_onnx_batch(texts: list[str]) -> list[list[float]]:
    return [_embed_onnx(t) for t in texts]


def _embed_ollama_batch(texts: list[str]) -> list[list[float]]:
    client = _get_ollama_client()
    kwargs: dict = {"model": settings.ollama_embed_model, "input": texts}
    if settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim
    response = client.embed(**kwargs)
    embeddings = response["embeddings"]
    for emb in embeddings:
        _validate_dim(emb)
    return embeddings


def _embed_openai_batch(texts: list[str]) -> list[list[float]]:
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
    return embeddings


def _embed_mistral_batch(texts: list[str]) -> list[list[float]]:
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
    return embeddings


def _embed_voyage_batch(texts: list[str]) -> list[list[float]]:
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY required when embedding_provider=voyage")
    client = _get_voyage_client()
    all_embeddings = []
    # Voyage max 1000 per request
    for start in range(0, len(texts), 1000):
        batch = texts[start : start + 1000]
        response = client.embed(
            texts=batch,
            model=settings.voyage_embed_model,
            output_dimension=settings.embedding_dim,
        )
        all_embeddings.extend(response.embeddings)
    for emb in all_embeddings:
        _validate_dim(emb)
    return all_embeddings


def _embed_gemini_batch(texts: list[str]) -> list[list[float]]:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY required when embedding_provider=gemini")
    client = _get_gemini_client()
    response = client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=texts,
        config={"output_dimensionality": settings.embedding_dim},
    )
    embeddings = [e.values for e in response.embeddings]
    for emb in embeddings:
        _validate_dim(emb)
    return embeddings


def clear_embedding_cache() -> int:
    """Clear the embedding cache. Returns number of entries cleared."""
    return _cache.clear()
