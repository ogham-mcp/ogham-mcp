import pytest

from ogham.embedding_cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    """Create a cache with a temp directory."""
    return EmbeddingCache(cache_dir=str(tmp_path), max_size=100)


def test_put_and_get(cache):
    """Should store and retrieve an embedding by hash key."""
    embedding = [0.1] * 768
    cache.put("abc123", embedding)
    result = cache.get("abc123")
    assert result == embedding


def test_get_miss(cache):
    """Should return None for unknown key."""
    assert cache.get("nonexistent") is None


def test_contains(cache):
    """Should support 'in' operator."""
    cache.put("key1", [0.1] * 768)
    assert "key1" in cache
    assert "key2" not in cache


def test_len(cache):
    """Should return number of cached entries."""
    assert len(cache) == 0
    cache.put("k1", [0.1] * 768)
    cache.put("k2", [0.2] * 768)
    assert len(cache) == 2


def test_persistence(tmp_path):
    """Cache should survive process restart (new instance, same dir)."""
    cache1 = EmbeddingCache(cache_dir=str(tmp_path), max_size=100)
    cache1.put("persistent_key", [0.5] * 768)
    del cache1

    cache2 = EmbeddingCache(cache_dir=str(tmp_path), max_size=100)
    result = cache2.get("persistent_key")
    assert result == [0.5] * 768


def test_eviction(tmp_path):
    """Should evict oldest entries when over max_size."""
    cache = EmbeddingCache(cache_dir=str(tmp_path), max_size=2)
    cache.put("first", [0.1] * 768)
    cache.put("second", [0.2] * 768)
    cache.put("third", [0.3] * 768)

    assert len(cache) == 2
    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.get("third") is not None


def test_clear(cache):
    """Should remove all entries."""
    cache.put("k1", [0.1] * 768)
    cache.put("k2", [0.2] * 768)
    count = cache.clear()
    assert count == 2
    assert len(cache) == 0


def test_stats(cache):
    """Should track hits, misses, and size."""
    cache.put("k1", [0.1] * 768)
    cache.get("k1")  # hit
    cache.get("missing")  # miss

    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
