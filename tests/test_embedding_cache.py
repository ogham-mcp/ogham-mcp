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


# --- Space reclamation ---------------------------------------------------
# SQLite reuses freed pages but never returns them to the filesystem unless
# auto_vacuum is on. Eviction and clear() both delete, so without it the file
# only grows: one real cache reached 2.16 GB holding ~7 MB of rows.


def _pragma(path, name):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


def _fill(cache, n, start=0):
    for i in range(start, start + n):
        cache.put(f"key-{i}", [float(i)] * 512)


def test_new_cache_file_uses_incremental_auto_vacuum(tmp_path):
    EmbeddingCache(cache_dir=str(tmp_path))
    assert _pragma(tmp_path / "embeddings.db", "auto_vacuum") == 2  # INCREMENTAL


def test_existing_bloated_file_is_converted_and_shrunk_on_open(tmp_path):
    import sqlite3

    db = tmp_path / "embeddings.db"
    # Build the shape of a pre-fix cache: auto_vacuum NONE, rows written then
    # deleted, freed pages left in the file.
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE embeddings (key TEXT PRIMARY KEY, value BLOB NOT NULL, "
        "created_at REAL NOT NULL DEFAULT (unixepoch('now')))"
    )
    conn.executemany(
        "INSERT INTO embeddings (key, value) VALUES (?, ?)",
        [(f"k{i}", "[" + ",".join(["0.5"] * 512) + "]") for i in range(400)],
    )
    conn.commit()
    conn.execute("DELETE FROM embeddings")
    conn.commit()
    conn.close()
    assert _pragma(db, "auto_vacuum") == 0
    bloated_free = _pragma(db, "freelist_count")
    assert bloated_free > 50, "fixture should leave freed pages behind"

    EmbeddingCache(cache_dir=str(tmp_path))

    assert _pragma(db, "auto_vacuum") == 2
    assert _pragma(db, "freelist_count") == 0


def test_clear_returns_space_to_the_filesystem(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path))
    _fill(cache, 300)
    full_pages = _pragma(tmp_path / "embeddings.db", "page_count")

    cache.clear()

    db = tmp_path / "embeddings.db"
    assert _pragma(db, "freelist_count") == 0
    assert _pragma(db, "page_count") < full_pages / 4


def test_eviction_returns_space_to_the_filesystem(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path), max_size=50)
    _fill(cache, 300)  # every put past 50 evicts
    assert len(cache) == 50
    assert _pragma(tmp_path / "embeddings.db", "freelist_count") == 0
