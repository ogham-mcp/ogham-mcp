"""Persistent SQLite-backed embedding cache."""

import json
import os
import sqlite3
import threading


class EmbeddingCache:
    """SQLite-backed persistent cache for embedding vectors.

    Key: SHA-256 hex digest of the text.
    Value: embedding vector (list of floats), stored as JSON blob.
    """

    def __init__(self, cache_dir: str | None = None, max_size: int = 10000):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ogham")
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "embeddings.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch('now'))
            )"""
        )
        self._conn.commit()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(row[0])

    def put(self, key: str, embedding: list[float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, value) VALUES (?, ?)",
                (key, json.dumps(embedding)),
            )
            self._conn.commit()
            self._evict()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM embeddings WHERE key = ?", (key,)).fetchone()
            return row is not None

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return row[0]

    def clear(self) -> int:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()
            self._hits = 0
            self._misses = 0
            return count

    def stats(self) -> dict:
        with self._lock:
            size = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            total = self._hits + self._misses
            return {
                "size": size,
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": 0,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def _evict(self) -> None:
        """Must be called while holding self._lock."""
        count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if count > self._max_size:
            excess = count - self._max_size
            self._conn.execute(
                "DELETE FROM embeddings WHERE key IN "
                "(SELECT key FROM embeddings ORDER BY created_at ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()
