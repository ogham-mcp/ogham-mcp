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
        self._migrate_sparse()
        self._ensure_incremental_vacuum()

    def _ensure_incremental_vacuum(self) -> None:
        """Make deleted rows give their space back to the filesystem.

        SQLite reuses freed pages for later inserts but never returns them to
        the disk unless auto_vacuum is on. Eviction and clear() both delete,
        so without this the file only grows -- one real cache reached 2.16 GB
        holding ~7 MB of rows. auto_vacuum is a property of the FILE, which the
        Go CLI shares and converts the same way; whichever opens it first does
        the conversion and the other sees it done.

        Converting an existing file needs one VACUUM, which can fail with
        SQLITE_BUSY while another process holds the database. The cache is
        best-effort, so skip and retry on the next open.
        """
        if self._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2:  # INCREMENTAL
            return
        try:
            self._conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            self._conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass

    def _reclaim(self) -> None:
        """Return freed pages to the filesystem. Caller holds self._lock."""
        try:
            # fetchall: each step of incremental_vacuum frees pages; stepping
            # once from execute() is not guaranteed to run it to completion.
            self._conn.execute("PRAGMA incremental_vacuum").fetchall()
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def _migrate_sparse(self) -> None:
        """Add sparse column if it doesn't exist yet."""
        cursor = self._conn.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        if "sparse" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN sparse TEXT")
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

    def get_full(self, key: str) -> tuple[list[float], str | None] | None:
        """Return (dense_embedding, sparse_sparsevec_string) or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, sparse FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(row[0]), row[1]

    def put(self, key: str, embedding: list[float], sparse: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, value, sparse) VALUES (?, ?, ?)",
                (key, json.dumps(embedding), sparse),
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
            self._reclaim()
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
            self._reclaim()
