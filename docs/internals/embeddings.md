# Ogham MCP — Embedding System

## Provider Architecture

Five embedding providers, selected via `EMBEDDING_PROVIDER` env var:

| Provider | Model | Default Dim | Batch Limit | Notes |
|----------|-------|-------------|-------------|-------|
| `ollama` (default) | `embeddinggemma` | 512 | 10 | Local, free, dimension control via `dimensions` param |
| `openai` | `text-embedding-3-small` | 1024 | 500 | Supports dimension reduction |
| `mistral` | `mistral-embed` | 1024 | 32 | 16,384 token limit per request |
| `voyage` | `voyage-4-lite` | 1024 | 500 | 1,000 inputs per request, auto-batched |
| `onnx` | `bge-m3` (local ONNX) | 1024 | 10 | Dense + sparse vectors, CPU-only, requires `[onnx]` extra |

Each provider has singleton client instances (lazy-created). Dimension validation runs after every embedding call.

## ONNX Provider (`onnx_embedder.py`)

Local BGE-M3 embedding via `onnxruntime` — produces both dense (1024-dim) and sparse vectors in a single forward pass. No API calls, no GPU required.

### Setup

```bash
pip install ogham[onnx]        # installs onnxruntime + tokenizers
ogham download-model bge-m3    # downloads ~2.2GB model to ~/.cache/ogham/bge-m3-onnx/
```

Set `EMBEDDING_PROVIDER=onnx` and optionally `ONNX_MODEL_PATH` to override the default model location.

### Architecture

- **Singleton session** — `_get_model()` lazy-loads the ONNX session and HuggingFace tokenizer (thread-safe, double-checked locking)
- **Selective outputs** — `session.run(["dense_embeddings", "sparse_weights"], ...)` skips ColBERT vector allocation
- **Sequential encoding** — no batch padding; sparse weight extraction needs per-token attention masks without padding noise
- **Memory safety** — `enable_cpu_mem_arena = False` prevents arena accumulation across inferences (critical for re-embed batches)

### Sparse Vectors

Sparse output is converted to pgvector `sparsevec` format (`{idx:weight, ...}/vocab_size`). Special tokens (PAD/UNK/CLS/SEP, IDs 0-3) are filtered, zero weights dropped, duplicate token IDs resolved by keeping max weight. Max observed non-zero elements: ~280 (well within pgvector's 1,000 limit).

### Three-Signal Search

When the ONNX provider is active, `generate_embedding_full()` returns `(dense, sparse_str)` instead of `(dense, None)`. The service layer detects this and uses `hybrid_search_memories_sparse()` — RRF fusion of FTS + dense cosine + sparse inner product — instead of the two-signal path.

### Performance (AMD Ryzen 5 7535U, CPU)

- Short text (~1.5K chars): ~0.3s/embedding
- Long text (~5K chars): ~10s/embedding
- RSS: ~4.3GB peak

## Caching (`embedding_cache.py`)

### Architecture

- **SQLite-backed** persistent cache at `~/.cache/ogham/embeddings.db`
- Thread-safe via `threading.Lock`
- Schema: `key TEXT PRIMARY KEY, value BLOB, created_at REAL`

### Cache Key

```python
SHA256(f"{provider}:{dim}:{text}")
```

Switching providers or dimensions automatically invalidates cached vectors because the key prefix changes.

### Eviction

- LRU-style: when size exceeds `max_size` (default 10,000), deletes oldest entries by `created_at`
- Eviction runs after every `put()`

### Operations

- `get(key)` → cache hit/miss tracking
- `put(key, embedding)` → insert/replace + evict
- `clear()` → wipe all + reset stats
- `stats()` → size, max_size, hits, misses, hit_rate

## Batch Embedding

`generate_embeddings_batch(texts, batch_size?, on_progress?)`:

1. Check cache for each text
2. Group uncached texts into batches (size from `EMBEDDING_BATCH_SIZE` or provider default)
3. Call provider batch API
4. Cache results
5. Return in original order

Progress callback `on_progress(embedded_so_far, total)` fires after each batch.

## Retry

All provider calls (single + batch) are wrapped with:

```python
@with_retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, OSError))
```

Exponential backoff: 0.5s → 1.0s → 2.0s

The retry decorator also catches `psycopg.OperationalError` if psycopg is installed (for database retry).
