# Ogham MCP — Storage Backends

## Protocol (`backends/protocol.py`)

`DatabaseBackend` is a `@runtime_checkable` Protocol defining the contract every backend must satisfy. All methods return plain dicts/lists — no ORM objects.

### Method Groups

| Group | Methods |
|-------|---------|
| **Core CRUD** | `store_memory`, `store_memories_batch`, `update_memory`, `get_memory_by_id`, `delete_memory` |
| **Search** | `search_memories`, `hybrid_search_memories`, `list_recent_memories`, `get_all_memories_full`, `get_all_memories_content` |
| **Batch** | `batch_check_duplicates`, `batch_update_embeddings` |
| **Access/Confidence** | `record_access`, `update_confidence` |
| **Profile/Stats** | `get_memory_stats`, `list_profiles`, `get_profile_ttl`, `set_profile_ttl`, `cleanup_expired`, `count_expired` |
| **Knowledge Graph** | `auto_link_memory`, `link_unlinked_memories`, `explore_memory_graph`, `create_relationship`, `get_related_memories` |

## PostgresBackend (`backends/postgres.py`)

Direct PostgreSQL via `psycopg` connection pool.

### Connection Pool

- `psycopg_pool.ConnectionPool` with min_size=1, max_size=5
- Lazily created on first use
- Uses `dict_row` row factory — all results are plain dicts

### Query Pattern

All queries go through `_execute(query, params, fetch)`:

- `fetch="all"` → `fetchall()` → `list[dict]`
- `fetch="one"` → `fetchone()` → `dict | None`
- `fetch="scalar"` → first value of first row
- `fetch="none"` → no fetch (side-effect queries)

### SQL Injection Prevention

- All values are parameterised (`%(name)s` placeholders)
- Column names in `update_memory` and `store_memories_batch` are validated against `_ALLOWED_MEMORY_COLUMNS` allowlist
- Embedding vectors are formatted as Postgres vector literals: `[0.1,0.2,...]`

### Database Functions Used

The backend calls several PostgreSQL functions defined in the `sql/` schema:

| Function | Purpose |
|----------|---------|
| `match_memories()` | Pure vector similarity search |
| `hybrid_search_memories()` | RRF of vector + full-text search |
| `auto_link_memory()` | Find similar memories and create edges |
| `link_unlinked_memories()` | Backfill links for unlinked memories |
| `explore_memory_graph()` | Seed search + graph traversal |
| `get_related_memories()` | Graph traversal from a memory |
| `record_access()` | Bump access count/timestamp |
| `update_confidence()` | Adjust confidence score |
| `get_memory_stats_sql()` | Profile statistics as JSON |
| `get_profile_counts()` | List profiles with counts |
| `cleanup_expired_memories()` | Delete expired rows |
| `count_expired_memories()` | Count expired rows |
| `batch_check_duplicates()` | Check array of embeddings for duplicates |
| `batch_update_embeddings()` | Update embeddings for array of IDs |

### Pagination

`get_all_memories_full` and `get_all_memories_content` use keyset pagination (cursor-based) with batch size 1000, using `(created_at, id)` or `(id)` as cursor keys.

### Retry

All search/retrieval methods are decorated with `@with_retry(max_attempts=2, base_delay=0.3)`.

## SupabaseBackend (`backends/supabase.py`)

Uses PostgREST client library. Calls the same PostgreSQL functions via Supabase RPC. Original backend — likely being superseded by PostgresBackend for self-hosted deployments.

## GatewayBackend (`backends/gateway.py`)

HTTP proxy that delegates all operations to a remote Ogham gateway REST API. Configured via `OGHAM_GATEWAY_URL` and `OGHAM_API_KEY`. Used when running Ogham as a shared service.

## Database Schema

### Tables

- `memories` — main table: id (UUID), content, embedding (vector), metadata (JSONB), source, profile, tags (text[]), confidence, importance, surprise, compression_level, original_content, recurrence_days, access_count, last_accessed_at, created_at, updated_at, expires_at, fts (tsvector)
- `memory_relationships` — edges: source_id, target_id, relationship (enum type), strength, created_by, metadata
- `profile_settings` — per-profile config: profile (unique), ttl_days

### Relationship Types

Defined as a PostgreSQL enum `relationship_type`: `supports`, `contradicts`, `similar`, `relates_to`, etc.

### Migrations

15 incremental migrations covering: cognitive scoring → confidence → embedding model switch → hybrid search → batch dedup → set-based batch update → CCF search → memory relationships → graph explorer → impact analysis → graph centrality boost → temporal columns → halfvec compression → LZ4 TOAST compression → temporal auto-extract.
