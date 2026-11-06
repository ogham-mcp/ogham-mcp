# Ogham MCP — Supabase & Gateway Backends

## SupabaseBackend (`backends/supabase.py`)

The original backend, using Supabase's PostgREST client library. Calls the same PostgreSQL functions as PostgresBackend but via Supabase RPC.

### Connection

- Uses `supabase.create_client(url, key)` with `SUPABASE_URL` and `SUPABASE_KEY` env vars
- Lazy singleton client creation
- All operations go through `client.table("memories")` or `client.rpc("function_name", params)`

### Query Patterns

**CRUD operations** use the PostgREST query builder:

```python
client.table("memories").insert({...}).execute()
client.table("memories").update({...}).eq("id", id).execute()
client.table("memories").delete().eq("id", id).execute()
client.table("memories").select("*").eq("id", id).single().execute()
```

**Search operations** use RPC calls to the same PostgreSQL functions:

```python
client.rpc("match_memories", {"query_embedding": [...], "match_count": 10, ...}).execute()
client.rpc("hybrid_search_memories", {"query_embedding": [...], "query_text": "...", ...}).execute()
```

**Listing** uses PostgREST query builder with ordering and filtering:

```python
client.table("memories").select("*").eq("profile", profile).order("created_at", desc=True).limit(
    n
).execute()
```

### Differences from PostgresBackend

- No connection pool management (Supabase handles it)
- PostgREST handles JSON serialization/deserialization
- RLS (Row Level Security) can be applied at the Supabase layer
- Embedding vectors passed as JSON arrays (PostgREST serializes them)
- Error handling wraps Supabase client exceptions

### Limitations

- Batch operations may be less efficient (PostgREST overhead per request)
- No direct cursor-based pagination (uses offset/limit instead)
- Being superseded by PostgresBackend for self-hosted deployments

## GatewayBackend (`backends/gateway.py`)

HTTP proxy backend that delegates all operations to a remote Ogham gateway REST API. Used when running Ogham as a shared service.

### Configuration

- `OGHAM_GATEWAY_URL` — Base URL of the gateway (e.g. `https://ogham.example.com`)
- `OGHAM_API_KEY` — Bearer token for authentication

### Architecture

```text
Client → GatewayBackend → HTTP → Ogham Gateway Server → PostgresBackend → PostgreSQL
```

### HTTP Client

- Uses `httpx.Client` (sync) with configurable timeout
- All requests include `Authorization: Bearer {api_key}` header
- Content-Type: `application/json`

### Endpoint Mapping

Every `DatabaseBackend` protocol method maps to a REST endpoint:

| Method | HTTP | Endpoint |
|--------|------|----------|
| `store_memory` | POST | `/api/memories` |
| `get_memory_by_id` | GET | `/api/memories/{id}` |
| `update_memory` | PATCH | `/api/memories/{id}` |
| `delete_memory` | DELETE | `/api/memories/{id}` |
| `search_memories` | POST | `/api/search` |
| `hybrid_search_memories` | POST | `/api/search/hybrid` |
| `list_recent_memories` | GET | `/api/memories?profile=...&limit=...` |
| `explore_memory_graph` | POST | `/api/graph/explore` |
| `get_related_memories` | POST | `/api/graph/related` |
| `auto_link_memory` | POST | `/api/graph/link/{id}` |
| `get_memory_stats` | GET | `/api/stats/{profile}` |
| `list_profiles` | GET | `/api/profiles` |
| `health_check` | GET | `/api/health` |
| ... | ... | ... |

### Error Handling

- HTTP errors (4xx, 5xx) are caught and re-raised as appropriate exceptions
- Connection errors trigger retries (via the standard `@with_retry` decorator)
- Response JSON is parsed and returned in the same dict format as other backends

### Use Cases

- Multi-user deployments where a central Ogham server serves multiple clients
- Environments where clients can't have direct database access
- Separating the MCP interface from the storage layer
