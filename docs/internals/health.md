# Ogham MCP — Health & HTTP Health

Two modules provide health checking at different layers.

## `health.py` — Dependency Health Check

Checks connectivity to both external dependencies:

### Database Check

- Calls `get_backend().get_memory_stats("default")`
- If it returns successfully, DB is healthy
- Captures any exception as the error message

### Embedding Check

- Calls `generate_embedding("health check probe")`
- Validates returned vector has correct dimensions (`settings.embedding_dim`)
- Reports: provider name, model, dimension, and whether it matches expected dim

### Return Format

```python
{
    "status": "healthy" | "degraded" | "unhealthy",
    "database": {"status": "ok" | "error", "error": "..."},
    "embeddings": {
        "status": "ok" | "error",
        "provider": "ollama",
        "model": "embeddinggemma",
        "dimension": 512,
        "dimension_match": True,
    },
}
```

- `healthy` = both OK
- `degraded` = one failed
- `unhealthy` = both failed

## `http_health.py` — HTTP Health Endpoint

Lightweight HTTP server that runs alongside the MCP server (spawned in a daemon thread by `server.py`).

### Endpoint

`GET /health` on configurable port (default 8765, via `OGHAM_HEALTH_PORT` env var).

### Response

- Calls `check_health()` from `health.py`
- Returns JSON with appropriate HTTP status:
  - `200` for healthy
  - `503` for degraded or unhealthy

### Caching

- Health check results cached for 30 seconds (`_CACHE_TTL`)
- Prevents hammering the database/embedding provider on frequent polls
- Cache stored in module-level `_health_cache` dict with timestamp

### Server Details

- Uses Python's built-in `http.server.HTTPServer`
- `ThreadingHTTPServer` mixin for concurrent requests
- Runs as daemon thread (dies with main process)
- Only serves `/health` — returns 404 for all other paths
- Suppresses request logging (`log_message` is a no-op)

### Usage

Useful for container orchestration (Docker health checks, Kubernetes liveness probes) and monitoring systems that need an HTTP endpoint to poll.
