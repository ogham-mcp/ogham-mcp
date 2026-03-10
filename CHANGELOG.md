# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-03-07 — Relationship Graph

### Added
- **Relationship graph** — `memory_relationships` table with typed, weighted edges (`similar`, `supports`, `contradicts`, `related`, `follows`, `derived_from`). Built entirely in PostgreSQL with recursive CTEs — no separate graph database or LLM needed
- **Auto-linking** — new memories are automatically linked to similar existing memories on store via HNSW vector search (threshold 0.85, max 5 links). One database round-trip, no LLM in the write path
- `explore_knowledge` tool — hybrid search seeds + recursive CTE graph traversal. Finds memories by meaning, then expands via relationship edges to pull in connected context
- `find_related` tool — traverses the graph outward from a known memory ID for impact analysis
- `store_decision` tool — stores architectural decisions with structured metadata (rationale, alternatives) and `supports` edges to related memories
- `link_unlinked` tool — backfills auto-links for memories that predate the relationship graph. Configurable threshold and batch size
- `store_memory` gains `auto_link` parameter (default `True`) and returns `links_created` count
- Stress test script (`tests/bench_stress.py`) — imports 1000+ memories, verifies dedup, backfills auto-links, benchmarks graph operations at scale
- Benchmark coverage for auto-link, explore graph, and get related operations

### Fixed
- `link_unlinked_memories` RPC infinite loop — `PERFORM` discarded return value, causing memories with no similar neighbors to be reprocessed indefinitely. Fixed with `SELECT INTO` and conditional increment

### Schema Changes
- Run `sql/migrations/008_memory_relationships.sql`, `009_graph_explorer.sql`, `010_impact_analysis.sql` in order, or re-run `sql/schema.sql` for fresh installs
- New table: `memory_relationships` (source_id, target_id, relationship, strength, metadata)
- New enum: `relationship_type`
- New RPCs: `auto_link_memory`, `link_unlinked_memories`, `explore_memory_graph`, `get_related_memories`

## [0.2.1] - 2026-03-05 — Supabase Best Practices

### Changed
- **RLS hardened** — `FORCE ROW LEVEL SECURITY` on `memories` and `profile_settings`, deny `anon` role access by default
- **Composite index** — replaced separate `profile` and `created_at` indexes with `(profile, created_at DESC)` for faster filtered queries
- **Keyset pagination** — `get_all_memories_full` and `get_all_memories_content` now use cursor-based pagination instead of OFFSET
- **Batch re-embed** — `re_embed_all` uses new `batch_update_embeddings` RPC (batches of 50) instead of individual updates

### Added
- `batch_update_embeddings` RPC function for efficient bulk embedding updates
- `CHECK` constraint on `profile_settings.ttl_days` (`>= 1` or NULL)
- RLS policies on `profile_settings` table

### Schema Changes
- Run `sql/migrate_best_practices.sql` to upgrade existing installs

## [0.2.0] - 2026-03-05 — Quality of Life Release

### Added
- **LRU embedding cache** with configurable max size (`EMBEDDING_CACHE_MAX_SIZE`, default 1000) and `get_cache_stats()` tool for monitoring
- **Memory expiration** with profile-bound TTLs — `set_profile_ttl()` sets days until memories expire, `cleanup_expired()` permanently removes them
- **Profile settings table** (`profile_settings`) for storing per-profile TTL configuration
- **MCP prompts** — `summarize-recent`, `find-decisions`, `profile-overview`, `cleanup-check`
- **HTTP health endpoint** — optional `GET /health` on configurable port (`ENABLE_HTTP_HEALTH`, `HEALTH_PORT`)
- **CLI tool** — `ogham` command with subcommands: serve, health, profiles, stats, search, list, cleanup, export, import, openapi
- **OpenAPI spec generation** — `ogham openapi` generates `docs/openapi.json` from MCP tool definitions
- **Export/import** — `export_profile()` exports as JSON or Markdown, `import_memories_tool()` imports with dedup

### Changed
- `match_memories`, `get_profile_counts`, `get_memory_stats_sql` RPCs now filter expired memories
- `list_recent_memories` now filters expired memories and returns `expires_at`
- `store_memory` now computes `expires_at` from profile TTL when set
- Entry point changed: `ogham` now runs CLI (use `ogham serve` for MCP server, or `ogham-serve`)
- Embedding cache changed from unbounded dict to OrderedDict-based LRU

### Schema Changes
- New table: `profile_settings` (profile, ttl_days, created_at, updated_at)
- New column: `memories.expires_at` (timestamptz, nullable)
- New RPCs: `cleanup_expired_memories()`, `count_expired_memories()`
- Run `sql/migrate_expiration.sql` to upgrade existing installs

## [0.1.1] - 2026-03-05

### Added
- `health_check` tool for diagnosing connection issues
- Startup validation -- exits with clear errors if Supabase or Ollama aren't reachable
- Retry with exponential backoff on transient failures (embedding: 3 attempts, DB reads: 2)
- Input validation for content, threshold, and limit parameters
- Embedding cache (SHA256-keyed) to skip re-embedding identical content
- SQL aggregation functions (`get_profile_counts`, `get_memory_stats_sql`) replacing Python-side counting
- Progress reporting for `re_embed_all`
- Tool execution timing logged to stderr

### Fixed
- `update_memory` / `store_memory` crash when Supabase returns empty results
- `delete_memory` / `update_memory` could affect memories in other profiles

### Changed
- `get_stats()` returns `total` instead of `total_memories`
- OpenAI is now an optional dependency (`uv sync --extra openai`)
- New RPC functions required -- re-run `sql/schema.sql`

## [0.1.0] - 2026-03-05

### Added
- Initial release
- MCP server with stdio transport
- Supabase PostgreSQL + pgvector backend
- Ollama and OpenAI embedding providers
- Memory profiles (Severance-style partitioning)
- Semantic search with cosine similarity
- Tag and source filtering
- Docker and native (UV) deployment options
- Support for Claude Desktop, Cursor, Claude Code, VS Code, Codex, Windsurf
