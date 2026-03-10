# Ogham MCP

Persistent, searchable shared memory for AI coding agents. Works across clients.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io%2Fogham--mcp%2Fogham--mcp-blue)](https://github.com/ogham-mcp/ogham-mcp/pkgs/container/ogham-mcp)

## The problem

AI coding agents forget everything between sessions. Switch from Claude Code to Cursor to OpenCode and context is lost. Decisions, gotchas, architectural patterns — all gone. You end up repeating yourself, re-explaining your codebase, re-debugging the same issues.

Ogham gives your agents a shared memory that persists across sessions and clients.

## Quick start

### 1. Set up Supabase

Create a free project at [supabase.com](https://supabase.com). Run `sql/schema.sql` in the SQL editor.

### 2. Install and configure

```bash
# Claude Code (one command)
claude mcp add ogham -- uvx ogham-mcp

# Set environment variables
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-service-role-key
export EMBEDDING_PROVIDER=openai  # or ollama, mistral, voyage
export OPENAI_API_KEY=sk-...      # for your chosen provider
```

### 3. Use it

Your agent now has persistent memory. Store decisions, search for context, build a knowledge graph — all through natural conversation.

## Installation methods

| Method | Command | Best for |
|--------|---------|----------|
| **uvx** (recommended) | `claude mcp add ogham -- uvx ogham-mcp` | Claude Code, quick setup |
| **Docker** | `docker pull ghcr.io/ogham-mcp/ogham-mcp` | Self-hosted, isolation |
| **Git clone** | `git clone` + `uv sync` | Development, contributions |

### Claude Code

```bash
claude mcp add ogham -- uvx ogham-mcp
```

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "ogham": {
      "type": "local",
      "command": ["uvx", "ogham-mcp"],
      "environment": {
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_KEY": "{env:SUPABASE_KEY}",
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "{env:OPENAI_API_KEY}"
      }
    }
  }
}
```

### Docker

```bash
docker run --rm \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_KEY=your-key \
  -e EMBEDDING_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/ogham-mcp/ogham-mcp
```

### From source

```bash
git clone https://github.com/ogham-mcp/ogham-mcp.git
cd ogham-mcp
uv sync
uv run ogham --help
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | Yes | — | Your Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Supabase service role key |
| `EMBEDDING_PROVIDER` | No | `ollama` | `ollama`, `openai`, `mistral`, or `voyage` |
| `EMBEDDING_DIM` | No | Per provider | Vector dimensions (see below) |
| `OPENAI_API_KEY` | If openai | — | OpenAI API key |
| `MISTRAL_API_KEY` | If mistral | — | Mistral API key |
| `VOYAGE_API_KEY` | If voyage | — | Voyage AI API key |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | No | `embeddinggemma` | Ollama embedding model |
| `MISTRAL_EMBED_MODEL` | No | `mistral-embed` | Mistral embedding model |
| `VOYAGE_EMBED_MODEL` | No | `voyage-4-lite` | Voyage embedding model |
| `DEFAULT_MATCH_THRESHOLD` | No | `0.7` | Similarity threshold (see below) |
| `DEFAULT_MATCH_COUNT` | No | `10` | Max results per search |
| `DEFAULT_PROFILE` | No | `default` | Memory profile name |

### Embedding providers

| Provider | Default dimensions | Recommended threshold | Notes |
|----------|-------------------|----------------------|-------|
| OpenAI | 1024 | 0.35 | Widest spread, scores run 0.5-0.6 |
| Ollama | 512 | 0.70 | Tight clustering, scores run 0.8-0.9 |
| Mistral | 1024 | 0.60 | Tight clustering, similar to Ollama |
| Voyage | 1024 | 0.45 | Moderate spread |

The threshold matters — each provider clusters vectors differently. Start with the recommended value and adjust.

## MCP tools

### Memory operations

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `store_memory` | Store a new memory with embedding | `content` (required), `source`, `tags[]`, `auto_link` |
| `store_decision` | Store an architectural decision | `decision`, `reasoning`, `alternatives[]`, `tags[]` |
| `update_memory` | Update content of existing memory | `memory_id`, `content`, `tags[]` |
| `delete_memory` | Delete a memory by ID | `memory_id` |
| `reinforce_memory` | Increase confidence score | `memory_id` |
| `contradict_memory` | Decrease confidence score | `memory_id` |

### Search

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `hybrid_search` | Combined semantic + full-text search (RRF) | `query`, `limit`, `threshold`, `tags[]` |
| `list_recent` | List recent memories | `limit`, `profile` |
| `find_related` | Find memories related to a given one | `memory_id`, `limit` |

### Knowledge graph

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `link_unlinked` | Auto-link memories by embedding similarity | `threshold`, `limit` |
| `explore_knowledge` | Traverse the knowledge graph | `memory_id`, `depth`, `direction` |

### Profiles

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `switch_profile` | Switch active memory profile | `profile` |
| `current_profile` | Show active profile | — |
| `list_profiles` | List all profiles with counts | — |
| `set_profile_ttl` | Set auto-expiry for a profile | `profile`, `ttl_days` |

### Import / Export

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `export_profile` | Export all memories in active profile | `format` (`json` or `markdown`) |
| `import_memories_tool` | Import memories with deduplication | `data`, `dedup_threshold` |

### Maintenance

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `re_embed_all` | Re-embed all memories (provider switch) | — |
| `cleanup_expired` | Remove expired memories (TTL) | — |
| `health_check` | Check database and embedding connectivity | — |
| `get_stats` | Memory counts, profiles, activity | — |
| `get_cache_stats` | Embedding cache hit rates | — |

## Database setup

Ogham uses Supabase PostgreSQL with pgvector. Run the schema file that matches your setup:

| File | Use case |
|------|----------|
| `sql/schema.sql` | Supabase Cloud (recommended) |
| `sql/schema_selfhost_supabase.sql` | Self-hosted Supabase with RLS |
| `sql/schema_postgres.sql` | Vanilla PostgreSQL / Neon (no RLS) |

## Architecture

Ogham runs as an MCP server over stdio. Your AI client connects to it like any other MCP tool.

```
AI Client (Claude Code, Cursor, OpenCode, ...)
    |
    | stdio (MCP protocol)
    |
Ogham MCP Server
    |
    | HTTPS (Supabase REST API)
    |
PostgreSQL + pgvector
```

Memories are stored as rows with vector embeddings. Search combines pgvector cosine similarity with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF).

The knowledge graph uses a `memory_relationships` table with recursive CTEs for traversal — no separate graph database needed.

## Documentation

Full documentation, architecture deep-dives, and guides at [ogham-mcp.dev](https://ogham-mcp.dev).

## Credits

Inspired by the work of [Nate B Jones](https://www.youtube.com/watch?v=2JiMmye2ezg) on persistent AI memory systems.

Named after [Ogham](https://en.wikipedia.org/wiki/Ogham), the ancient Irish alphabet carved into stone — the original persistent memory.

## License

MIT
