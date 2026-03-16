# Ogham MCP

*Ogham* (pronounced "OH-um") -- persistent, searchable shared memory for AI coding agents. Works across clients.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io%2Fogham--mcp%2Fogham--mcp-blue)](https://github.com/ogham-mcp/ogham-mcp/pkgs/container/ogham-mcp)

## The problem

AI coding agents forget everything between sessions. Switch from Claude Code to Cursor to Kiro to OpenCode and context is lost. Decisions, gotchas, architectural patterns -- gone. You end up repeating yourself, re-explaining your codebase, re-debugging the same issues.

Ogham gives your agents a shared memory that persists across sessions and clients.

## Quick start

### 1. Install

```bash
uvx ogham-mcp init
```

This runs the setup wizard. It walks you through everything: database connection, embedding provider, schema migration, and writes MCP client configs for Claude Code, Cursor, VS Code, and others.

> **You need a database before running this.** Either create a free [Supabase](https://supabase.com) project or a [Neon](https://neon.tech) database. The wizard handles the rest.

### 2. Add to your MCP client

The wizard configures everything and writes your client config -- including all environment variables the server needs. For Claude Code, it runs `claude mcp add` automatically. For other clients, copy the config snippet it prints.

### 3. Use it

Tell your agent to remember something, then ask about it later -- from the same client or a different one. It works because they all share the same database.

### Manual setup (if you prefer)

If you'd rather configure things yourself instead of using the wizard:

```bash
# Supabase
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-service-role-key
export EMBEDDING_PROVIDER=openai  # or ollama, mistral, voyage
export OPENAI_API_KEY=sk-...      # for your chosen provider

# Or Postgres (Neon, self-hosted)
export DATABASE_BACKEND=postgres
export DATABASE_URL=postgresql://user:pass@host/db
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

Run the schema migration (`sql/schema.sql` for Supabase, `sql/schema_postgres.sql` for Neon/self-hosted), then add the MCP server to your client.

## Installation methods

| Method | Command | When to use |
|--------|---------|-------------|
| **uvx** (recommended) | `uvx ogham-mcp` | Quick setup, auto-updates |
| **Docker** | `docker pull ghcr.io/ogham-mcp/ogham-mcp` | Isolation, self-hosted |
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

## SSE transport (multi-agent)

By default, Ogham runs in stdio mode -- each MCP client spawns its own server process. For multiple agents sharing one server, use SSE mode:

```bash
ogham serve --transport sse --port 8742
```

The server runs as a persistent background process. All clients connect to the same instance -- one database pool, one embedding cache, shared memory.

Client config for SSE (any MCP client):

```json
{
  "mcpServers": {
    "ogham": {
      "url": "http://127.0.0.1:8742/sse"
    }
  }
}
```

Health check at `http://127.0.0.1:8742/health` (cached, sub-10ms).

Configure via env vars (`OGHAM_TRANSPORT=sse`, `OGHAM_HOST`, `OGHAM_PORT`) or CLI flags. The init wizard (`ogham init`) walks through SSE setup if you choose it.

## Entry points

Ogham has two entry points:

- **`ogham`** -- the CLI. Use this for `ogham init`, `ogham health`, `ogham search`, and other commands you run yourself. Running `ogham` with no arguments starts the MCP server.
- **`ogham-serve`** -- starts the MCP server directly. This is what MCP clients should call. When you run `uvx ogham-mcp`, it invokes `ogham-serve`.

## CLI

```bash
ogham init                      # Interactive setup wizard
ogham health                    # Check database + embedding provider
ogham search "query"            # Search memories from the terminal
ogham store "some fact"         # Store a memory
ogham list                      # List recent memories
ogham profiles                  # List profiles and counts
ogham stats                     # Profile statistics
ogham export -o backup.json     # Export memories
ogham import backup.json        # Import memories
ogham cleanup                   # Remove expired memories
ogham serve                     # Start MCP server (stdio, default)
ogham serve --transport sse     # Start SSE server on port 8742
ogham openapi                   # Generate OpenAPI spec
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_BACKEND` | No | `supabase` | `supabase` or `postgres` |
| `SUPABASE_URL` | If supabase | -- | Your Supabase project URL |
| `SUPABASE_KEY` | If supabase | -- | Supabase secret key (service_role) |
| `DATABASE_URL` | If postgres | -- | PostgreSQL connection string |
| `EMBEDDING_PROVIDER` | No | `ollama` | `ollama`, `openai`, `mistral`, or `voyage` |
| `EMBEDDING_DIM` | No | `512` | Vector dimensions -- must match your schema (see below) |
| `OPENAI_API_KEY` | If openai | -- | OpenAI API key |
| `MISTRAL_API_KEY` | If mistral | -- | Mistral API key |
| `VOYAGE_API_KEY` | If voyage | -- | Voyage AI API key |
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
| OpenAI | 512 (schema default) | 0.35 | Set `EMBEDDING_DIM=512` explicitly -- OpenAI defaults to 1024 |
| Ollama | 512 | 0.70 | Tight clustering, scores run 0.8-0.9 |
| Mistral | 1024 | 0.60 | Fixed 1024 dims, can't truncate. Schema must be `vector(1024)` |
| Voyage | 512 (schema default) | 0.45 | Moderate spread |

`EMBEDDING_DIM` must match the `vector(N)` column in your database schema. The default schema uses `vector(512)`. If you use Mistral, you need to alter the column to `vector(1024)` before storing anything.

Each provider clusters vectors differently, so the similarity threshold matters. Start with the recommended value and adjust based on your results.

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
| `hybrid_search` | Combined semantic + full-text search (RRF) | `query`, `limit`, `tags[]`, `graph_depth` |
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
| `current_profile` | Show active profile | -- |
| `list_profiles` | List all profiles with counts | -- |
| `set_profile_ttl` | Set auto-expiry for a profile | `profile`, `ttl_days` |

### Import / export

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `export_profile` | Export all memories in active profile | `format` (`json` or `markdown`) |
| `import_memories_tool` | Import memories with deduplication | `data`, `dedup_threshold` |

### Maintenance

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `re_embed_all` | Re-embed all memories (after switching providers) | -- |
| `compress_old_memories` | Condense old inactive memories (full text to summary to tags) | -- |
| `cleanup_expired` | Remove expired memories (TTL) | -- |
| `health_check` | Check database and embedding connectivity | -- |
| `get_stats` | Memory counts, profiles, activity | -- |
| `get_cache_stats` | Embedding cache hit rates | -- |

## Skills

Ogham ships with three workflow skills in `skills/` that wire up common MCP tool chains. Install them in Claude Code, Cursor, or any client that supports skills.

| Skill | Triggers on | What it does |
|-------|-------------|-------------|
| `ogham-research` | "remember this", "store this finding", "save what we learned" | Checks for duplicates via hybrid_search before storing. Auto-tags with a consistent scheme (`type:decision`, `type:gotcha`, etc.). Uses `store_decision` for architectural choices. |
| `ogham-recall` | "what do I know about X", "find related", "context for this project" | Chains hybrid_search, find_related, and explore_knowledge to surface connections. Bootstraps session context at project start. |
| `ogham-maintain` | "memory stats", "clean up my memory", "export my brain" | Runs health_check, get_stats, cleanup_expired, re_embed_all, link_unlinked. Warns before irreversible operations. |

Skills call existing MCP tools -- they don't replace them. The MCP server must be connected for skills to work.

Install all three with npx:

```bash
npx skills add ogham-mcp/ogham-mcp
```

Or install a specific skill:

```bash
npx skills add ogham-mcp/ogham-mcp --skill ogham-recall
```

Manual install (copy from a local clone):

```bash
cp -r skills/ogham-research skills/ogham-recall skills/ogham-maintain ~/.claude/skills/
```

## Scoring and condensing

Ogham goes beyond storing and retrieving. Three server-side features run automatically, no configuration needed.

**Novelty detection.** When you store a memory, Ogham checks how similar it is to what you already have. Redundant content gets a lower novelty score and ranks quieter in search results. You can still find it, but it won't push out more useful memories.

**Content signal scoring.** Memories that mention decisions, errors, architecture, or contain code blocks get a higher signal score. A debug session where you fixed a real bug ranks above a casual note about a meeting. The scoring is pure regex, no LLM involved.

**Automatic condensing.** Old memories that nobody accesses gradually shrink. Full text becomes a summary of key sentences, then a one-line description with tags. The original is always preserved and can be restored if the memory becomes relevant again. Run `compress_old_memories` manually or on a schedule. High-importance and frequently-accessed memories resist condensing.

## Database setup

Ogham works with Supabase or vanilla PostgreSQL. Run the schema file that matches your setup:

| File | Use case |
|------|----------|
| `sql/schema.sql` | [Supabase](https://supabase.com) Cloud |
| `sql/schema_selfhost_supabase.sql` | Self-hosted Supabase with RLS |
| `sql/schema_postgres.sql` | Vanilla PostgreSQL / [Neon](https://neon.tech) (no RLS) |

Supabase and Neon both include pgvector out of the box -- no extra setup needed. If you're self-hosting Postgres, you need PostgreSQL 15+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed. We develop and test against PostgreSQL 17.

For Postgres, set `DATABASE_BACKEND=postgres` and `DATABASE_URL=postgresql://...` in your environment.

## Architecture

Ogham runs as an MCP server over stdio or SSE. Your AI client connects to it like any other MCP tool.

```
AI Client (Claude Code, Cursor, Kiro, OpenCode, ...)
    |
    | stdio (MCP protocol)
    |
Ogham MCP Server
    |
    | HTTPS (Supabase REST API) or direct connection (Postgres)
    |
PostgreSQL + pgvector
```

Memories are stored as rows with vector embeddings. Search combines pgvector cosine similarity with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF).

The knowledge graph uses a `memory_relationships` table with recursive CTEs for traversal -- no separate graph database.

## Documentation

Full docs and integration guides at [ogham-mcp.dev](https://ogham-mcp.dev).

## Credits

Inspired by [Nate B Jones](https://www.youtube.com/watch?v=2JiMmye2ezg) and his work on persistent AI memory.

Named after [Ogham](https://en.wikipedia.org/wiki/Ogham), the ancient Irish alphabet carved into stone -- the original persistent memory.

## License

MIT
