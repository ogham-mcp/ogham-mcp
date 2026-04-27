# Ogham MCP

*Ogham* (pronounced "OH-um") -- persistent, searchable shared memory for AI coding agents. Works across clients.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io%2Fogham--mcp%2Fogham--mcp-blue)](https://github.com/ogham-mcp/ogham-mcp/pkgs/container/ogham-mcp)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/ogham-mcp)](https://pypi.org/project/ogham-mcp/)

## Contents

- [Retrieval quality](#retrieval-quality) -- 97.2% R@10 on LongMemEval
- [The problem](#the-problem)
- [Quick start](#quick-start)
- [Installation methods](#installation-methods) -- Claude Code, OpenCode, Docker, source
- [SSE transport](#sse-transport-multi-agent) -- multi-agent setup
- [CLI](#cli) -- command-line interface
- [Configuration](#configuration) -- env vars, embedding providers, temporal search, lifecycle hooks
- [MCP tools](#mcp-tools) -- memory, search, graph, profiles, import/export
- [Wiki layer](#wiki-layer) -- synthesize topics, walk the graph, lint health
- [Obsidian export](#obsidian-export) -- snapshot your wiki to a vault of plain markdown
- [Skills](#skills) -- ogham-research, ogham-recall, ogham-maintain
- [Scoring and condensing](#scoring-and-condensing)
- [Cross-encoder reranking](#cross-encoder-reranking) -- optional FlashRank for self-hosters
- [ONNX local embeddings](#onnx-local-embeddings) -- BGE-M3 dense + sparse, no API costs
- [Database setup](#database-setup) -- Supabase, Neon, vanilla Postgres
  - [Upgrading from v0.4.x](#upgrading-from-v04x)
- [Architecture](#architecture)

## Retrieval quality

**85.8% QA accuracy on the [AMB benchmark harness](https://github.com/vectorize-io/agent-memory-benchmark)** (500 questions, April 2026) -- 429/500 questions answered correctly using GPT-5-mini with reasoning, evaluated by Gemini 2.5 Flash Lite as a strict judge. Retrieval R@10: 99.5%. AMB is the standardised evaluation harness built by the [Vectorize](https://vectorize.io) team (creators of Hindsight). Thanks to Nicolo and the Vectorize team for making the harness open.

Previously: 91.8% on our internal LongMemEval benchmark pipeline (gpt-5.4-mini reader, rubric judge). The AMB number is lower because AMB uses a stricter substring-matching judge -- see the [full write-up](https://ogham-mcp.dev/blog/longmemeval-92/) for methodology differences.

**0.554 nugget score on [BEAM](https://arxiv.org/abs/2510.27246) 100K** (400 questions across 10 memory abilities, ICLR 2026), using the paper's exact judge prompt from Appendix G. The published baseline is 0.358 (Llama-4-Maverick + LIGHT). Retrieval R@10: 0.737. Seven of nine categories beat the paper. [Full write-up](https://ogham-mcp.dev/blog/beam-benchmark-v090/).

**End-to-end QA accuracy** on LongMemEval (retrieval + LLM reads and answers):

| System | Accuracy | Architecture |
|--------|----------|-------------|
| [OMEGA](https://dev.to/singularityjason/how-i-built-a-memory-system-that-scores-954-on-longmemeval-1-on-the-leaderboard-2md3) | 95.4% | Classification + extraction pipeline |
| [Observational Memory (Mastra)](https://mastra.ai/research/observational-memory) | 94.9% | Observation extraction + GPT-5-mini |
| **Ogham v0.9.2** | **85.8%** | Verbatim + read-time extraction + gpt-5-mini (AMB harness, strict judge) |
| Ogham v0.9.1 | 91.8% | Hybrid search + context engineering + gpt-5.4-mini (internal benchmark) |
| [Hindsight (Vectorize)](https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision) | 91.4% | 4 memory types + Gemini-3 |
| [Zep (Graphiti)](https://blog.getzep.com/state-of-the-art-agent-memory/) | 71.2% | Temporal knowledge graph + GPT-4o |
| [Mem0](https://mem0.ai) | 49.0% | RAG-based |

**Retrieval only** (R@10 -- no LLM in the search loop):

| System | R@10 | Architecture |
|--------|------|-------------|
| **Ogham** | **97.2%** | 1 SQL query (pgvector + tsvector CCF hybrid search) |
| [LongMemEval paper](https://arxiv.org/abs/2410.10813) baseline | 78.4% | Session decomposition + fact-augmented keys |

Other retrieval systems that report similar R@10 numbers typically use cross-encoder reranking, NLI verification, knowledge graph enrichment, and LLM-as-a-judge pipelines. Ogham reaches 97.2% with one Postgres query. Optional [FlashRank reranking](#cross-encoder-reranking) is available for self-hosters who want extra ranking precision.

These tables measure different things. QA accuracy tests whether the full system (retrieval + LLM) produces the correct answer. R@10 tests whether retrieval alone finds the right memories. Ogham is a retrieval engine -- it finds the memories, your LLM reads them.

| Category | R@10 | Questions |
|----------|------|-----------|
| single-session-assistant | 100% | 56 |
| knowledge-update | 100% | 78 |
| single-session-user | 98.6% | 70 |
| multi-session | 97.3% | 133 |
| single-session-preference | 96.7% | 30 |
| temporal-reasoning | 93.5% | 133 |

Full breakdown: [ogham-mcp.dev/features](https://ogham-mcp.dev/features/#retrieval-quality)

## The problem

AI coding agents forget everything between sessions. Switch from Claude Code to Cursor to Kiro to OpenCode and context is lost. Decisions, gotchas, architectural patterns -- gone. You end up repeating yourself, re-explaining your codebase, re-debugging the same issues.

Ogham gives your agents a shared memory that persists across sessions and clients.

## Quick start

### 1. Install

```bash
uvx --from ogham-mcp ogham init
```

This runs the setup wizard. It walks you through everything: database connection, embedding provider, schema migration, and writes MCP client configs for Claude Code, Cursor, VS Code, and others.

> **You need a database before running this.** Either create a free [Supabase](https://supabase.com) project or a [Neon](https://neon.tech) database. The wizard handles the rest.

> **Using Neon or self-hosted Postgres?** Install with the postgres extra so the driver is available:
> ```bash
> uvx --from 'ogham-mcp[postgres]' ogham init
> ```

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
ogham config                    # Show runtime configuration (secrets masked)
ogham store "some fact"         # Store a memory
ogham search "query"            # Search memories (hybrid: semantic + keyword)
ogham search "q" --json         # JSON output for scripting
ogham search "q" --tags "a,b"   # Filter by comma-separated tags
ogham list                      # List recent memories
ogham list --json               # JSON output
ogham delete <id>               # Delete a memory by ID
ogham use <profile>             # Switch default profile
ogham profiles                  # List profiles and counts
ogham stats                     # Profile statistics
ogham export -o backup.json     # Export memories
ogham import backup.json        # Import memories
ogham cleanup                   # Remove expired memories
ogham hooks install             # Auto-detect client + configure hooks
ogham hooks recall              # Read from the stone (load project context)
ogham hooks inscribe            # Carve into the stone (capture activity)
ogham serve                     # Start MCP server (stdio, default)
ogham serve --transport sse     # Start SSE server on port 8742
ogham openapi                   # Generate OpenAPI spec
```

### Multi-profile search

Search across multiple profiles in a single query (v0.8.5+):

```python
# MCP tool
hybrid_search(query="architecture decisions", profiles=["work", "shared"])

# Python library
from ogham.service import search_memories_enriched
results = search_memories_enriched(
    query="architecture decisions",
    profile="work",
    profiles=["work", "shared", "project-alpha"],
)
```

When `profiles` is set, results include memories from all listed profiles with a `profile` field showing which profile each result came from.

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_BACKEND` | No | `supabase` | `supabase` or `postgres` |
| `SUPABASE_URL` | If supabase | -- | Your Supabase project URL |
| `SUPABASE_KEY` | If supabase | -- | Supabase secret key (service_role) |
| `DATABASE_URL` | If postgres | -- | PostgreSQL connection string |
| `EMBEDDING_PROVIDER` | No | `ollama` | `ollama`, `openai`, `mistral`, `voyage`, `gemini`, or `onnx` |
| `EMBEDDING_DIM` | No | `512` | Vector dimensions -- must match your schema (see below) |
| `OPENAI_API_KEY` | If openai | -- | OpenAI API key |
| `MISTRAL_API_KEY` | If mistral | -- | Mistral API key |
| `VOYAGE_API_KEY` | If voyage | -- | Voyage AI API key |
| `GEMINI_API_KEY` | If gemini | -- | Google Gemini API key |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | No | `embeddinggemma` | Ollama embedding model |
| `MISTRAL_EMBED_MODEL` | No | `mistral-embed` | Mistral embedding model |
| `VOYAGE_EMBED_MODEL` | No | `voyage-4-lite` | Voyage embedding model |
| `GEMINI_EMBED_MODEL` | No | `gemini-embedding-2-preview` | Gemini embedding model |
| `RERANK_ENABLED` | No | `false` | Enable FlashRank cross-encoder reranking |
| `RERANK_ALPHA` | No | `0.55` | Cross-encoder score weight (0-1) |
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
| Gemini | 512 | 0.35 | `gemini-embedding-2-preview`, supports MRL truncation |
| ONNX | 1024 | 0.35 | Local BGE-M3 inference, dense + sparse vectors. See [ONNX section](#onnx-local-embeddings) |

`EMBEDDING_DIM` must match the `vector(N)` column in your database schema. The default schema uses `vector(512)`. If you use Mistral, you need to alter the column to `vector(1024)` before storing anything.

Each provider clusters vectors differently, so the similarity threshold matters. Start with the recommended value and adjust based on your results.

### Temporal search

Search queries with time expressions like "last week" or "three months ago" are resolved automatically using parsedatetime -- no configuration needed. This handles roughly 80% of temporal queries at zero cost.

For expressions that parsedatetime cannot parse ("the quarter before last", "around Thanksgiving"), set `TEMPORAL_LLM_MODEL` to call an LLM as a fallback:

```bash
# Self-hosted with Ollama (free, local)
TEMPORAL_LLM_MODEL=ollama/llama3.2

# Cloud API
TEMPORAL_LLM_MODEL=gpt-4o-mini
```

Any [litellm](https://docs.litellm.ai/docs/providers)-compatible model string works -- `deepseek/deepseek-chat`, `moonshot/moonshot-v1-8k`, etc. The LLM is only called when parsedatetime fails and the query has temporal intent, so costs stay near zero.

If `TEMPORAL_LLM_MODEL` is empty (the default), parsedatetime handles everything on its own. Requires the `litellm` package (`pip install litellm` or install Ogham with the appropriate extra).

### Lifecycle hooks

Ogham hooks inject memory context at session start and preserve it across compaction. Install for your client:

```bash
ogham hooks install
```

| Client | What gets installed |
|--------|-------------------|
| Claude Code | Hooks in `~/.claude/settings.json` (recall on SessionStart/PostCompact, inscribe on PostToolUse/PreCompact) |
| Kiro | Instructions for Hook UI (recall on Prompt Submit, inscribe on Agent Stop) |
| Codex, Cursor, others | Project instruction file (CLAUDE.md, AGENTS.md, or .cursorrules) |

**Two commands, named after the Ogham stones:**

- **recall** -- read from the stone. Searches Ogham for memories relevant to your project and injects them as context. Fires at session start and after compaction.
- **inscribe** -- carve into the stone. Captures meaningful tool activity as memories. Skips noise (`ls`, `cat`, `git status`) and only stores signal (commits, deploys, errors, config changes). Fires after tool use and before compaction. Secrets are masked before storing.

**Smart filtering:** Hooks don't capture everything. Routine commands (`ls`, `pwd`, `git add`) are skipped. Only signal events (errors, deployments, commits, config changes) are stored -- typically 20-30 memories per session instead of hundreds.

**Secret masking:** API keys, tokens, passwords, and JWTs are automatically replaced with `***MASKED***` before storing. The event is captured ("configured Stripe API key") but the actual secret never touches the database.

## MCP tools

### Memory operations

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `store_memory` | Store a new memory with embedding | `content` (required), `source`, `tags[]`, `auto_link` |
| `store_decision` | Store an architectural decision | `decision`, `reasoning`, `alternatives[]`, `tags[]` |
| `store_preference` | Store a user preference with strength metadata | `preference`, `subject`, `alternatives[]`, `strength` |
| `store_fact` | Store a factual statement with confidence and citation | `fact`, `subject`, `confidence`, `source_citation` |
| `store_event` | Store an event with temporal and participant metadata | `event`, `when`, `participants[]`, `location` |
| `update_memory` | Update content of existing memory | `memory_id`, `content`, `tags[]` |
| `delete_memory` | Delete a memory by ID | `memory_id` |
| `reinforce_memory` | Increase confidence score | `memory_id` |
| `contradict_memory` | Decrease confidence score | `memory_id` |

### Search

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `hybrid_search` | Combined semantic + full-text search (RRF) | `query`, `limit`, `tags[]`, `graph_depth`, `profiles[]`, `extract_facts` |
| `list_recent` | List recent memories | `limit`, `profile` |
| `find_related` | Find memories related to a given one | `memory_id`, `limit` |

### Knowledge graph

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `link_unlinked` | Auto-link memories by embedding similarity | `threshold`, `limit` |
| `explore_knowledge` | Traverse the knowledge graph | `memory_id`, `depth`, `direction` |
| `suggest_connections` | Find hidden connections via shared entities | `memory_id`, `min_shared_entities`, `limit` |

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
| `get_config` | Show runtime configuration with masked secrets | -- |
| `get_stats` | Memory counts, sources, tags, and profile health (orphans, decay, tagging) | -- |
| `get_cache_stats` | Embedding cache hit rates | -- |

## Wiki layer

The wiki layer turns a tag full of related memories into a synthesized markdown page. Run `compile_wiki` on a tag, get back a summarized topic; the cache invalidates automatically when underlying memories change. Four MCP tools cover the lifecycle:

| Tool | Description | Key parameters |
|------|-------------|----------------|
| `compile_wiki` | Compile a tag's memories into a synthesized markdown page (LLM call, cached) | `topic`, `provider`, `model`, `force` |
| `query_topic_summary` | Read the cached page for a topic without recomputing | `topic` |
| `walk_knowledge` | Direction-aware graph walk from a known memory along relationship edges | `start_id`, `depth`, `direction` (`outgoing`, `incoming`, `both`), `min_strength`, `relationship_types` |
| `lint_wiki` | Health report: contradictions, orphans, stale lifecycle, stale summaries, summary drift | `stable_days`, `sample_size`, `include_drift` |

**The wiki layer needs an LLM.** Synthesis is the LLM step that turns a list of memories into a coherent page; embeddings alone aren't enough. You can run that LLM **locally** (Ollama with `llama3.2`, vLLM, or any OpenAI-compatible local server) or **in the cloud** (Gemini, OpenAI, Anthropic, Mistral, Groq, OpenRouter). Local keeps everything private and free; cloud generally writes more polished prose at the cost of a few cents per compile.

Set the default with `LLM_PROVIDER` and `LLM_MODEL` in your environment (e.g. `LLM_PROVIDER=gemini` + `LLM_MODEL=gemini-2.5-flash`, or `LLM_PROVIDER=ollama` + `LLM_MODEL=llama3.2`). Override per call with `compile_wiki(topic=..., provider=..., model=...)`. The provider/model is stamped into the resulting page's frontmatter, so you can re-compile the same topic with a different LLM and see how the synthesis changes.

`compile_wiki` short-circuits when the source memories haven't changed since the last compile -- the call is effectively free if nothing has moved. Pass `force=True` to bypass that check (useful for re-compiling with a different model on the same source set).

The wiki layer requires migrations 028, 030, and 031 applied to your database. See [Database setup](#database-setup) for details.

## Obsidian export

Snapshot your wiki layer to a folder of Obsidian-compatible markdown files. One `.md` per topic with full YAML frontmatter, plus a `README.md` index. Wikilinks between topics are auto-detected and wrapped in `[[brackets]]` for Obsidian's graph view.

```bash
ogham export-obsidian /path/to/vault
ogham export-obsidian /path/to/vault --profile work --force
```

The export is read-only -- it writes files but never reads them back. Edits in Obsidian stay in Obsidian; re-run the export to refresh the snapshot. By design the exporter refuses to write into a directory that already contains files it didn't create; pass `--force` to override that guardrail.

Full guide with frontmatter reference, troubleshooting, and screenshots: [obsidian export docs](https://ogham-mcp.dev/docs/obsidian-export/).

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

## Entity enrichment

Every memory is automatically enriched at ingest with structured entity tags -- no LLM calls, pure regex and dictionary matching across 18 languages.

**Six entity categories.** Events (wedding, concert, meeting), activities (hiking, coding, cooking), emotions (frustrated, happy, relieved), relationships (sister, boss, colleague), quantities (3 books, 5 miles), and locations (Berlin, Tokyo -- via GeoNames database).

**18 languages.** English, German, French, Spanish, Italian, Portuguese, Brazilian Portuguese, Dutch, Polish, Russian, Ukrainian, Turkish, Arabic, Hindi, Japanese, Korean, Chinese, and Irish. Each language includes common inflected forms (case endings, verb tenses, lenition) so "svadʹbu" matches "svadʹba" in Russian and "bhainis" matches "bainis" in Irish.

**Timeline table.** Search results include a chronological timeline with pre-computed "days ago" and memory ID cross-references. Helps LLM readers answer temporal questions without doing date arithmetic.

**Lost in the Middle reordering.** Search results are reordered so the highest-relevance memories appear at the start and end of the context, where LLMs pay the most attention (Liu et al., 2023).

## Cross-encoder reranking

Optional FlashRank cross-encoder reranking for self-hosters who want better ranking precision. Adds ~300ms per search on CPU.

After Ogham's hybrid search returns candidates, FlashRank (ms-marco-MiniLM-L-12-v2, 21MB) rescores each result against the query using deeper token-level attention. The final score blends retrieval ranking with cross-encoder ranking.

**BEAM benchmark impact:** R@10 0.69 → 0.70, MRR +8pp. Biggest gain: temporal reasoning 0.84 → 0.98. [Full results](https://ogham-mcp.dev/blog/flashrank-reranking/).

Install and enable:

```bash
pip install ogham-mcp[rerank]
# or: uv add ogham-mcp[rerank]

export RERANK_ENABLED=true
export RERANK_ALPHA=0.55   # 55% cross-encoder, 45% retrieval score
```

The model downloads on first use (~21MB). Self-hosters who want speed over precision leave it off (the default).

## ONNX local embeddings

Run BGE-M3 locally with ONNX Runtime -- dense and sparse vectors in a single model pass, no API calls, no GPU required. Contributed by [@ninthhousestudios](https://github.com/ninthhousestudios).

The ONNX provider produces 1024-dim dense vectors plus neural sparse vectors. When sparse vectors are available, Ogham automatically uses three-signal Reciprocal Rank Fusion (dense + FTS + sparse) instead of the default two-signal path.

Install and configure:

```bash
pip install ogham-mcp[onnx]

# Download the model (~2.2GB)
ogham download-model bge-m3

export EMBEDDING_PROVIDER=onnx
export EMBEDDING_DIM=1024
```

Your database schema must use `vector(1024)` for the embedding column. Performance on CPU: ~0.3s per short text, ~10s for long documents (5K+ chars). RSS: ~4.3GB peak.

The ONNX provider is designed for self-hosters who want zero API costs. Cloud users should use Gemini or Voyage for lower latency.

## Database setup

Ogham works with Supabase or vanilla PostgreSQL. Run the schema file that matches your setup:

| File | Use case |
|------|----------|
| `sql/schema.sql` | [Supabase](https://supabase.com) Cloud |
| `sql/schema_selfhost_supabase.sql` | Self-hosted Supabase with RLS |
| `sql/schema_postgres.sql` | Vanilla PostgreSQL / [Neon](https://neon.tech) (no RLS) |

Supabase and Neon both include pgvector out of the box -- no extra setup needed. If you're self-hosting Postgres, you need PostgreSQL 15+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed. We develop and test against PostgreSQL 17.

For Postgres, set `DATABASE_BACKEND=postgres` and `DATABASE_URL=postgresql://...` in your environment.

### Upgrading an existing Ogham database

**For v0.10.x → v0.11.0 (memory lifecycle release):**
see the dedicated guide at [UPGRADING.md](UPGRADING.md). The short version:

```bash
./sql/upgrade.sh $DATABASE_URL     # applies 025 + 026 + 027 idempotently
```

Fresh installers do NOT need this -- `sql/schema.sql` already reflects
the post-v0.11.0 state.

**For older versions (v0.4.x through v0.10.x):**
the same `upgrade.sh` script walks through every migration in order --
temporal columns, halfvec compression, sparse embeddings, RRF/BM25
search, then the v0.11.0 lifecycle additions. Each is idempotent.

```bash
# Postgres / Neon (psql required)
./sql/upgrade.sh $DATABASE_URL

# Supabase: paste migration files into the SQL Editor in order
#   (025_memory_lifecycle.sql → 026_memory_lifecycle_split.sql → 027_audit_log_backfill.sql
#   are the v0.11.0 set)
```

Selected migration highlights:
- **016** adds the `sparse_embedding` column for ONNX BGE-M3 sparse vectors.
- **017** upgrades the search function to true Reciprocal Rank Fusion
  with length-normalised keyword scoring ([Cormack et al., 2009](https://doi.org/10.1145/1571941.1572114)).
- **025 / 026** add the memory lifecycle table + triggers (see [UPGRADING.md](UPGRADING.md)).
- **027** backfills the `audit_log` table for installs that predate it.

**Rollback scripts** live under `sql/migrations/rollback/` with a
`DANGER_` prefix and require explicit session-variable opt-in before
they do anything. See `sql/migrations/rollback/README.md`.

All migrations are idempotent -- safe to re-run. The upgrade script checks your pgvector version and skips halfvec if pgvector is below 0.7.0.

New installs don't need migrations -- the schema files already include everything.

### Upgrading the CLI (uv tool)

uv caches aggressively. A plain `uv tool install ogham-mcp` after a new release may install the old version. Use `--refresh` to force a fresh resolve from PyPI:

```bash
uv tool uninstall ogham-mcp
uv cache clean
uv tool install --refresh "ogham-mcp[gemini,postgres]"
```

Verify the installed version:

```bash
ogham config   # Shows version and provider at the top
```

If you still see the old version, nuke the tool environment directory and retry:

```bash
rm -rf ~/.local/share/uv/tools/ogham-mcp
uv tool install --refresh "ogham-mcp[gemini,postgres]"
```

This is a [known uv caching behaviour](https://docs.astral.sh/uv/concepts/cache/) -- the resolver cache is separate from the package cache and survives `uv cache clean` without `--refresh`.

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

Memories are stored as rows with vector embeddings. Search combines pgvector cosine similarity with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF) -- position-based, score-agnostic fusion that handles different score scales correctly. Optional FlashRank cross-encoder reranking adds a second pass for self-hosters. The Supabase backend uses `postgrest-py` directly (not the full Supabase SDK) for a lightweight dependency footprint.

The knowledge graph uses a `memory_relationships` table with recursive CTEs for traversal -- no separate graph database.

## Research foundations

Ogham's retrieval pipeline combines established information retrieval and cognitive science techniques:

- **Hybrid search** -- Reciprocal Rank Fusion ([Cormack, Clarke & Butt, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)) combining dense vector similarity (pgvector) with BM25-style keyword matching (PostgreSQL tsvector). Two independent retrieval systems, rank-fused without score normalisation.

- **Entity overlap boost** -- memories sharing named entities with the query receive a bounded relevance boost (up to 1.4x), inspired by entity-linking literature ([Kolitsas et al., CoNLL 2018](https://aclanthology.org/K18-1050/)). Entity extraction covers 18 languages via YAML-based word lists with no LLM in the write path.

- **Matryoshka embeddings** -- flexible dimensionality via Matryoshka Representation Learning ([Kusupati et al., NeurIPS 2022](https://arxiv.org/abs/2205.13147)). Embedding providers (OpenAI, Voyage, Gemini, Ollama) produce native-dimension vectors truncated to 512d, enabling provider-portable storage without re-embedding.

- **Temporal diversity re-ranking** -- density-gated soft penalty preventing semantic clustering on a single time period, extending Maximal Marginal Relevance principles ([Carbonell & Goldstein, SIGIR 1998](https://dl.acm.org/doi/10.1145/290941.291025)). Only activates when the top-k results are temporally concentrated, leaving well-distributed results untouched.

- **ACT-R importance scoring** -- cognitive-architecture-inspired memory weighting based on recency, access frequency, and surprise ([Anderson & Lebiere, 1998](https://act-r.psy.cmu.edu/about/)). Frequently accessed memories stay sharp, rarely accessed ones fade, disputed ones drop in ranking without deletion.

- **Hebbian decay and potentiation** -- memories that are not accessed lose importance over time (5% per 30-day idle period). Memories accessed 10+ times become "potentiated" with a slower decay rate (1% per 30 days), simulating long-term potentiation. Based on Hebb's learning rule ([Hebb, 1949](https://doi.org/10.4324/9781315735368)) and computational models of synaptic plasticity ([Bi & Poo, 2001](https://doi.org/10.1146/annurev.neuro.24.1.139)). Importance serves as a multiplier in the relevance formula -- decayed memories sink in rankings but remain retrievable (floor at 0.05). Original importance is preserved in metadata for recovery. Run as a batch job via `ogham decay` or pg_cron.

- **Memory lifecycle (v0.11.0): FRESH / STABLE / EDITING.** Every memory
  now has an explicit stage tracked in a dedicated `memory_lifecycle`
  table. New memories land at `fresh`. The session-start hook sweeps
  aged fresh memories to `stable` when they clear an importance-or-surprise
  gate and have dwelled long enough. Retrieval opens a 30-minute
  `editing` window on the returned memories so follow-up
  `update_memory` calls refine recent thoughts in place; windows
  auto-close on the next sweep. Memories retrieved together also
  strengthen their pairwise graph edges (eta=0.01 per co-retrieval,
  capped at 1.0). The design draws on three lines of prior art:
  Hebbian co-activation ([Hebb, 1949](https://doi.org/10.4324/9781315735368)),
  the hybrid exponential-then-power-law forgetting curve characterised
  by [Wixted (2004)](https://doi.org/10.1146/annurev.psych.55.090902.141555)
  building on [Ebbinghaus (1885)](https://psychclassics.yorku.ca/Ebbinghaus/memory1.htm),
  and the memory reconsolidation window from neuroscience
  ([Nader, Schafe & LeDoux, 2000](https://doi.org/10.1038/35021052))
  for the editing-on-retrieval mechanic. Stage state lives in its
  own table so transitions do not touch the HNSW vector index.

- **Spreading activation** -- when a search hits one memory, activation spreads along relationship edges to pull in connected memories that wouldn't have matched on their own. Integrated into cross-reference, ordering, and summary queries. Density-adaptive weighting means sparse graphs lean harder on graph signal, dense graphs rely more on retrieval score. Inspired by Collins & Loftus ([1975](https://doi.org/10.1037/0033-295X.82.6.407)) semantic network theory.

- **Contradiction detection** -- when a new memory has opposite polarity to a high-similarity existing memory, Ogham automatically creates a `contradicts` relationship edge. Polarity detection uses negation markers across 18 languages loaded from YAML word lists. Contradicted memories are not deleted -- the edge records that the newer memory superseded the older one.

- **Read-time fact extraction** -- query-aware extraction at retrieval time preserves verbatim storage for auditability, contrasting with write-time compression approaches. Verbatim storage ensures the ground truth is always available for re-extraction with different questions later -- a design choice informed by alignment considerations in persistent agent memory ([Anthropic, arXiv:2510.05179](https://arxiv.org/abs/2510.05179)). Supports local models via Ollama for full data sovereignty.

- **Append-only audit trails** -- every store, search, delete, and update operation is logged to an `audit_log` table in the same Postgres instance. Designed for GDPR Article 15 subject access requests and cost governance. Fields align with [OTEL GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/). Query via `ogham audit` CLI. No extra infrastructure -- runs in the same database as memories.

## Documentation

Full docs and integration guides at [ogham-mcp.dev](https://ogham-mcp.dev).

## Credits

Inspired by [Nate B Jones](https://www.youtube.com/watch?v=2JiMmye2ezg) and his work on persistent AI memory.

Named after [Ogham](https://en.wikipedia.org/wiki/Ogham), the ancient Irish alphabet carved into stone -- the original persistent memory.

## License

MIT
