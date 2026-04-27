# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.12.0] - 2026-04-27 -- Wiki Tier 1 + Obsidian export

Compile your memory into synthesized wiki pages, then export them as
Obsidian-compatible markdown.

### New: wiki layer

Four new MCP tools turn your memory into a navigable wiki:

- **`compile_wiki(topic)`** -- LLM-synthesizes all memories carrying a tag
  into one markdown page. Cached so repeat calls are free until sources
  change. Pass `provider="gemini"` / `model="gemini-2.5-flash"` (or
  `provider="ollama"`, `provider="openai"`, etc.) to override the default
  LLM. `force=True` re-compiles against a different model on the same
  source set.
- **`query_topic_summary(topic)`** -- read the cached page. No LLM cost.
- **`walk_knowledge(start_id, depth, direction)`** -- direction-aware
  graph walk along memory relationships (`outgoing`, `incoming`, or
  `both`). Cycle-safe.
- **`lint_wiki()`** -- maintenance health report covering five
  categories: contradictions, orphans, stale lifecycle, stale summaries,
  summary drift.

The wiki layer needs an LLM. You can run that locally (Ollama with
`llama3.2`, vLLM) or in the cloud (Gemini, OpenAI, Anthropic, Mistral,
Groq, OpenRouter). Set `LLM_PROVIDER` and `LLM_MODEL` in
`~/.ogham/config.env`.

### New: Obsidian export

```bash
ogham export-obsidian /path/to/vault
```

Snapshots your wiki layer to a folder of plain markdown files with full
YAML frontmatter, auto-detected wikilinks, and a README index. Open it
in Obsidian -- or any text editor. Read-only: edits in Obsidian stay in
Obsidian; re-run to refresh.

### Migrations

Four new migrations. Apply in order via Supabase SQL Editor (or psql for
self-hosters):

1. `src/ogham/sql/migrations/028_topic_summaries.sql` -- new
   `topic_summaries` table for compiled wiki pages.
2. `src/ogham/sql/migrations/030_topic_summaries_dim_agnostic.sql` --
   aligns the embedding column dimension to `memories.embedding`.
   Required for non-512-dim deployments.
3. `src/ogham/sql/migrations/031_wiki_rpc_functions.sql` -- 14 RPC
   functions powering the wiki tools on Supabase. SQL hardened against
   HNSW planner pessimisation, recursive-CTE cycle blow-up, and concurrent-write
   races on the upsert path.
4. `src/ogham/sql/migrations/032_topic_summaries_rls_policy.sql` --
   `Deny anon access` RLS policy on the two new tables, mirrors the
   pattern on `memories` / `memory_relationships`. Idempotent;
   non-Supabase self-hosters skip cleanly.

### Fixed

- LLM provider keys (Gemini, OpenAI, Anthropic, etc.) now read from
  `~/.ogham/config.env` correctly when launched as an MCP server.
  Previously required exporting them in the parent shell.
- `source_hash` field in wiki responses no longer null on Supabase.

## [0.11.1] - 2026-04-24 -- Security + performance patch

### Security

- **Migration 029 -- `SET search_path` on trigger functions.** The two trigger
  functions shipped in migration 026 (`init_memory_lifecycle`,
  `sync_memory_lifecycle_profile`) did not set an explicit search_path, which
  Supabase's linter flags as "Function Search Path Mutable." Without an
  explicit path, a user with CREATE privilege on any schema earlier on the
  function's search_path could shadow `now()`, the `memory_lifecycle` table,
  or other referenced identifiers and hijack every memories INSERT and
  profile UPDATE. Migration 029 adds `SET search_path = public, pg_catalog`
  to both functions via `CREATE OR REPLACE FUNCTION`. Idempotent; the
  existing triggers auto-pick-up the new definition. The source of
  migration 026 has also been patched so fresh installs get the hardened
  form inline.

  If you run Ogham on Supabase, apply migration 029 by pasting
  `src/ogham/sql/migrations/029_function_search_path.sql` into
  Dashboard -> SQL Editor.

### Performance

- **`PostgresBackend.store_memories_batch` now does one multi-row INSERT.**
  The method was named "batch" but the body looped `cur.execute()` per row.
  Rewritten to a single multi-row `VALUES` INSERT. Measured 2.4x faster
  on real ingest workload; benchmark reruns that previously took hours now
  take minutes. Same rows in, same uuids + timestamps out. A new regression
  test (`tests/test_postgres_batch_ingest.py`) asserts the single-execute
  invariant so this can't silently return to per-row form.

## [0.11.0] - 2026-04-23 -- Memory lifecycle (FRESH / STABLE / EDITING)

### Added

- **Memory lifecycle state machine.** Every memory now has an explicit stage
  (FRESH / STABLE / EDITING). New memories land at `fresh`. The session-start
  hook promotes aged fresh memories to `stable` when they clear an
  importance-or-surprise gate. Retrieving a memory opens a 30-minute
  `editing` window so a follow-up `update_memory` call refines in place;
  windows auto-close on the next sweep. No new API surface the agent has to
  learn -- transitions are automatic.
- **Hebbian edge strengthening on co-retrieval.** Memories retrieved
  together build stronger graph edges over time (eta=0.01 per
  co-retrieval, capped at 1.0). Pairs are canonically ordered to prevent
  deadlocks and mirror-image double edges under concurrent writes.
- **`advance_lifecycle` MCP tool** for manually triggering the stage sweep
  (useful after bulk imports or from the dashboard).
- **Upgrade guide at [UPGRADING.md](UPGRADING.md).** Covers uv tool / uvx /
  pip / git-checkout / Docker install variants, a Supabase paste-SQL
  variant, verification query, and common-issue troubleshooting.
- **Three new SQL migrations** in `sql/migrations/`:
  - `025_memory_lifecycle.sql` — lifecycle columns + decay tuning params
  - `026_memory_lifecycle_split.sql` — moves lifecycle state to its own
    table so transitions don't touch the HNSW vector index. Adds triggers
    that auto-maintain lifecycle rows on memory insert + profile update.
  - `027_audit_log_backfill.sql` — creates `audit_log` for installs that
    predate its introduction (skips RLS on plain Postgres; only applies
    on Supabase where the `anon` role exists).

### Changed

- **Search-triggered side-effects are now fire-and-forget** on a background
  thread pool. `hybrid_search` returns as soon as the results are
  assembled; the editing-window open and Hebbian strengthening run off
  the hot path. Failures are logged, never raised to the caller.
- **Fresh-install schemas include the lifecycle design directly.** New
  installers get the `memory_lifecycle` table + triggers from
  `sql/schema.sql` and friends. Existing installs apply migrations via
  `./sql/upgrade.sh`.

### Safety

- **Rollback SQL files now refuse to run by accident.** Piping a rollback
  into psql blindly fails with a clear message. Rollbacks live under
  `sql/migrations/rollback/` with a `DANGER_` filename prefix and require
  an explicit session-variable opt-in
  (`SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING'`) before they do
  anything. See `sql/migrations/rollback/README.md`.

### Upgrade notes

**Existing users on v0.10.x:**

```bash
./sql/upgrade.sh "$DATABASE_URL"     # apply 025 + 026 + 027 idempotently
# then upgrade the package normally:
#   uv tool upgrade --refresh ogham-mcp
# or: pip install -U ogham-mcp
# or: uvx --refresh ogham-mcp
```

**Fresh installers:** nothing extra. `sql/schema.sql` now reflects the
post-v0.11.0 state directly; `upgrade.sh` is only needed for existing
deployments.

See [UPGRADING.md](UPGRADING.md) for the detailed walkthrough.

---

## [0.10.4] - 2026-04-22 -- Hook signal filter: verb-only, multilingual

### Fixed

- **Hook noise capture.** `ogham hooks inscribe` was storing every shell
  command that mentioned a common infrastructure noun (`config`, `docker`,
  `supabase`, `neon`, `railway`, `auth`, `token`, `schema`, `migration`,
  etc.) — on an infrastructure-heavy day this captured 100+ routine
  commands as "memories". The fix was a verb-only, multilingual signal
  list: the hook now only fires when the command or output contains an
  explicit decision / error / resolution verb.

### Changed

- `hooks_config.yaml` rewritten with verb-first signals across seven
  languages (English, German, French, Spanish, Italian, Portuguese,
  Dutch) and four categories (errors, decisions, architecture,
  annotations).

---

## [0.10.3] - 2026-04-16 -- Gemini health check fix, Antigravity client

### Fixed

- **Gemini provider health check.** Was falling through as "unknown" and blocking server startup for all Gemini users. Now validates `google-genai` package and `GEMINI_API_KEY`.

### Added

- **Gemini in init wizard.** Option 3 in `ogham init` provider selection with `GEMINI_API_KEY` prompt.
- **Antigravity (Google) client support.** Init wizard detects `~/.gemini/antigravity/` and writes `mcp_config.json` with `serverUrl` for SSE transport.

### Credits

- **Iain** ([@iainmcinnes-rs](https://github.com/iainmcinnes-rs)) for PR #39 (Antigravity client support)

## [0.10.2] - 2026-04-16 -- Hook noise fix, visual dashboard, FastMCP 3.2.4

### Added

- **Visual dashboard.** `ogham dashboard --port 3113` serves a standalone dashboard with KPI metrics, source breakdown, and a searchable memories table with expandable rows. Install via `pip install ogham-mcp[dashboard]`.
- **MCP dashboard tools.** `show_profile_health`, `show_audit_log`, `show_decay_chart` render inline in Claude Desktop and Goose.

### Changed

- **Hook noise reduction.** The `inscribe` hook now skips Edit, Write, WebFetch, Agent, and other reconnaissance tools. Only Bash commands with signal keywords (git commit/push/merge, errors, deploys) and `gh` CLI commands are captured. Reduces hook-generated memories from ~100/session to ~20-30.
- **FastMCP bumped to >=3.2.4.** The 3.1.x pin was based on a stale bytecache artifact, not a real regression. Verified on Python 3.13 and 3.14.

### Fixed

- **Git signal bypass.** `git push`, `git commit` etc. were incorrectly filtered by the routine tools keyword check. Now correctly captured by hooks.

## [0.10.1] - 2026-04-16 -- Structured wrappers, contradiction detection, contributor PRs

### Added

- **Structured store wrappers** -- `store_preference`, `store_fact`, `store_event` MCP tools with typed metadata (strength, confidence, citations, temporal/participant/location fields). Uses Pydantic `BeforeValidator` coercion for FastMCP client compatibility.
- **Contradiction producer** -- when a new memory has opposite polarity to a high-similarity existing memory, a `contradicts` relationship edge is created automatically. Polarity detection uses negation markers across 18 languages loaded from YAML word lists. Write-side only -- retrieval-side suppression deferred to a future release.
- **Query reformulation gating** -- reformulation now fires only on simple-lookup queries. Temporal, ordering, multi-hop, cross-reference, and summary intents skip reformulation to preserve their specialised retrieval paths.
- **Profile health counters** -- `get_stats` now returns `relationships.orphan_count`, `tagging.untagged_count`, `tagging.distinct_tag_count`, `decay.eligible_count`, and `decay.floor_count`. Migration 022 updates the SQL function for existing installs.
- **18-language negation markers** -- `negation_markers` section added to all language YAML files (327 markers total). Used by the contradiction producer and polarity detection.
- **CLI rich renderer hardening** -- `_safe_text()` helper prevents crashes when `store` or `list` receives UUID/datetime types from the backend.

### Changed

- **FastMCP pin** -- rolled back to `>=3.1.0,<3.2` after 3.2.4 introduced a broken internal import (`task_redis_prefix`). Prefab UI work blocked until 3.2.5+ ships with the fix.
- **Import-time initialization** -- Settings validation, embedding cache construction, and active profile lookup are now lazy. Importing modules no longer requires a configured backend, fixing test collection without `SUPABASE_URL`.

### Fixed

- **CONTRIBUTING.md** -- updated with "Regression-proof rules" for side-effect-free imports. Install command fixed to `uv sync --extra dev --extra postgres`.
- **ONNX embedder** -- deferred `onnxruntime` and `tokenizers` imports until after model file existence check.

### Credits

- **Cemre** ([@ersahinco](https://github.com/ersahinco)) for PRs #36 (import-time init fix) and #37 (profile health counters + migration 022)
- **Torres** ([@torres-ai-author](https://github.com/torres-ai-author)) for PR #38 (CLI rich renderer hardening) -- first contribution
- **Pablo** ([@pablo-brown-rodriguez](https://github.com/pablo-brown-rodriguez)) for PR #34 (FastMCP ListStr coercion pattern)

## [0.10.0] - 2026-04-14 -- Cognitive memory: audit, decay, spreading activation

### Added

- **Audit trails.** Append-only `audit_log` table records every store, search, delete, and update. Query via `ogham audit` CLI or the new MCP tool. Useful for GDPR Article 15 subject access requests and cost governance. Runs in the same Postgres instance as memories -- no extra infrastructure.

- **Hebbian decay and potentiation.** New `apply_hebbian_decay()` SQL function. Memories not accessed within 7 days lose importance over time (5% per 30-day idle period). Memories accessed 10+ times become "potentiated" and decay much slower (1% per 30 days). Floor at 0.05 keeps decayed memories findable but low-ranked. Run as a batch job: `ogham decay --dry-run` first to preview, then `ogham decay`. Schedule via system cron or pg_cron. Based on Hebb (1949) and Bi & Poo (2001).

- **Spreading activation.** Entity graph walk now informs search for cross-reference, ordering, and summary queries. The SQL function `spread_entity_activation_memories()` walks the bipartite entity/memory graph at query time, merging activation scores with hybrid search results. Based on Collins & Loftus (1975).

- **Density-adaptive activation weight.** The graph signal adapts to the profile's entity density at runtime. Sparse multi-session profiles get more graph influence (up to 0.30), dense single-chat profiles get less (0.05). Mitigates cluster saturation where uniform activation degrades retrieval.

- **Read-time fact extraction (opt-in).** `hybrid_search(extract_facts=True)` runs an LLM over retrieved memories to produce focused facts. Supports Ollama (local), Gemini, and OpenAI. Default off -- verbatim results remain the ground truth.

- **Conflict detection on store.** Stores over 75% similar to an existing memory trigger a warning with the conflict IDs and content preview. Threshold configurable via `OGHAM_CONFLICT_THRESHOLD`. The CLI `ogham store` now goes through the full enrichment pipeline (entity extraction, importance scoring, auto-link) like the MCP tool.

- **Suggest connections.** New MCP tool `suggest_connections` finds memories that share entities but have no explicit relationship edge -- surfaces hidden connections that an agent might otherwise miss.

- **Multilingual entity enrichment.** Event, preference, quantity, and possessive triggers expanded across all 18 supported languages (German, French, Spanish, Italian, Portuguese, Dutch, Russian, Polish, Turkish, Irish, Arabic, Hindi, Japanese, Korean, Chinese, Ukrainian). Weddings in Japanese, preferences in Arabic, quantities in Russian now tag at ingest time.

- **`ogham audit` and `ogham decay` CLI commands.** Query the audit log and run Hebbian decay batch jobs without writing SQL.

- **Schema parity test.** Uses Python `inspect.unwrap()` to verify the backend's method signatures match the SQL function signatures. Catches "Python sends N parameters, SQL expects M" bugs at CI time.

- **CI smoke test.** GitHub Action applies `schema_postgres.sql` to an ephemeral Postgres, verifies `hybrid_search_memories` signature, then calls it from Python with all parameters. Also runs the upgrade path (base schema + all migrations).

### Changed

- `hybrid_search_memories` relevance formula now includes `m.importance` as a multiplier, an entity overlap boost, and an exponential recency decay term. Same 12 parameters as v0.9.2.

### Fixed

- **Migration 021: dimension-aware halfvec casts.** Previous migrations hardcoded `halfvec(512)` in function bodies, which broke non-512 dimension providers (bge-m3 at 1024, OpenAI text-embedding-3-large at 3072). The new migration introspects the actual `memories.embedding` dimension via `pg_attribute` + `format_type` and templates the cast via `format()` + `EXECUTE`. The HNSW index rebuild is gated behind an opt-in session GUC (`SET ogham.rebuild_hnsw = 'on'`) because index recreation on a populated table is expensive.

### Upgrading

Apply `sql/migrations/021_dim_aware_halfvec.sql` to your database. For Postgres/Neon:

```bash
psql "$DATABASE_URL" -f sql/migrations/021_dim_aware_halfvec.sql
```

If you're running at a non-512 dimension or want HNSW queries to use the new cast:

```bash
psql "$DATABASE_URL" -c "SET ogham.rebuild_hnsw = 'on';" -f sql/migrations/021_dim_aware_halfvec.sql
```

Supabase users: paste the migration body into the SQL Editor.

### Credits

- **Josh** ([@ninthhousestudios](https://github.com/ninthhousestudios)) for issues #22, #24, and the dimension-aware halfvec pattern in PR #25.
- **Bram** ([@bramvera](https://github.com/bramvera)) for PRs #17, #21, and #23 catching `hybrid_search_memories` parameter mismatches.

## [0.9.1] - 2026-04-08

### Fixed
- **ONNX provider**: `_embed_onnx()` was using `result["dense"]` against an `OnnxResult` dataclass and failed on cache misses with `'OnnxResult' object is not subscriptable`. Now uses `result.dense`. Cached queries worked because the bug only triggered on the embedding path -- thanks to Josh for the report.

### Added
- **Preference extraction** -- new `preference:` entity tag, with 280 trigger words across all 18 supported languages (English, German, French, Spanish, Italian, Portuguese, Dutch, Russian, Polish, Turkish, Irish, Arabic, Hindi, Japanese, Korean, Chinese, Ukrainian, plus pt-BR variants). Detects "prefer", "favorite", "like better", "rather", "always get", "go-to" and equivalents at ingest time. Extracted memories get tagged automatically and surface for recommendation-style queries.
- **Multi-word phrase matching** in entity extraction `_match()` -- phrases like "always get", "better than", "tercih ederim" now match via substring lookup. Single Latin words still use word-boundary matching to avoid partial false positives.
- **`format_results_with_sessions()`** in `service.py` -- formats search results with timeline table at the top, session boundary headers (`=== SESSION: 2024-04-12 ===`), entity and date annotations per memory. Used by gateway chat endpoints and benchmark scripts. Produces the same enriched context that drove the LongMemEval 91.8% and BEAM 100K 0.554 results.
- **`RERANK_MODEL` config option** -- choose between `flashrank` (default, 33M params, CPU-only) and `bge` (BAAI/bge-reranker-v2-m3, 568M params, multilingual, via `sentence-transformers`). Both disabled by default. Benchmark experiments showed neither helps when retrieval is already above 95% R@10, but the plumbing is here for users who want to test it on their own data.
- **BEAM batch QA harness** (`benchmarks/beam_batch.py`) -- three-phase pipeline (`prepare` / `submit` / `judge`) using OpenAI Batch API for 50% cost savings. Implements the BEAM paper's exact Appendix G nugget judge prompt for direct comparability to published numbers. Skips event_ordering (which uses Kendall tau-b in the paper, needs separate equivalence-detector pipeline).

### Changed
- BEAM benchmark default `EMBEDDING_BATCH_SIZE` is now `None` (uses provider default) instead of hardcoded 1000. Gemini caps batches at 100 requests; Voyage allows 1000. The previous hardcoded value broke ingest on Gemini.
- Added comment to `BEAM_GRAPH_DEPTH` env var explaining why graph augmentation is disabled by default. Experiments showed that graph_depth=1 with the current memory-similarity graph hurts retrieval significantly (-13 to -44pp across all categories), because similarity edges duplicate vector-search hits and displace diverse evidence. The win requires a real entity-relationship graph (memory→entity edges), planned for v0.10.

### Documentation
- New blog post: ["BEAM benchmark -- a fair look at where we stand on long-term memory"](https://ogham-mcp.dev/blog/beam-benchmark-v090/) -- 0.554 nugget score vs paper's 0.358, honest gaps in retrieval, what we're fixing next.
- New blog post: ["From 62% to 92% -- what we learned about reading, not retrieval"](https://ogham-mcp.dev/blog/longmemeval-92/) -- the LongMemEval journey from 62% to 91.8% via context engineering.

### Benchmark numbers (v0.9.1)
- **LongMemEval QA accuracy**: 0.918 (459/500), gpt-5.4-mini reader with reasoning, paper Appendix G judge
- **LongMemEval R@10**: 0.972
- **BEAM 100K QA nugget score**: 0.554 (vs paper best 0.358)
- **BEAM 100K R@10**: 0.737

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
