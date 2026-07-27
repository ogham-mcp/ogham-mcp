# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.17.1] - 2026-07-27 -- Streamable HTTP transport, adapter credentials, CLI fixes

### Added

- **Streamable HTTP transport.** `ogham serve --transport streamable-http`
  (`http` works as an alias), with the endpoint at `/mcp`. The MCP
  specification defines stdio and Streamable HTTP as its two standard
  transports and treats HTTP+SSE as deprecated. The difference that matters is
  session recovery: Streamable HTTP assigns an `Mcp-Session-Id` and defines the
  way back when a session is gone (HTTP 404, then re-initialize), whereas
  HTTP+SSE defines none -- a session that loses its initialized state rejects
  every later request with `-32602`, and the client cannot tell a lifecycle
  problem from a bad argument. One dropped connection can leave an agent
  failing every call until someone restarts it, which is what @mAAdhaTTah ran
  into in #71. Remote and containerised deployments should prefer
  streamable-http. `--transport sse` still works and now logs a deprecation
  warning naming the replacement.

  Running a network transport in Docker needs `--host 0.0.0.0`; the default
  `127.0.0.1` is only reachable from inside the container. The README now
  covers this.

### Fixed

- **Credentials in `~/.ogham/config.env` now reach the ingestion adapters.**
  The Telegram, Slack, GitHub and Beads adapters read their tokens from the
  process environment, but env-file values were only ever loaded into Ogham's
  settings object -- so a token placed exactly where the adapter docs say to
  put it was invisible, and the adapter reported it as unset. Env-file values
  are now exported to the environment, with a real environment variable still
  taking precedence. Obsidian was unaffected, since it takes a path rather than
  a credential.

- **`ogham download-model` exists again.** With `EMBEDDING_PROVIDER=onnx` and no
  model present, Ogham tells you to run `ogham download-model bge-m3` -- but the
  command had been dropped in v0.8.5, while the error message and the README
  kept pointing at it. Restored, so there is once more a working way to fetch
  the BGE-M3 model instead of unpacking the release archive by hand. Reported by
  @mAAdhaTTah (#68).

- **`ogham delete` accepts a short ID prefix.** `--help` promised "full UUID or
  prefix", but passing a prefix failed against the database -- and since `list`
  and `search` display only the first 8 characters, there was no straightforward
  way to obtain the full UUID the command actually wanted. Prefixes now resolve
  to a full ID, and an ambiguous prefix lists its candidates and stops rather
  than guessing. `list` and `search` also gain `--full-id` for copyable UUIDs.
  Prefix resolution requires a Postgres-side cast, so it works on the Postgres
  backend and reports clearly on the others. Reported by @mAAdhaTTah (#70).

- **`ogham init` can configure the recommended transport.** The setup wizard
  offered only stdio and SSE and always wrote an SSE client URL, so anyone
  choosing the multi-agent option during setup was placed on the deprecated
  transport. It now offers Streamable HTTP as the multi-agent choice, keeps SSE
  as a labelled legacy option, and writes the endpoint matching whichever you
  pick.

### Notes

- No migrations. v0.17.1 is safe to install over any v0.17.0 database.

## [0.17.0] - 2026-07-09 -- Ingestion adapters, portable predicate URIs, and provenance chains

### Added

- **Obsidian / markdown vault capture.** `ogham ingest-obsidian <vault>` (CLI)
  and `ingest_obsidian` (MCP tool) scan a local vault and store each note as a
  memory tagged `source: obsidian`. YAML frontmatter tags become memory tags.
  The scan is idempotent -- unchanged notes are skipped by a content
  fingerprint -- so you can run it on a timer and re-runs never double-store.
- **Telegram capture.** `ogham ingest-telegram` (CLI) and `ingest_telegram`
  (MCP tool) pull messages from a Telegram bot with `getUpdates` and store them
  tagged `source: telegram`. Deduped on the Telegram update id, with an optional
  per-chat allowlist. Outbound-only -- the adapter calls out to Telegram; there
  is no webhook and no public endpoint to expose.
- **Slack capture.** `ogham ingest-slack` (CLI) and `ingest_slack` (MCP tool)
  poll `conversations.history` for the channels you list and store their
  messages tagged `source: slack`. Deduped on channel id + message timestamp.
  Outbound-only (a bot token, no Socket Mode, no exposed endpoint); the bot must
  be a member of each channel it reads.
- **One shared ingestion path.** All three adapters forward to the same
  server-side enrichment -- embedding, entity extraction, and store-side dedup --
  so an adapter is a thin forwarder, not a re-implementation of the pipeline.
  Each is a one-shot, idempotent poll designed to run under a timer
  (launchd/cron); none runs as a daemon and none listens for inbound
  connections, so your memory store is never exposed.
- **GitHub Issues capture.** `ogham import-github` (CLI) and `import_github`
  (MCP tool) fetch issues and their comments from the repositories you list and
  store them tagged `source: github`. Each issue and each comment becomes its
  own memory; re-runs dedup on a stable id so nothing double-stores.
- **Beads capture.** `ogham import-beads` (CLI) and `import_beads` (MCP tool)
  read issues and comments from a local Beads project via the `bd` CLI and store
  them tagged `source: beads`. No network and no token -- a local command over a
  local database.
- **Portable predicate URIs.** Every predicate in the typed-edge vocabulary now
  carries a stable URI, plus a Schema.org alignment where a standard property
  genuinely matches -- five of them (`OWNS`, `OWNED_BY`, `MENTIONS`, `PART_OF`,
  `CONTAINS`). The rest are left unmapped rather than forced onto an approximate
  term. `describe_predicates` (MCP tool) and `ogham predicates` (CLI) list the
  vocabulary with its URIs.
- **Provenance chains.** `store_triple` now accepts a `derived_from` argument
  recording the evidence an edge was built from -- other edges and/or source
  memories. Two new tools walk that lineage: `trace_provenance` follows an edge
  back to its root evidence, and `find_derivatives` walks forward to everything
  that depends on a fact ("if this is retracted, what breaks?").

### Migrations

- `045_predicate_uris.sql` -- adds URI columns to the predicate vocabulary.
- `046_edge_provenance.sql` -- adds a `derived_from` column + index to the edge
  table. Both are additive and idempotent; safe to apply to an existing v0.16
  install.

## [0.16.0] - 2026-07-03 -- Typed-edge context graph + Linear importer + dim-parameterized schemas

### Added

- **Typed-edge context graph: two new tools, `store_triple` and
  `query_join`.** `store_triple(subject, predicate, object)` records a
  typed relationship -- `AuthService DEPENDS_ON Postgres` -- against a
  fixed predicate vocabulary. `query_join(start_entity, predicate_path,
  hop_limit)` walks that path exactly and returns the entities, edges,
  and source citations along it. This answers the question a vector
  search cannot: "which team owns the component that depends on the
  service the planner chose?" The answer is a path through your stored
  facts, not a passage of text that happens to mention all three.
- **Controlled predicate vocabulary.** Sixteen predicate names --
  `DEPENDS_ON`, `OWNS`, `ASSIGNED_TO`, `DECIDED`, `MENTIONS`, `BLOCKS`,
  and their inverses -- validated at write time. Off-vocabulary
  predicates are rejected, so the graph stays queryable.
- **Write-time supersession.** When you store a new edge for a
  subject/predicate pair that already has one, the old edge moves to
  history with a timestamp instead of being deleted. Current-view
  queries see only the live fact; you can still read the full history
  when you need the timeline. No more stale relationships reading as
  current.
- **Entity aliases.** "the auth module" resolves to `AuthModule` before
  the graph is queried, so triples and joins line up even when the same
  thing is named different ways.
- **Linear importer.** `import_linear` (MCP tool) and `ogham import
  linear` (CLI) pull your Linear issues in as memories, so past
  decisions and task context are searchable alongside everything else.
  Re-runs are de-duplicated on the issue id.

### Changed

- **The embedding dimension is now a parameter, not a hardcoded 512.**
  All three schema files (`schema.sql`, `schema_postgres.sql`,
  `schema_selfhost_supabase.sql`) use `:embedding_dim` instead of a
  literal `vector(512)`. If you run a non-512 provider (Mistral at 1024,
  Voyage, Gemini at 768, OpenAI's large model at 3072) you no longer
  edit SQL by hand -- set `EMBEDDING_DIM` and apply. The interactive
  setup wizard and the Python apply path handle the substitution for
  you; for a manual apply, substitute the token first:
  `sed 's/:embedding_dim/'"$EMBEDDING_DIM"'/g' sql/schema_postgres.sql | psql "$DATABASE_URL"`.
  **Existing 512-dim installs are unaffected** -- applying at
  `EMBEDDING_DIM=512` produces exactly the same schema as before.

### Reliability

- **Startup guard against a silent dimension mismatch.** On the Postgres
  backend, Ogham now checks at boot that your configured `EMBEDDING_DIM`
  matches the actual dimension of the `memories.embedding` column, and
  refuses to start on a mismatch with a message telling you which side
  to fix. This closes a trap where writes succeed at the wrong dimension
  and searches quietly return nothing useful later.

- **Fixed a lockout of the graph tables on self-hosted Postgres.** If
  you self-host on plain Postgres (not Supabase) with a non-superuser
  database role, row-level security was left forced on the entity and
  typed-edge tables without a matching policy, which silently hid every
  row -- graph lookups and `store_triple` / `query_join` returned
  nothing. Upgrading now runs a self-heal migration that unlocks those
  tables automatically. Supabase installs are unaffected.

### Fixed

- **Fixed a crash in `hybrid_search` on the self-hosted Postgres
  backend.** Timestamps come back as datetime objects on Postgres (and
  as strings on Supabase), and a date parser assumed strings, so every
  hybrid search raised an error. The same root-cause fix also resolves a
  crash in OKF export for time-limited memories and makes exported OKF
  bundles byte-consistent across backends.

- **The setup wizard now sets the right embedding dimension per
  provider.** It previously wrote `EMBEDDING_DIM=512` even for providers
  that default to 1024 (OpenAI, Mistral, Voyage), which could stop the
  server starting after the new dimension check. Both the interactive
  and `ogham init --provider ...` paths now use the provider's real
  default; an explicit dimension still wins.

- **Self-hosted Supabase schema `hybrid_search_memories` was behind the
  other two schema variants by two parameters.** Brought to signature
  parity so hybrid search behaves the same across Supabase Cloud,
  self-hosted Supabase, and vanilla Postgres. The test suite now checks
  SQL function signatures across all three variants, not just their
  columns.



## [0.15.3] - 2026-07-05 -- Gemini pre-normalize skip + column-parity gate

### Performance

- **Skip client-side L2 normalization for `gemini-embedding-2` at
  sub-3072 output dims.** The GA model pre-normalizes at every output
  dim (verified across 512 / 768 / 1536 / 3072, `||v||` within 2e-7
  of 1.0). Skip is gated on the model name, so anyone still on
  `gemini-embedding-001` or a preview alias keeps the defensive
  normalize path. Small speedup on batch embedding, no behavior
  change for GA callers.

### Testing

- **Extended `test_schema_parity.py` to check column-set parity
  across all three schema variants** (Supabase Cloud, self-hosted
  Supabase, vanilla Postgres). Intentional divergences live in
  `tests/schema_parity_whitelist.yaml`. Catches the "column added to
  one variant, forgotten in the others" class of drift.

## [0.15.2] - 2026-06-18 -- Linux import ENAMETOOLONG hotfix

### Fixed

- **`import_memories_tool` crashed on Linux with `OSError: [Errno 36]
  File name too long`** for any non-trivial JSON payload. The OKF
  directory-detection logic added in v0.15.0 called
  `Path(data).is_dir()` on the raw input string. On Linux, `pathlib`
  raises `OSError` when any path component exceeds `NAME_MAX` (255 bytes),
  which is every realistic export payload. macOS silently returns `False`,
  so local dev never caught it -- but Linux users hit `OSError` on any
  call to `import_memories_tool` larger than a trivial test fixture. Fix:
  wrap the `is_dir()` call in a try/except that swallows `OSError` and
  returns `False`, so the function falls through to the JSON branch as
  intended. Two regression tests pin the helper and the end-to-end path
  against simulated `ENAMETOOLONG`. PyPI 0.15.0 and 0.15.1 are both
  affected; upgrade to 0.15.2 if you import on Linux.

## [0.15.1] - 2026-06-18 -- Supabase OKF round-trip hotfix

### Fixed

- **Supabase upsert was silently doing plain INSERT.** The
  `SupabaseBackend` client was being initialised with a global
  `Prefer: return=representation` header. The postgrest client's
  per-call `upsert()` builds its own Prefer header containing
  `resolution=merge-duplicates`, but the global header silently
  overwrote it during a `headers.update(self.headers)` merge step.
  Result: every Supabase upsert hit the plain-INSERT path, and the
  second OKF re-import failed with
  `duplicate key value violates unique constraint memories_pkey`.
  This affected anyone re-importing an OKF bundle into the same
  profile. Fix: drop the global Prefer header (each operation builds
  its own correctly) and pass `on_conflict="id"` explicitly to the
  upsert call. A regression test pins both the absent global header
  and the per-call Prefer + on_conflict combination.

- **`ogham import <okf-bundle-dir>` crashed with `IsADirectoryError`.**
  The CLI shell wrapper around `import_memories_tool` unconditionally
  called `open(file)` to read JSON/markdown payloads, even though the
  MCP tool already auto-detects OKF bundle directories. Fix: branch
  on `os.path.isdir(file)` and pass the directory path through as-is
  for OKF bundles. Existing JSON/markdown imports keep their read flow.

- **`ogham export --help` did not list `okf` as a valid format.** The
  underlying code accepted it; only the typer help text was stale.

## [0.15.0] - 2026-06-18 -- Open Knowledge Format + offline bundle viewer

### Added

- **Open Knowledge Format (OKF) v0.1 round-trip.** Ogham can now export
  memories as OKF v0.1 conformant bundles (markdown documents with YAML
  frontmatter, per the
  [Google Cloud spec](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md))
  and import OKF bundles back. Round-trip preserves UUID, content, tags,
  source, and any extension metadata; embeddings are regenerated on import
  by design. The v0.12 Obsidian export (`--format markdown`) is unchanged
  and continues to ship alongside OKF.

  - `ogham export --format okf` writes a bundle directory to the working
    directory.
  - `ogham import <dir>` auto-detects OKF bundles via the bundle-root
    `index.md` declaring `okf_version`. Without that file the directory
    is rejected, preventing accidental ingest of arbitrary paths.
  - Concepts missing the optional `id:` extension are imported as new
    memories; the count is surfaced as `missing_id_count` in the import
    result. Concepts that fail to parse are surfaced as `skipped_count`.
  - See `docs/okf-format.md` for the full round-trip contract, filename
    scheme, and type-tag mapping rules.

- **OKF bundle viewer.** Every OKF export writes a self-contained
  `viewer.html` at the bundle root by default. Open it with any browser
  via `file://` -- no server, no internet, no CDN required. The page
  renders the bundle as a Cytoscape.js concept graph: each memory is a
  node coloured by type, edges follow markdown links between concepts,
  clicking a node shows its body and tags in a side panel. Opt out with
  `include_viewer=false`. Cytoscape.js 3.34.0 is vendored inside the
  wheel (MIT-licensed; see `docs/snapshots/cytoscape-js/NOTICE.md`).

- **`gap_contradictions_for_ids` RPC (#262, migration 039).** New
  read-only Postgres function used by gap-analysis queries to surface
  `contradicts` edges where one endpoint is inside a result set and the
  other is in the same profile but outside it. Pure addition; no
  existing tables or RPCs touched.

### Changed

- **Default `gemini_embed_model` flipped to `gemini-embedding-2`.**
  Previously `gemini-embedding-2-preview`. Both names point at the same
  GA model on Google's API; `-preview` is a legacy alias slated for
  retirement. Existing deployments overriding the setting are
  unaffected. `embedding_pricing.yaml` lists both entries with the
  preview marked as legacy.

- **`wiki_lint_contradictions` filter hardened (migration 040).** The
  function shipped in v0.12 (migration 031) joined only the source side
  of each contradiction edge to `memories` and filtered profile on that
  side alone. In normal operation Ogham's auto-linker only creates
  within-profile edges, so realistic counts are unchanged; this change
  hardens the filter so cross-profile edges are not over-counted and
  edges whose source has moved profile are not under-counted. Forward
  migration replaces the function via `CREATE OR REPLACE`; rollback is
  guarded.

- **Dependency floors bumped.** `fastmcp>=3.4.2,<4`, `typer>=0.24`.
  Within-major bumps only; `google-genai`, `mistralai`, and `voyageai`
  majors are deferred to a separate refresh release.

### Fixed

- **Gemini batch embedding silent truncation (#60).** Gemini's
  `batchEmbedContents` endpoint occasionally returns HTTP 200 with
  fewer embeddings than items submitted. Before this fix the short
  response was silently zipped against the request batch, leaving
  `None` slots that raised `Embedding batch completed with missing
  results` at the end of the batch with no retry. The batch path now
  raises immediately when `len(response.embeddings) != len(texts)`,
  and the tenacity retry decorator fires on it the same way it fires
  on 429 / RESOURCE_EXHAUSTED / 503 / UNAVAILABLE.

### Migrations

- **039_gap_contradictions_scoped.sql** -- new `gap_contradictions_for_ids`
  RPC for gap-analysis. Idempotent (`CREATE OR REPLACE FUNCTION`).
- **040_fix_lint_contradiction_filter.sql** -- replaces
  `wiki_lint_contradictions` with a both-endpoint scoped definition.
  Idempotent.

Apply both via the Supabase SQL Editor or your usual migration path.
Self-hosters on plain Postgres can apply them directly with `psql`.
Both have `DANGER_*` rollback files in `sql/migrations/rollback/` with
the standard `ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING'`
session-variable guard.

### Build

- The sdist now includes the `shared/` cross-stack data tree so the
  `src/ogham/hooks_config.yaml` symlink resolves when `pip install`
  extracts the tarball and builds the wheel from sdist. Without this,
  the build failed with `FileNotFoundError`. Same class of fix as
  v0.14 onwards for the Docker build context.

## [0.14.3] - 2026-05-25 -- Supabase Data API explicit grants

### Added

- **Migration 038: explicit Data API grants for `service_role`.**
  Supabase is changing a platform default
  ([announcement](https://github.com/orgs/supabase/discussions/45329)):
  new tables in the `public` schema are no longer automatically exposed
  to the Data API (PostgREST / GraphQL / supabase-js). This becomes the
  default for new projects on 2026-05-30 and applies to all existing
  projects on 2026-10-30.

  Ogham reaches Supabase through PostgREST with the `service_role` /
  `sb_secret_` key. Without explicit table-level grants, requests start
  failing with `42501: permission denied` once Supabase removes the old
  default. Migration 038 adds the required
  `GRANT SELECT, INSERT, UPDATE, DELETE` on every Ogham table to
  `service_role`. `anon` stays denied at the RLS layer; nothing else
  about the access model changes.

  The grants are wrapped in a role-existence guard, so on vanilla
  Postgres or Neon (no `service_role`) the migration is a clean no-op.
  Fresh installs pick the grants up from `schema.sql` automatically;
  `schema_selfhost_supabase.sql` carries them too. A `DANGER_038`
  rollback is included for the rare case you need to revert.

  **Self-hosting on Supabase?** Apply `sql/migrations/038_data_api_grants.sql`
  before the October 30 enforcement date -- or run it now to get ahead
  of it. Neon and vanilla-Postgres backends are unaffected.

## [0.14.2] - 2026-05-05 -- compress_old_memories Postgres fetch fix

### Fixed

- **`compress_old_memories` ProgrammingError on Postgres backend** (#51,
  reported by @wmemorgan). When a memory aged into Level 1 or Level 2
  compression, `_update_compression` called `backend._execute` without
  `fetch="none"`. The backend defaults to `fetch="all"` and runs
  `cursor.fetchall()` after a plain `UPDATE ... WHERE`, which produces
  no result set -- raising `ProgrammingError("no results to fetch")`.
  Fixed by passing `fetch="none"` explicitly. Two regression tests
  pin the call signature so this can't silently regress.

  Latent since v0.9.0. Self-hosters on the Postgres backend hitting
  compression were affected; Supabase backend was unaffected (different
  code path via the postgrest client).

## [0.14.1] - 2026-04-30 -- Schema-smoke fix for vanilla Postgres

### Fixed

- **Migrations 036 + 037 anon-role guard** for vanilla Postgres
  self-hosters. v0.14.0's two new migrations (`036_entities_backfill`
  and `037_revoke_rpc_anon`) referenced the Supabase-specific `anon`
  role unconditionally, so on plain Postgres installs (no `anon`
  role) both migrations failed with `role "anon" does not exist`.
  Wrapped the role-specific blocks in `DO $$ ... $$` that check
  `pg_roles` first; on non-Supabase installs they now emit a
  `NOTICE` and no-op gracefully. Mirrors the existing pattern in
  migration 032. Supabase installs unaffected.

## [0.14.0] - 2026-04-30 -- Memory hygiene + ingestion control

Theme: explicit control over what flows in and out of memory. Three
features compose into one story: PR #42's flow gates set the policy,
the Claude Code importer is the first ingestion surface that respects
it, and the entity graph -- dormant since v0.10 -- finally lights up.

### New: explicit recall + inscribe flow controls (#42, thanks @ersahinco)

Two-layer gating across hook, CLI, and MCP surfaces:

- Per-call flags: `--recall/--no-recall` on `ogham hooks recall` and
  `ogham search`; `--inscribe/--no-inscribe` on `ogham hooks inscribe`
  and `ogham store`.
- Process-wide env vars: `OGHAM_RECALL_ENABLED=false` /
  `OGHAM_INSCRIBE_ENABLED=false` disable the corresponding flow for
  the whole server.

Use this when you want an agent attached to Ogham without letting it
pull memory into context (`--no-recall`) or write new memory
(`--no-inscribe`). Admin operations like config / health / stats /
audit / export remain available -- only the read-into-context and
write-new-memory paths are gated.

```bash
ogham search "query" --no-recall              # one-off CLI search skip
ogham store "fact" --no-inscribe              # one-off CLI store skip
ogham hooks recall --no-recall                # one-off hook recall skip
ogham hooks inscribe --no-inscribe            # one-off hook capture skip
OGHAM_INSCRIBE_ENABLED=false ogham serve      # process-wide
```

### New: Claude Code local-memory importer (#216)

```bash
ogham import-claude-code ~/.claude/projects/<encoded-cwd>/memory \
    --project ogham --dedup 0.8
```

Reads YAML-frontmatter markdown files from Claude Code's auto-memory
directory and imports each as a memory tagged
`source:claude-code-memory + type:<frontmatter type>`. The encoded-cwd
heuristic for inferring a project tag is lossy on hyphenated repo
names (e.g. `openbrain-sharedmemory` -> `sharedmemory`), so `--project`
overrides the inferred tag. `MEMORY.md` (the index file) and dotfiles
are skipped. The importer respects `inscribe_enabled()` -- gating from
PR #42 applies here too.

MCP tool: `import_claude_code_memories(directory, project_tag=...)`.

### New: Claude.ai conversation export importer

```bash
ogham import-claude-ai ~/Downloads/data-<id>-batch-0000 --profile claude-ai
```

Imports a Claude.ai data export (Settings -> Privacy -> Request your
data). Accepts the ZIP Anthropic emails, the unzipped directory, or
`conversations.json` directly. Walks each conversation as consecutive
(human, assistant) turn-pairs and stores one memory per pair with
the assistant turn as content and the human prompt in
`metadata.user_prompt` -- this flips retrieval onto the answer's
substance while keeping the question recoverable.

Tagging: `source:claude-ai`, `claude-conversation:<title-slug>`,
optional `project:<tag>`. UUIDs from the export land in metadata so
re-importing the same export six months later only pulls in new
turns. A conservative smart filter drops pleasantry exchanges
("thanks" / "got it"); pass `--no-smart-filter` to keep them. Use
`--since 2026-01-01` to import only recent conversations and
`--mode raw` for one memory per individual message.

To get the LLM-distilled summary view of an imported conversation,
run `compile_wiki(topic="claude-conversation:<slug>")` -- verbatim
ingest plus on-demand synthesis means you keep the raw turns and
also get the digest, without the importer making LLM calls upfront.

MCP tool: `import_claude_ai_export(path, profile, mode=...)`.

### Note: bulk importers skip per-memory enrichment

All three bulk importers (Claude Code, Claude.ai, Agent Zero) write
through `import_memories`, which embeds + dedups + inserts in
batches but skips per-memory entity extraction and auto-link. This
is by design -- a 600-memory import would otherwise run thousands
of secondary RPCs. Imported memories are immediately searchable
via embedding + keyword; to populate the entity graph after import
run:

```bash
ogham backfill-entities --profile <name>
```

This walks the profile, runs `extract_entities(content)` per memory,
and populates `memory_entities` + relationship edges. Re-runs are
idempotent (ON CONFLICT DO NOTHING).

### New: entity graph end-to-end (#240)

The `entities` and `memory_entities` tables landed in v0.10's schema
but never had a write path. Spreading-activation, density, and
suggest-connections were silent no-ops on every install. v0.14 closes
the loop:

- **Migration 036** retrofits the schema (`CREATE IF NOT EXISTS` for
  tables, indexes, RLS) on older deployments and adds the
  `link_memory_entities`, `refresh_entity_temporal_span`, and
  `spread_entity_activation_memories` RPCs. Idempotent on fresh
  installs that already have the tables.
- **Live write hook**: `service.store_memory` calls
  `link_memory_entities` after every successful insert, populating
  the graph from now on. Failure is logged + swallowed so older
  deployments missing the RPC degrade rather than break ingest.
- **Backfill loop**: `ogham backfill-entities [--profile NAME]`
  walks existing memories and populates entities + edges from
  `extract_entities(content)`. ON CONFLICT DO NOTHING makes re-runs
  free.

After backfill, `suggest_connections`, `entity_graph_density`, and
the `spread_entity_activation_memories` RPC all start returning real
results. MCP tool: `backfill_entities`.

### New: `compile_wiki` source-count cap (#243)

`settings.compile_max_sources=100` (default; 0 disables) refuses
mega-rollup tags before they hit the LLM. Tags like `type:gotcha`
that accumulate hundreds of memories produce LLM outputs that fail
JSON escape and saturate context budgets. Refused topics return
`status="skipped_oversize"` in ~0.3s instead of burning a 70-second
LLM call. Pass `force_oversize=True` (CLI / MCP) to override.

### Security: lock down entity-graph SECURITY DEFINER RPCs

Migration 037 revokes EXECUTE from `anon` and `authenticated` on
all 10 SECURITY DEFINER functions added in v0.13.1 (Hotfix A) and
v0.14 (migration 036). These are infrastructure used by the Ogham
server (which holds the service_role key); they were never intended
to be part of the public REST surface. Without the revoke, anyone
with the project's anon key could call them and bypass RLS by design.

Apply 037 in order after 036. Schema files (`schema.sql`,
`schema_postgres.sql`, `schema_selfhost_supabase.sql`) also patched
so fresh installs ship locked-down.

### Performance: bounded conflict-detection in `store_memory`

Supabase telemetry showed memory INSERTs at 45.8% of total database
time with 7-8 second max-time outliers. Tracing it back:
`store_memory_enriched` ran a synchronous `hybrid_search` inside the
write path for surprise scoring + conflict warning, with no upper
bound on its latency. p95 hybrid_search times multiplied across every
store created the long tail.

Two bounds in v0.14:

- `limit=1` on the conflict-detection search (was 3). Surprise scoring
  only needs the top neighbour; the conflict warning is a soft hint
  where one example suffices. Halves the HNSW probe cost.
- Wall-clock timeout via `OGHAM_CONFLICT_TIMEOUT_MS` (default 1500).
  Out-of-budget calls log a warning and fall through with
  `surprise=0.5`; the store completes regardless. The in-flight RPC
  keeps running on its background thread until it returns -- we just
  stop waiting.

Plus a boot-time warmup: `ogham serve` now runs one discardable
`hybrid_search` before taking traffic, so the user's first query
doesn't pay the Supabase cold-connect cost. Disable with
`OGHAM_BOOT_WARMUP=false`.

Steady-state store latency dropped from 8000 ms max to ~600 ms typical
on a 1300-memory profile.

### Fixed: LLM compile path for `compile_wiki`

Three latent bugs in `synthesize_json` surfaced during a bulk
recompile of v0.13's TLDR rows:

- `response_format={"type": "json_object"}` is now passed to
  OpenAI-compat providers (was prompt-only before; default Ollama
  fallbacks ignored the schema hint and returned markdown prose).
  Ollama path also lifts to top-level `format=json` for older
  runtimes.
- `max_tokens` bumped 4096 -> 8192 for compile bodies; long-source
  topics were truncating mid-string.
- `JSONDecoder(strict=False)` fallback for bare control characters
  (`\n`, `\t`) inside markdown body fields. Strict parse is still
  tried first so real structural errors surface.

### New: `OGHAM_COMPILE_MAX_TOKENS` knob

The 8192-token compile body fits typical conversation tags but can
truncate on very long technical discussions (~10-15+ source memories
of dense code/explanation). Modern models support far more --
Gemini 2.5 Flash takes 1M tokens of input and emits up to 64K
output, GPT-4o and Claude 3.5 Sonnet are similarly generous. If
you're bringing your own LLM, you bear the cost; we shouldn't cap
you at 8192 just because that fits Ollama defaults.

Override with the env var:

```bash
OGHAM_COMPILE_MAX_TOKENS=32768 ogham serve
```

Default stays 8192 (cheap, predictable). Values above 16384 emit a
warning so the cost implication isn't invisible. Provider
rate-limits and per-call billing still apply -- `compile_wiki` runs
one synthesize call per recompiled topic.

### Migrations

Apply in order on existing deployments:

| | |
|---|---|
| **036** | `entities` + `memory_entities` tables + supporting RPCs |
| **037** | REVOKE EXECUTE on SECURITY DEFINER RPCs from anon/authenticated |

After 036, run `ogham backfill-entities --profile <name>` per profile
to populate historical entities. New writes after upgrade are linked
automatically by the live write hook.

## [0.13.1] - 2026-04-29 -- Smart hook capture + Supabase background-task fix

Patch release. The headline is smart inscribe extraction (#43, thanks
@ersahinco) -- inscribe hooks now record what was learned, changed,
decided, created, corrected, or failed instead of generic tool
activity. Plus a fix for six call sites that have been silently failing
on Supabase deployments since v0.11.

### New: smart inscribe hooks (#43)

`Edit` and `Write` events are no longer blanket-skipped. Response-gated
extraction captures meaningful diffs, signature changes, and new-file
docstring summaries; skips typo-only edits and ambiguous overwrites.

`Bash` capture upgraded for `git commit -m`, publish/deploy/release
outcomes, and `gh pr/issue/release` parsing.

`UserPromptSubmit` capture added for preferences, decisions, dated
facts, corrections, and personal/work context.

New: `ogham hooks inscribe --dry-run` previews what would be stored
without writing or mutating dedup state.

All extraction is heuristic -- no LLM calls, no embedding calls, hook
latency unchanged.

### Fixed: Supabase parity for lifecycle, graph, density, and suggestions

Six call sites outside `backends/` were calling internal `_execute`
(a Postgres-only API) directly. `SupabaseBackend` doesn't have it, so
all six raised `AttributeError` on Supabase deployments since v0.11 --
swallowed silently in fire-and-forget background tasks (lifecycle,
Hebbian edges) and in caller-side `try/except` (density, hidden-link
suggestions). The user-visible symptom: the `suggest_connections` MCP
tool returned `[]` to every Supabase user, and the Lifecycle pipeline
counts on the dashboard were wrong.

Fix: published the operations as RPC functions via migration 035, and
routed all six call sites through the public backend facade methods.
Postgres deployments keep their fast inline-SQL path; Supabase
deployments now go through PostgREST RPC with the same SQL semantics.

If you've been running on Supabase, expect a one-time burst of
fresh→stable lifecycle transitions on the next search call -- that's
the backlog catching up.

### Fixed: build-system

- `make tag` now fails fast if `git commit --allow-empty` doesn't move
  HEAD. v0.13.0 shipped pointing at v0.12.1 code because a pre-commit
  hook silently aborted the empty commit; the release recipe kept
  going and tagged the wrong HEAD. Caught after PyPI publish and
  fixed forward.
- `demo-scripts/` no longer leaks into the sdist tarball.
- `make sync` SYNC_SOURCES list now includes `lifecycle.py`, `graph.py`,
  `tools/memory.py`, all test files, `docs/internals/hooks.md`, and
  globs migrations + rollbacks. Closes the v0.9.2 / v0.13.0 sync-gap
  class of bugs.
- `make publish-check` now audits sdist file count (100-400 expected)
  and forbidden paths (docs/, benchmarks/, .claude/, .github/,
  demo-scripts/, rollbacks, .env*, DANGER_*.sql). Catches bloat before
  it reaches PyPI.
- `make publish` now depends on `make smoke`, which installs the built
  wheel into a fresh venv and validates 13 module imports + the CLI
  entrypoint.

### Migration

Apply via Supabase SQL Editor (or `psql` for self-hosters):

```sql
\i sql/migrations/035_lifecycle_graph_rpcs.sql
```

Idempotent (`CREATE OR REPLACE FUNCTION` only; no data changes; no
table modifications). The migration tolerates missing
`memory_entities`/`entities` tables gracefully -- the entity-density
and hidden-link functions return empty results until those tables
exist (older deployments that predate the v0.10 entities feature).

**Install:** `uv tool install ogham-mcp` or
`pip install ogham-mcp==0.13.1`

**Full changelog:** https://github.com/ogham-mcp/ogham-mcp/compare/v0.13.0...v0.13.1

## [0.13.0] - 2026-04-28 -- Progressive recall

Multi-resolution topic summaries, an 8-dimension health readout, Go CLI
parity for the recall fast path, and a single canonical SQL tree.

### New: three resolutions for every wiki page

`compile_wiki` now produces three forms of every topic summary in a
single LLM call:

- `tldr_one_line` (~50 tokens) -- glanceable status-bar fit.
- `tldr_short` (~150-300 tokens) -- one-paragraph context preamble.
- `body` (~1000-2000 tokens) -- the full compiled page.

Two MCP tools gain a `level=` parameter so callers can pick the
resolution that fits their context budget:

- `query_topic_summary(topic, level="one_line"|"short"|"body")`
- `hybrid_search(query, wiki_preamble_level="short"|"body")` --
  **the default flipped from `body` to `short`**, a 3-5x token-cost
  reduction on the wiki preamble per query. Pass
  `wiki_preamble_level="body"` to keep v0.12 behavior.

Back-compat: pre-v0.13 rows have NULL `tldr_short` / `tldr_one_line`.
Asking for `level="short"` on a legacy row returns the body and reports
`level: "body"` with `requested_level: "short"` so the fallback is
visible in the response. Recompile with `compile_wiki(topic, force=True)`
to populate the new columns.

### New: 8-dimension `ogham health`

`ogham health` was binary green/red. v0.13 replaces it with a
score-out-of-10 readout across eight dimensions: DB freshness, schema
integrity, hybrid-search latency (p50/p95), corpus size, wiki coverage
(fresh vs stale summaries), profile health (avg tags + orphan %),
concurrency (pool busy + max wait), and an end-to-end probe
(store → search → delete round-trip).

`ogham health --json` emits the same structure for scripting.

### Changed: one canonical SQL tree (Phase B)

For most of v0.12, migrations lived in two places: `sql/migrations/`
(canonical) and `src/ogham/sql/migrations/` (mirror, shipped in the
wheel). v0.13 deletes the duplicate. Migrations now live only at
`sql/migrations/`. The wheel still ships them via hatchling
`force-include`, so `pip install ogham-mcp` users keep the same import
path. Self-host upgrade is unchanged.

### Migrations

Two new migrations. Apply in order via Supabase SQL Editor (or psql for
self-hosters):

```sql
\i sql/migrations/033_topic_summaries_tldr.sql
\i sql/migrations/034_wiki_topic_search_tldr.sql
```

`033` adds `tldr_one_line` and `tldr_short` columns + updates
`wiki_topic_upsert` to accept the new params.
`034` widens `wiki_topic_search` `RETURNS TABLE` to surface the new
columns so `hybrid_search` `wiki_preamble_level=` actually routes by
level (without 034, the RPC silently dropped the new columns and every
preamble request fell back to body content).

Idempotent. Pre-existing topic summaries keep their `content` column
populated and get NULL `tldr_short` / `tldr_one_line`. Recompile when
ready -- no surprise compute bill from a forced backfill.

### Added

- `tldr_one_line` and `tldr_short` columns on `topic_summaries`.
- `level=` parameter on `query_topic_summary` and
  `wiki_preamble_level=` on `hybrid_search`.
- `health_dimensions.py` module + 8-dim CLI rewire.
- pyright as CI quality gate.

### Changed

- Default `hybrid_search` wiki preamble: `body` → `short`.
- `compile_wiki` now generates three resolutions per call (no extra
  LLM round trips).

### Removed

- `src/ogham/sql/` duplicate migration tree. Migrations are now
  exclusively at `sql/migrations/`.

## [0.12.1] - 2026-04-27 -- hybrid_search shape + Supabase fix

### Fixed

- **`delete_memory`, `update_memory`, `reinforce_memory`, `contradict_memory`
  crashed on Supabase.** All four called a Postgres-only internal method
  to look up a memory's tags before mutating it, so Supabase users hit
  `AttributeError` on every delete / update / confidence operation in
  v0.12.0. Now uses the backend facade and works on every backend.

### Changed

- **`hybrid_search` returns `{"results": [...], "wiki_preamble": [...]}`**
  instead of a single mixed list. MCP clients render `wiki_preamble`
  separately as compiled topic context; benchmark scorers and downstream
  pipelines consume only `results` and see deterministic retrieval hits.
  `wiki_preamble` is always present (empty list when wiki injection is
  off, no summary clears the threshold, or `extract_facts=True`).

  **Breaking only for v0.12.0 callers who explicitly enabled
  `wiki_injection_enabled`** and consumed the mixed list directly.
  Everyone else was already getting a plain memory list which is now
  reachable via `result["results"]`. Update consumers accordingly.

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
