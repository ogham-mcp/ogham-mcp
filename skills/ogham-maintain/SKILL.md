---
name: ogham-maintain
description: |
  Admin and maintenance workflows for Ogham shared memory. Use when the user wants
  to clean up memories, review their knowledge graph, check memory stats, export
  their brain, re-embed memories after switching providers, or backfill links.
  Triggers on "clean up my memory", "memory stats", "how many memories",
  "export my brain", "export memories", "review knowledge graph", "re-embed",
  "link unlinked", "backfill links", "memory health", "ogham stats",
  "cleanup expired", "condense old memories", "compress memories",
  or any admin/maintenance request for Ogham.
  Requires the Ogham MCP server to be connected.
---

# Ogham maintenance

You handle admin tasks for Ogham shared memory. Most of these are infrequent operations -- provider switches, bulk cleanup, health checks.

## Available operations

### Health check

Run `health_check` first if the user reports problems. It tests database connectivity, embedding provider, and configuration. Report what it finds plainly -- if something is broken, say what and suggest a fix.

### Stats overview

Run `get_stats` and `list_profiles` to give the user a picture of their memory:

- Total memories and breakdown by profile
- Top sources (which clients are storing)
- Top tags (what categories dominate)
- Cache stats via `get_cache_stats` if they ask about performance

Present it as a concise summary, not raw JSON.

### Cleanup expired memories

1. Run `get_stats` to show how many memories exist
2. Check if any profiles have TTLs set (this info comes from `list_profiles`)
3. If there are expired memories, tell the user how many before running `cleanup_expired`
4. Run `cleanup_expired` only after confirming with the user -- deletion is permanent

### Export

Run `export_profile` with the format the user wants (JSON or Markdown). Tell them where the output goes and how to use it.

If they want to export a specific profile, switch to it first with `switch_profile`, export, then switch back.

### Re-embed all memories

This is needed after switching embedding providers (e.g. Ollama to OpenAI). It regenerates every vector in the active profile.

Before running:

1. Confirm the user has switched providers in their config
2. Warn that this takes time -- roughly 100ms per memory with a remote provider
3. Run `re_embed_all` which reports progress as it goes

After: suggest running `link_unlinked` to rebuild the knowledge graph with the new embeddings, since similarity scores will be different.

### Backfill knowledge graph links

`link_unlinked` scans memories that don't have relationship edges yet and creates links where embedding similarity is above threshold.

- Default threshold 0.85 is conservative -- only very similar memories get linked
- Suggest 0.7 for broader connections in diverse collections
- The user can set `batch_size` to control how many are processed at once

Report how many links were created when it finishes.

### Condense old memories

`compress_old_memories` shrinks old, inactive memories to save space and reduce search noise.

Three levels:

- Full text (default, recent memories)
- Condensed (key sentences, code blocks preserved, ~30% of original)
- Trace (one-line summary with tags)

Before running:

1. Explain that condensing is based on age and activity -- important, frequently-accessed, or high-confidence memories resist condensing
2. Explain that original content is always preserved and can be restored
3. Run `compress_old_memories` -- it reports how many were condensed at each level

### Profile management

- `list_profiles` -- show all profiles with memory counts
- `switch_profile` -- change active profile (session only)
- `set_profile_ttl` -- set auto-expiry. Explain that expired memories are filtered from searches immediately but not deleted until `cleanup_expired` runs
- To remove a TTL, call `set_profile_ttl` with `ttl_days=None`

## General approach

These are power-user operations. Be direct about what each one does, what it costs (time, data loss), and whether it's reversible. Deletion and re-embedding are not reversible. Exports, stats, and health checks are read-only and safe to run anytime.

If the user asks for something vague like "clean up my ogham", start with stats to understand what they have, then suggest specific actions based on what you see.
