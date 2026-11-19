# Ogham MCP — Tool Reference

All MCP tools are defined in `tools/memory.py` and `tools/stats.py`, registered on the FastMCP instance from `app.py`.

## Session State

A module-level `_active_profile` variable tracks the current profile. All tools operate on this profile implicitly.

## Content Validation (`_require_content`)

Before storing, content is validated:

- Must be non-empty, ≥10 chars, ≤100,000 chars
- Rejects noise: git diffs, shell dumps, binary content (detected by regex patterns like `^diff --git`, `^@@`, null bytes)
- If ≥2 noise patterns match, raises `ValueError` suggesting the user store a summary instead

## Tool Catalog

### Profile Management

| Tool | Purpose |
|------|---------|
| `switch_profile(profile)` | Set session-level active profile |
| `current_profile()` | Return current profile name |
| `list_profiles()` | List all profiles with memory counts, marks active |

### Store

| Tool | Purpose |
|------|---------|
| `store_memory(content, source?, tags?, metadata?, auto_link?)` | Store via full enrichment pipeline (`service.store_memory_enriched`) |
| `store_decision(decision, rationale, alternatives?, reasoning_trace?, tags?, related_memories?, source?)` | Store a structured decision — assembles content from parts, adds `type:decision` tag, creates `supports` relationships to `related_memories` |

### Search & Retrieval

| Tool | Purpose |
|------|---------|
| `hybrid_search(query, limit?, tags?, source?, graph_depth?)` | Hybrid semantic + keyword search via `service.search_memories_enriched`. `graph_depth > 0` follows relationship edges |
| `list_recent(limit?, source?, tags?)` | Chronological list of recent memories |
| `explore_knowledge(query, depth?, min_strength?, limit?, tags?, source?)` | Graph-augmented exploration: seed search + N-hop relationship traversal via SQL function `explore_memory_graph()`. Records access for depth-0 results |
| `find_related(memory_id, relationship_types?, depth?, min_strength?, limit?)` | Impact analysis: traverse relationship graph from a specific memory |

### Mutation

| Tool | Purpose |
|------|---------|
| `update_memory(memory_id, content?, tags?, metadata?)` | Update fields. If content changes, re-generates embedding |
| `delete_memory(memory_id)` | Delete from active profile |
| `reinforce_memory(memory_id, strength?)` | Increase confidence score (0.5–1.0). Higher = stronger boost |
| `contradict_memory(memory_id, strength?)` | Decrease confidence score (0.0–0.5). Lower = stronger penalty |

### Maintenance

| Tool | Purpose |
|------|---------|
| `re_embed_all(ctx)` | Re-generate all embeddings after switching providers. Async, reports progress via MCP context. Clears embedding cache first, then batch-embeds all content |
| `compress_old_memories()` | Tiered compression: level 0→1 (gist), level 0/1→2 (tags). Preserves `original_content` for restoration |
| `link_unlinked(batch_size?, threshold?, max_links?)` | Backfill auto-links for memories without relationship edges |
| `cleanup_expired()` | Permanently delete expired memories (already hidden from search) |
| `set_profile_ttl(profile, ttl_days?)` | Configure auto-expiry for new memories in a profile |

### Export/Import

| Tool | Purpose |
|------|---------|
| `export_profile(format?)` | Export all memories as JSON or markdown |
| `import_memories_tool(data, dedup_threshold?)` | Import from JSON export, with similarity-based deduplication |

### Diagnostics

| Tool | Purpose |
|------|---------|
| `health_check()` | Check database + embedding provider connectivity |
| `get_cache_stats()` | Return embedding cache hit/miss statistics (in `tools/stats.py`) |

## Timing

All significant tools are wrapped with `@log_timing("tool_name")` — logs execution time in milliseconds to stderr.
