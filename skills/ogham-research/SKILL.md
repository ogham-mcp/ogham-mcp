---
name: ogham-research
description: |
  Structured memory capture for Ogham shared memory. Use when the user wants to
  store findings, remember something, save what was learned, or capture a decision.
  Triggers on "remember this", "store this", "save this finding", "save what we learned",
  "capture this decision", "log this", or any request to persist knowledge to Ogham.
  Also use when the user says "store to ogham", "save to memory", or "remember for later".
  Requires the Ogham MCP server to be connected.
---

# Ogham research capture

You capture knowledge to Ogham shared memory. Your job is to store memories that are findable, deduplicated, and well-connected to existing knowledge.

## Before storing anything

Every memory goes through three checks:

### 1. Is it worth storing?

Not everything deserves a memory. Skip anything that's:

- Obvious from the code or docs (someone can just read the file)
- Temporary or in-progress (store when it's settled)
- Generic knowledge (how Python imports work, what Docker does)

Store things that would save time next session: decisions and why they were made, gotchas that surprised you, configuration that isn't obvious, patterns worth reusing.

### 2. Does it already exist?

Search before storing. Run `hybrid_search` with a query that captures the core idea. If something similar comes back (check the content, not just the score):

- **Same information, same detail level** -- skip it entirely
- **Same topic but your version adds detail** -- use `update_memory` on the existing one instead of creating a duplicate
- **Related but distinct** -- store it, the auto-linker will connect them

### 3. How should it be tagged?

Use a consistent scheme so memories are filterable later:

- `type:decision` -- why something was built a certain way
- `type:gotcha` -- bugs, workarounds, surprising behavior
- `type:pattern` -- conventions or approaches that worked
- `type:config` -- environment variables, service setup, deployment
- `type:architecture` -- how components connect
- `type:reference` -- links, docs, external resources
- `project:<name>` -- infer from the repo name, CLAUDE.md, or ask
- `branch:<name>` -- current git branch (auto-detect with `git branch --show-current`). Scope memories to the branch you're working on so they don't pollute search results on other branches. Skip this tag on `main`/`master` -- those memories are global.

Always set `source` to identify which client stored it (e.g. "claude-code", "cursor", "agent-zero").

## How to store

Write content that stands alone. Someone reading this memory in six months, in a different project, with no context about today's session, should understand it. Include the "why" not just the "what".

**Good**: "Ogham's match_memories RPC needs 'set search_path = public, extensions' for pgvector operators. Without it, you get 'operator does not exist: extensions.vector <=> extensions.vector'. The default search_path in Supabase functions is empty."

**Bad**: "Fixed the pgvector error by updating the search path."

The first one is searchable, specific, and includes the error message. The second one is useless to future-you.

## Parameter formatting

MCP tool parameters must be valid JSON types. Common mistakes to avoid:

- **tags**: must be a JSON array, NOT a string. Correct: `["type:decision", "project:ogham"]`. Wrong: `'["type:decision", "project:ogham"]'` (stringified array).
- **alternatives** (store_decision): same rule, must be a JSON array.
- **related_memories** (store_decision): same rule, must be a JSON array of UUID strings.
- **content**: plain string, no JSON encoding needed.
- **metadata**: must be a JSON object if provided, e.g. `{"source_url": "https://..."}`.

These rules apply to ALL Ogham MCP tools, including `update_memory` and `store_decision`, not just `store_memory`. The `update_memory` tool accepts `content`, `tags`, and `metadata` -- same formatting rules.

If an MCP call fails with a Pydantic validation error about "Input should be a valid list," you passed an array as a string. Fix the format and retry.

## For decisions, use store_decision

When the user is capturing a decision (chose X over Y, picked an architecture, settled a debate), use `store_decision` instead of `store_memory`. It structures the rationale and links to related context automatically.

```python
store_decision(
    decision="Use UUID primary keys instead of bigint",
    rationale="Supabase recommends UUIDs for distributed systems. Bigint requires sequences which don't work well across regions.",
    alternatives=["bigint with sequences", "ULID", "nanoid"],
    reasoning_trace="Evaluated 4 options. Bigint needs sequences which break across regions. ULID is sortable but adds a dependency. Nanoid is short but not universally supported. UUID is native to Postgres and Supabase recommends it.",
    tags=["type:decision", "project:ogham", "database"],
    related_memories=["<id-of-related-memory-if-known>"],
)
```

The `reasoning_trace` is optional but valuable. It captures the full chain of thought, not just the conclusion. When someone revisits this decision in 6 months, the trace tells them why the alternatives were rejected.

## After storing

Report what you did:

1. How many memories you checked for duplicates
2. How many you skipped (already existed)
3. How many you stored, with their tags
4. How many you updated (existing memories that needed more detail)

If you stored multiple memories, mention that auto-linking will connect related ones automatically.
