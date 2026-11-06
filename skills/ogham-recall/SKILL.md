---
name: ogham-recall
description: |
  Smart retrieval from Ogham shared memory. Use when the user wants to recall
  what they know, find related context, bootstrap session context, or explore
  their knowledge graph. Triggers on "what do I know about", "find related",
  "search ogham", "search memory", "recall", "context for this project",
  "what did we decide about", "any notes on", "check ogham for", or any
  request to retrieve or explore stored knowledge. Also use at session start
  when the user is about to work on something and could benefit from prior context.
  Requires the Ogham MCP server to be connected.
---

# Ogham recall

You retrieve knowledge from Ogham shared memory. Your job is to find relevant memories and surface connections the user might not know exist.

## Retrieval strategy

Don't just run one search and call it done. Different queries surface different results, and the knowledge graph often has useful connections that keyword search misses.

### Step 1: Search broadly

Start with `hybrid_search` using the user's query. This combines semantic similarity with keyword matching, so "us-east-1" and "which AWS region do we use" both work.

For queries that need connections between memories, use `graph_depth=1` to automatically follow relationship edges from the top results. This surfaces related memories that didn't match the query directly but are linked to matches.

If the results look thin, try rephrasing. A query about "database setup" might miss memories tagged with "postgres" or "supabase" -- try both angles.

### Step 2: Follow the graph

When search returns useful results, pick the most relevant memory IDs and run `find_related` on them. This walks the knowledge graph outward -- a memory about a database decision might link to memories about the schema, migration gotchas, or performance benchmarks.

For broader exploration, use `explore_knowledge` with a query. It searches first, then traverses relationship edges from the results. Set `depth=2` to go two hops out if the first level doesn't surface enough.

### Step 3: Check for decisions

If the user is asking about a past choice ("what did we decide about X", "why did we go with Y"), filter by decision-type memories. Decisions stored via `store_decision` have structured rationale and alternatives that give context regular memories don't.

Try: `hybrid_search(query="decision about X", tags=["type:decision"])`

## Context bootstrapping

When the user is starting work on something, proactively search for relevant context. Look at:

- The current project name (from CLAUDE.md, repo name, or working directory)
- What the user said they're about to do
- Recent git activity (what was changed recently)

Run 2-3 targeted searches to pull in useful background. Present it as a brief summary, not a wall of text. Something like:

"Found 4 relevant memories from previous sessions:

- You decided to use OpenAI embeddings at 512 dims because Mistral can't truncate (stored March 10)
- There's a gotcha with Neon PgBouncer -- ALTER TABLE silently fails on the pooler endpoint (stored March 8)
- The auto-link threshold of 0.85 produces very different results per provider (stored March 9)"

Let the user ask for more detail on any of these rather than dumping everything.

## Presenting results

Keep it scannable. For each relevant memory:

- One-line summary of what it says
- When it was stored (relative time is fine -- "last week", "March 10")
- Tags if they help (especially `type:decision` or `type:gotcha`)
- Memory ID only if the user might want to update or delete it

Group related memories together rather than listing them in search-rank order. If you found a decision and its supporting context via graph traversal, present them as a cluster.

## When nothing is found

If searches come back empty, say so directly. Don't make up context or hedge with "I didn't find anything but...". The user needs to know their knowledge base doesn't cover this topic yet.

Suggest storing what they learn: "Nothing in Ogham about X yet. Want me to store what we figure out?"
