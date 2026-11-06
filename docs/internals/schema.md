# Ogham MCP — PostgreSQL Schema

Source: `sql/schema_postgres.sql` (for vanilla PostgreSQL/Neon, no Supabase RLS).

## Extensions

- `pgvector` — vector similarity search

## Tables

### `memories`

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | uuid | `gen_random_uuid()` | PK |
| `content` | text | — | LZ4 TOAST compression |
| `embedding` | vector(512) | — | Indexed via HNSW (halfvec) |
| `metadata` | jsonb | `{}` | LZ4 TOAST, GIN indexed |
| `source` | text | null | e.g. "claude-code", "hook:post-tool" |
| `profile` | text | `'default'` | Memory namespace |
| `tags` | text[] | `{}` | GIN indexed, array overlap filtering |
| `created_at` | timestamptz | `now()` | |
| `updated_at` | timestamptz | `now()` | Auto-updated via trigger |
| `expires_at` | timestamptz | null | Partial B-tree index |
| `access_count` | integer | 0 | Bumped by `record_access()` |
| `last_accessed_at` | timestamptz | null | |
| `confidence` | float | 0.5 | Bayesian-updated via `update_confidence()` |
| `importance` | float | 0.5 | Set at store time |
| `surprise` | float | 0.5 | Novelty score (1 - max_similarity to existing) |
| `compression_level` | integer | 0 | 0=full, 1=gist, 2=tags |
| `original_content` | text | null | LZ4, preserved before compression |
| `occurrence_period` | tstzrange | null | GiST indexed |
| `recurrence_days` | int[] | null | GIN indexed, day-of-week (0=Sun..6=Sat) |
| `fts` | tsvector | generated | `to_tsvector('english', content)`, GIN indexed |

### `memory_relationships`

| Column | Type | Notes |
|--------|------|-------|
| `id` | bigint | IDENTITY PK |
| `source_id` | uuid | FK → memories, CASCADE delete |
| `target_id` | uuid | FK → memories, CASCADE delete |
| `relationship` | relationship_type | Enum |
| `strength` | float | 0.0–1.0 |
| `metadata` | jsonb | |
| `created_by` | text | `'auto'` or `'user'` |
| `created_at` | timestamptz | |
| **Constraint** | | `UNIQUE (source_id, target_id, relationship)` |

### `profile_settings`

| Column | Type | Notes |
|--------|------|-------|
| `profile` | text | PK |
| `ttl_days` | integer | CHECK ≥ 1 or NULL |

## Enum: `relationship_type`

`similar`, `related`, `contradicts`, `supports`, `follows`, `derived_from`

## Indexes

| Index | Type | On | Notes |
|-------|------|------|-------|
| `memories_embedding_idx` | HNSW | `embedding::halfvec(512)` | `halfvec_cosine_ops`, m=16, ef_construction=64 |
| `memories_metadata_idx` | GIN | `metadata` | `jsonb_path_ops` |
| `memories_tags_idx` | GIN | `tags` | Array overlap queries |
| `memories_fts_idx` | GIN | `fts` | Full-text search |
| `memories_profile_created_at_idx` | B-tree | `(profile, created_at DESC)` | |
| `memories_source_idx` | B-tree | `source` | |
| `memories_expires_at_idx` | B-tree (partial) | `expires_at` | WHERE expires_at IS NOT NULL |
| `idx_memories_occurrence` | GiST (partial) | `occurrence_period` | WHERE occurrence_period IS NOT NULL |
| `idx_memories_recurrence` | GIN (partial) | `recurrence_days` | WHERE recurrence_days IS NOT NULL |
| `idx_relationships_source` | B-tree | `(source_id, relationship)` | |
| `idx_relationships_target` | B-tree | `(target_id, relationship)` | |
| `idx_relationships_auto` | B-tree (partial) | `created_at` | WHERE created_by = 'auto' |

## Key Functions

### `match_memories()` — Pure Vector Search

**Relevance formula:**

```text
relevance = cosine_similarity
           × softplus(ACT-R_score)
           × confidence
           × (1 + graph_boost × 0.2)
```

**ACT-R component** (cognitive architecture model):

```text
B(M) = ln(access_count + 1) - 0.5 × ln(age_days / (access_count + 1))
softplus(B) = ln(1 + exp(B))
```

This models human memory: frequently-accessed and recently-accessed memories rank higher.

**Graph boost:** Sum of relationship edge strengths for all edges touching this memory, scaled by 0.2.

### `hybrid_search_memories()` — RRF Hybrid Search

Combines:

- **Semantic search**: Top 3×limit by cosine similarity (via HNSW index)
- **Keyword search**: Top 3×limit by `ts_rank_cd` on `websearch_to_tsquery`

**Fusion:**

```text
score = semantic_weight(0.7) × cosine_similarity
      + full_text_weight(0.3) × keyword_rank
```

**Final relevance:**

```text
relevance = score
           × (1 + ln(access_count + 1) × 0.1)
           × confidence
           × (1 + graph_boost × 0.2)
```

### `update_confidence()` — Bayesian Update

```text
posterior = (current × signal) / (current × signal + (1-current) × (1-signal))
new_confidence = 0.95 × posterior + 0.025
```

The `0.95/0.025` constants prevent confidence from reaching exactly 0 or 1 (laplace smoothing).

### `explore_memory_graph()` — Graph Traversal

Uses `WITH RECURSIVE`:

1. Seed from `hybrid_search_memories()` results
2. Traverse `memory_relationships` edges bidirectionally
3. Relevance propagates: `parent_relevance × edge_strength`
4. Dedup by ID (keep highest relevance)
5. Return ordered by depth ASC, relevance DESC

### `auto_link_memory()` — Auto-linking

1. Find top-N most similar memories (cosine > threshold) in same profile
2. Insert `'similar'` relationship edges
3. `ON CONFLICT DO NOTHING` prevents duplicate edges

### `batch_check_duplicates()` — Import Dedup

- Lowers `hnsw.ef_search` to 40 for fast approximate matching
- Returns boolean array: true if any memory exceeds similarity threshold
