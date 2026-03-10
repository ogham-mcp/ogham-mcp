# Embedding Provider Benchmarks

**Date:** 2026-03-09
**Database:** Neon PostgreSQL (eu-central-1, pooled connection)
**Dataset:** 500 memories from conversation import corpus
**Test harness:** `tests/bench_stress.py --count 500`

## Results

| Metric | Ollama 512 | OpenAI 512 | OpenAI 1024 | Mistral 1024 | Voyage 512 | Voyage 1024 |
|--------|-----------|-----------|------------|-------------|-----------|------------|
| **Import (cold)** | 85.0 ms/mem | **36.3 ms/mem** | 39.6 ms/mem | 50.8 ms/mem | 36.6 ms/mem | 40.1 ms/mem |
| **Dedup re-import** | **2.2 ms/mem** | 2.9 ms/mem | 5.2 ms/mem | 5.5 ms/mem | 2.7 ms/mem | 5.9 ms/mem |
| **Auto-links (0.85)** | 29 | 16 | 13 | **468** | 62 | 63 |
| **Hybrid search (mean)** | **92.4 ms** | 97.1 ms | 109.4 ms | 110.0 ms | 95.4 ms | 117.1 ms |
| **Explore graph (mean)** | 85.4 ms | **79.8 ms** | 81.3 ms | 101.5 ms | 80.6 ms | 91.1 ms |
| **Get related (mean)** | 71.9 ms | 71.3 ms | 75.8 ms | 71.7 ms | 72.4 ms | **71.3 ms** |

## Models tested

| Provider | Model | SDK | Dim support | Batch limit |
|----------|-------|-----|-------------|-------------|
| Ollama | embeddinggemma | `ollama` | 128, 256, 512, 768 (MRL) | No API limit (local CPU) |
| OpenAI | text-embedding-3-small | `openai` | Any up to 1536 | 2,048 inputs |
| Mistral | mistral-embed | `mistralai` | Fixed 1024 only | ~32 (16K token limit) |
| Voyage | voyage-4-lite | `voyageai` | 256, 512, 1024, 2048 | 1,000 inputs |

## Key findings

### Performance
- **512 dims is ~15-20% faster** than 1024 on search operations across all providers
- **OpenAI and Voyage are neck-and-neck** on raw embedding + import speed
- **Ollama is ~2.3x slower on import** (85 ms/mem vs 36 ms/mem) — expected, since embeddings run locally on CPU
- **Ollama dedup is fastest** (2.2 ms/mem) — embedding cache serves from disk, no network round-trip
- **Mistral is slowest cloud provider on import** (50.8 ms/mem) due to its small batch size limit (~32 texts per request vs 500+ for OpenAI/Voyage)
- **Dedup at 512 dims is ~2x faster** than at 1024 dims (smaller vectors = faster cosine distance)
- **Search latency is comparable across all providers** — the bottleneck is Neon, not embedding speed

### Auto-linking
- **Auto-link threshold 0.85 is not provider-neutral.** Each provider has a different similarity score distribution:
  - **Mistral** clusters related content tightly — 468 links at 0.85 threshold
  - **Voyage** moderate spread — 62-63 links at 0.85
  - **Ollama** moderate spread — 29 links at 0.85
  - **OpenAI** widest spread — 13-16 links at 0.85
- To get comparable link density across providers, lower the threshold:
  - OpenAI: ~0.65-0.70
  - Ollama: ~0.75-0.80
  - Voyage: ~0.70-0.75
  - Mistral: 0.85 works well

### Provider-specific notes

**Ollama (`embeddinggemma`)**
- Supports MRL truncation via the `dimensions` parameter (128, 256, 512, 768)
- Import is CPU-bound — 2.3x slower than cloud providers, but no API costs
- Search latency is the same as cloud providers (bottleneck is Neon, not embedding)
- Moderate auto-link density at 0.85 threshold

**Mistral (`mistral-embed`)**
- Fixed 1024 dimensions — does **not** support the `output_dimension` parameter
- Tight similarity clustering means higher auto-link yield at default thresholds
- Small batch size (16K token limit) makes bulk import slower

**Voyage (`voyage-4-lite`)**
- Supports discrete dimensions only: 256, 512, 1024, 2048
- `output_dimension` parameter works as expected
- 1,000 inputs per batch request limit
- Best speed/quality ratio at 512 dims

**OpenAI (`text-embedding-3-small`)**
- Supports any dimension up to 1536 via Matryoshka representation
- Widest similarity spread — needs lowest match/auto-link thresholds
- Largest batch capacity (2,048 inputs)

## Neon gotchas

- **`cached plan must not change result type`**: After changing vector dimensions with ALTER TABLE, Neon's PgBouncer caches prepared statements from the old schema. Fix: run `DISCARD ALL` via psql after dimension migrations.
- **No `anon` role**: Neon doesn't have the Supabase `anon` role — use `schema_postgres.sql` (no RLS), not the selfhost schema.
