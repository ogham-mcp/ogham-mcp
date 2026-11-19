# BEAM Benchmark: Neural Sparse vs tsvector Hybrid Search

**Date:** 2026-03-27
**Database:** Local PostgreSQL with pgvector (sparsevec), no indexes on sparse column
**Dataset:** BEAM 100K bucket — 20 chats, 2,866 memories, 400 probing questions across 10 categories
**Embedding model:** BGE-M3 (1024-dim dense via Ollama, sparse via FlagEmbedding 1.3.5)

## Background

Ogham's hybrid search combines two retrieval signals:

- **Semantic leg (70%):** cosine similarity on dense vectors via pgvector HNSW
- **Keyword leg (30%):** a term-matching signal fused with the semantic score

The keyword leg has historically used PostgreSQL's built-in tsvector full-text search — bag-of-words with `ts_rank_cd` scoring. This works but has known limitations: no learned term importance, no sub-word matching, no contextual weighting.

BGE-M3 produces three vector types from a single model pass: dense (1024-dim), sparse (learned lexical weights), and ColBERT (per-token vectors). Ollama only exposes the dense vectors — the sparse and ColBERT heads are stripped in GGUF quantization. To get sparse vectors, we used FlagEmbedding's Python library which loads the full model.

The hypothesis: replacing tsvector with BGE-M3's neural sparse vectors in the keyword slot will improve retrieval quality, since the model learns contextual term importance rather than relying on raw term frequency.

## Method

### Dataset

[BEAM](https://github.com/mohammadtavakoli78/BEAM) (Tavakoli et al., ICLR 2026) is a benchmark for episodic and associative memory in conversational AI. Each of 20 chats contains a long multi-turn conversation (~100K tokens), and 20 probing questions across 10 categories that test different memory abilities.

### Ingestion

Each chat's conversation turns were grouped into user-assistant rounds. Each round became one memory with the date prepended (`[Date: ...]`), truncated to 10,000 characters. Dense embeddings were generated via Ollama (bge-m3, 1024-dim) with batch_size=5 and OLLAMA_TIMEOUT=300 to handle CPU latency.

| Stat | Value |
|------|-------|
| Chats ingested | 20 |
| Total memories | 2,866 |
| Dense embedding time | ~2.8 hours (CPU) |
| Dense embedding provider | Ollama (bge-m3 GGUF) |

### Sparse vector backfill

After ingestion, sparse vectors were generated for all 2,866 memories using FlagEmbedding's `BGEM3FlagModel.encode()` with `return_sparse=True`. This runs the full PyTorch bge-m3 model on CPU (fp32, no CUDA).

Sparse vectors were stored in a new `sparse_embedding sparsevec` column. No HNSW index — sequential scan is sufficient for the per-profile memory counts (44–280 per profile).

| Stat | Value |
|------|-------|
| Sparse embedding time | ~1.5 hours (CPU, fp32) |
| Sparse embedding provider | FlagEmbedding 1.3.5 (PyTorch) |
| Min non-zero elements | 43 |
| Max non-zero elements | 280 |
| Mean non-zero elements | 175 |
| Over pgvector 1,000 nnz limit | 0 (0%) |
| Vocabulary dimension | 250,002 |

The sparse vectors comfortably fit within pgvector's sparsevec limits. No VectorChord dependency needed.

### Search functions

Two SQL functions with identical structure except the keyword CTE:

- **`hybrid_search_memories`** (baseline): semantic CTE (dense cosine, top 3N) FULL OUTER JOIN keyword CTE (tsvector `ts_rank_cd`, top 3N), fused with `0.7 * semantic + 0.3 * keyword`, then multiplied by access/confidence/graph boosts.

- **`hybrid_search_memories_sparse`** (test): same semantic CTE, but keyword CTE replaced with sparse vector inner product (`-(sparse_embedding <#> query_sparse)`) against the query's sparse vector. Same fusion weights (0.7/0.3), same boosts.

### Evaluation

For each of the 400 probing questions:

1. Dense query embedding generated via Ollama
2. Sparse query vector generated via FlagEmbedding (sparse mode only)
3. `search_memories_enriched()` called with limit=50
4. Retrieved memories compared against gold-standard message IDs from the BEAM dataset
5. Metrics computed: Recall@K (K=5,10,20,30,50), NDCG@10, MRR

Each question was evaluated independently with both search functions. The dense vectors, memory content, and evaluation harness were identical — the only variable was the keyword component.

## Results

### Overall

| Metric | tsvector | sparse | delta |
|--------|----------|--------|-------|
| Recall@5 | 0.3855 | 0.5354 | **+0.1499** |
| Recall@10 | 0.4756 | 0.6158 | **+0.1403** |
| Recall@20 | 0.6003 | 0.7057 | **+0.1054** |
| Recall@30 | 0.6740 | 0.7532 | **+0.0792** |
| Recall@50 | 0.7619 | 0.8251 | **+0.0633** |
| NDCG@10 | 0.2675 | 0.4129 | **+0.1454** |
| MRR | 0.3241 | 0.4811 | **+0.1570** |

Neural sparse improves every metric. The gains are largest at small K (Recall@5: +0.15) and in ranking quality (NDCG: +0.15, MRR: +0.16), meaning relevant results are both found more often and ranked higher.

### Per category — Recall@10

| Category | tsvector | sparse | delta |
|----------|----------|--------|-------|
| abstention | 1.0000 | 1.0000 | +0.0000 |
| contradiction_resolution | 0.4929 | 0.8412 | **+0.3483** |
| knowledge_update | 0.5979 | 0.8458 | **+0.2479** |
| information_extraction | 0.4167 | 0.6958 | **+0.2792** |
| instruction_following | 0.3187 | 0.5458 | **+0.2271** |
| temporal_reasoning | 0.6750 | 0.8250 | **+0.1500** |
| multi_session_reasoning | 0.3953 | 0.5297 | **+0.1344** |
| summarization | 0.2876 | 0.3092 | +0.0217 |
| event_ordering | 0.1965 | 0.2157 | +0.0192 |
| preference_following | 0.3750 | 0.3500 | -0.0250 |

### Per category — MRR

| Category | tsvector | sparse | delta |
|----------|----------|--------|-------|
| knowledge_update | 0.4572 | 0.8081 | **+0.3509** |
| contradiction_resolution | 0.4991 | 0.7940 | **+0.2949** |
| multi_session_reasoning | 0.3610 | 0.6573 | **+0.2963** |
| temporal_reasoning | 0.5787 | 0.7899 | **+0.2112** |
| information_extraction | 0.3519 | 0.5319 | **+0.1800** |
| preference_following | 0.1917 | 0.3013 | **+0.1096** |
| event_ordering | 0.2348 | 0.2851 | +0.0503 |
| instruction_following | 0.2329 | 0.2725 | +0.0396 |
| summarization | 0.3333 | 0.3708 | +0.0375 |
| abstention | 0.0000 | 0.0000 | +0.0000 |

### Per difficulty

| Difficulty | tsvector R@10 | sparse R@10 | delta | tsvector MRR | sparse MRR | delta |
|-----------|--------------|-------------|-------|-------------|------------|-------|
| easy (n=124) | 0.5851 | 0.7592 | **+0.1741** | 0.4335 | 0.6885 | **+0.2550** |
| medium (n=193) | 0.3797 | 0.4596 | **+0.0799** | 0.2277 | 0.3054 | **+0.0777** |
| hard (n=42) | 0.5636 | 0.6868 | **+0.1232** | 0.2848 | 0.3894 | **+0.1046** |
| clear (n=40) | 0.4929 | 0.8412 | **+0.3483** | 0.4991 | 0.7940 | **+0.2949** |

## Analysis

### Where sparse wins big

**Contradiction resolution (+0.35 R@10, +0.29 MRR):** When a user corrects earlier information, the sparse model recognizes shared terminology between the correction and the original statement far better than bag-of-words. tsvector treats all terms equally; the sparse model learns which terms are semantically important in context.

**Knowledge update (+0.25 R@10, +0.35 MRR):** Similar to contradiction — updated facts share vocabulary with originals. The sparse model's learned weights surface the right memories; tsvector gets confused by term frequency.

**Information extraction (+0.28 R@10, +0.18 MRR):** Specific factual queries benefit from the sparse model's sub-word and contextual matching. A question about "sprint deadline" retrieves the right memory even when the stored text uses "sprint timeline" — the sparse model bridges these synonyms through learned weights.

**Temporal reasoning (+0.15 R@10, +0.21 MRR):** Date-related queries get better lexical matching. The sparse model assigns appropriate weight to temporal markers.

### Where sparse doesn't help

**Event ordering (+0.02 R@10):** This category requires reconstructing the sequence of events across multiple memories. No single retrieval improvement helps — the problem is that the answer is distributed across many memories and requires reasoning about order, not just finding the right one.

**Summarization (+0.02 R@10):** Similar — summarization queries need broad coverage of a topic across many memories. The semantic leg already finds topically relevant content; the keyword leg (sparse or tsvector) adds little because summarization questions are typically phrased abstractly.

**Preference following (-0.03 R@10, but +0.11 MRR):** Slight regression in recall but improvement in ranking. Preference queries often use vague language ("what do I prefer") where the sparse model's learned weights may not help recall but do help ranking.

### Ranking improvement is the bigger story

The MRR improvement (+0.16 overall) is arguably more important than the recall improvement (+0.14). In practice, ogham returns a limited number of results to the LLM context window. Having relevant results ranked first means the LLM sees them — having them buried at position 30 is nearly as bad as not finding them at all.

## Implications for ogham

### Production path

The benchmark validates that neural sparse vectors are a significant upgrade. To ship this:

1. **Runtime:** FlagEmbedding pulls PyTorch (~2GB). For production, investigate the [ONNX export](https://github.com/yuniko-software/bge-m3-onnx) which avoids PyTorch. Need to verify the ONNX model exposes the sparse head correctly.

2. **Integration:** Currently the sparse path is benchmark-only. To make it the default, FlagEmbedding (or ONNX) needs to be added as an embedding provider in `embeddings.py`, and `store_memory_enriched()` needs to store sparse vectors alongside dense vectors on every write.

3. **Migration:** Existing memories would need sparse vectors backfilled (same approach as this benchmark, but run against the production profile).

4. **Ollama coexistence:** FlagEmbedding and Ollama both load bge-m3 weights (~1.2GB each). On memory-constrained systems, running both simultaneously may be impractical. Consider using FlagEmbedding for both dense and sparse (it produces both in one pass) and dropping Ollama dependency for embedding.

### What sparse doesn't fix

Event ordering (R@10=0.22) and summarization (R@10=0.31) need a different approach:

- **ColBERT reranking** (Phase 4 in TODO.md): fine-grained token-level matching could help surface the specific memories that contain ordering or summary information.
- **Better chunking:** current round-based chunking may merge too many turns into one memory, diluting temporal signals.
- **Multi-memory reasoning:** these categories may ultimately need a retrieval-then-reason pipeline rather than better single-query retrieval.

## Reproduction

```bash
# Prerequisites: BEAM dataset at /tmp/BEAM, local postgres with pgvector

# 1. Ingest (creates dense embeddings via Ollama)
uv run python3 benchmarks/beam_benchmark.py --ingest --bucket 100K

# 2. Baseline eval (tsvector hybrid)
uv run python3 benchmarks/beam_benchmark.py --eval --bucket 100K

# 3. Backfill sparse vectors (requires FlagEmbedding)
uv run python3 benchmarks/beam_benchmark.py --backfill-sparse --bucket 100K

# 4. Sparse eval
uv run python3 benchmarks/beam_benchmark.py --eval --bucket 100K --search-mode sparse

# 5. Compare
uv run python3 benchmarks/beam_benchmark.py --compare --bucket 100K
```

## Raw data

Full per-question results are in:

- `benchmarks/beam_results/eval_100K_all.json` (tsvector)
- `benchmarks/beam_results/eval_100K_all_sparse.json` (sparse)
- `benchmarks/beam_results/ingest_100K.json` (ingestion stats)
