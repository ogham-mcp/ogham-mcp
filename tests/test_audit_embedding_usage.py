from unittest.mock import patch

import pytest

FAKE_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.config import settings

    settings._reset()
    yield
    settings._reset()


@pytest.fixture
def isolated_cache(tmp_path):
    import ogham.embeddings as emb
    from ogham.embedding_cache import EmbeddingCache

    old_cache = emb._cache
    emb._cache = EmbeddingCache(cache_dir=str(tmp_path), max_size=20)
    try:
        yield emb
    finally:
        emb._cache = old_cache


def test_cache_key_includes_model(monkeypatch):
    from ogham.config import settings
    from ogham.embeddings import _cache_key

    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-4-lite")
    settings._reset()
    key_one = _cache_key("same text")

    monkeypatch.setenv("VOYAGE_EMBED_MODEL", "voyage-4")
    settings._reset()
    key_two = _cache_key("same text")

    assert key_one != key_two


def test_generate_embedding_usage_out_cache_hit_returns_zero_tokens(isolated_cache):
    from ogham.embeddings import generate_embedding

    def fake_generate(text, usage_out=None):
        if usage_out is not None:
            usage_out.update({"model": "openai:text-embedding-3-small", "input_tokens": 25})
        return FAKE_EMBEDDING

    with patch("ogham.embeddings._generate_uncached", side_effect=fake_generate) as mock_gen:
        first_usage = {}
        second_usage = {}
        generate_embedding("cached text", usage_out=first_usage)
        generate_embedding("cached text", usage_out=second_usage)

    assert mock_gen.call_count == 1
    assert first_usage == {"model": "openai:text-embedding-3-small", "input_tokens": 25}
    assert second_usage == {
        "model": "openai:text-embedding-3-small",
        "input_tokens": 0,
        "cache_hit": True,
    }


def test_generate_embeddings_batch_usage_out_counts_uncached_only(isolated_cache):
    from ogham.embeddings import _cache_key, generate_embeddings_batch

    isolated_cache._cache.put(_cache_key("cached"), FAKE_EMBEDDING)

    def fake_batch_generate(texts, usage_out=None):
        if usage_out is not None:
            usage_out.update({"model": "openai:text-embedding-3-small", "input_tokens": 40})
        return [[0.2] * 1024 for _ in texts]

    with patch(
        "ogham.embeddings._generate_batch_uncached",
        side_effect=fake_batch_generate,
    ) as mock_batch:
        usage = {}
        embeddings = generate_embeddings_batch(
            ["cached", "new one", "new two"],
            usage_out=usage,
        )

    assert len(embeddings) == 3
    assert embeddings[0] == FAKE_EMBEDDING
    assert mock_batch.call_count == 1
    assert mock_batch.call_args.args[0] == ["new one", "new two"]
    assert usage == {"model": "openai:text-embedding-3-small", "input_tokens": 40}


def test_calculate_embedding_cost_uses_repo_model_rate():
    from ogham.pricing import calculate_embedding_cost

    cost = calculate_embedding_cost({"model": "openai:text-embedding-3-small", "input_tokens": 250})

    assert cost == pytest.approx(0.000005)


def test_calculate_embedding_cost_returns_none_for_unknown_and_gemini_models():
    from ogham.pricing import calculate_embedding_cost

    assert calculate_embedding_cost({"model": "voyage:voyage-unknown", "input_tokens": 100}) is None
    assert (
        calculate_embedding_cost(
            {"model": "gemini:gemini-embedding-2-preview", "input_tokens": 100}
        )
        is None
    )
    assert calculate_embedding_cost({"model": "ollama:embeddinggemma"}) == 0.0


def test_store_memory_audit_includes_embedding_usage():
    from ogham.service import store_memory_enriched

    with (
        patch(
            "ogham.service.generate_embedding",
            side_effect=lambda text, usage_out=None: (
                (
                    usage_out.update(
                        {"model": "openai:text-embedding-3-small", "input_tokens": 250}
                    )
                    if usage_out is not None
                    else None
                )
                or FAKE_EMBEDDING
            ),
        ),
        patch("ogham.service.hybrid_search_memories", return_value=[]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-123", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.emit_audit_event") as audit,
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda text: text),
    ):
        result = store_memory_enriched(
            content="This memory is long enough to pass validation.",
            profile="default",
            source="test",
        )

    assert result["status"] == "stored"
    audit.assert_called_once()
    assert audit.call_args.kwargs["embedding_model"] == "openai:text-embedding-3-small"
    assert audit.call_args.kwargs["tokens_used"] == 250
    assert audit.call_args.kwargs["cost_usd"] == pytest.approx(0.000005)


def test_store_memory_precomputed_embedding_leaves_usage_null():
    from ogham.service import store_memory_enriched

    with (
        patch("ogham.service.generate_embedding") as generate,
        patch("ogham.service.hybrid_search_memories", return_value=[]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-123", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.emit_audit_event") as audit,
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda text: text),
    ):
        store_memory_enriched(
            content="This memory is long enough to pass validation.",
            profile="default",
            source="test",
            embedding=FAKE_EMBEDDING,
        )

    generate.assert_not_called()
    assert "embedding_model" not in audit.call_args.kwargs
    assert "tokens_used" not in audit.call_args.kwargs
    assert "cost_usd" not in audit.call_args.kwargs


def test_search_audit_accumulates_full_embedding_usage():
    from ogham.service import search_memories_enriched

    usage_calls = [
        (FAKE_EMBEDDING, {"model": "openai:text-embedding-3-small", "input_tokens": 10}),
        (FAKE_EMBEDDING, {"model": "openai:text-embedding-3-small", "input_tokens": 20}),
        (FAKE_EMBEDDING, {"model": "openai:text-embedding-3-small", "input_tokens": 30}),
    ]
    bridge_result = {"id": "mem-1", "content": "bridge result", "relevance": 0.9}

    with (
        patch(
            "ogham.service.generate_embedding",
            side_effect=lambda text, usage_out=None: (
                (usage_out.update(usage_calls.pop(0)[1]) if usage_out is not None else None)
                or FAKE_EMBEDDING
            ),
        ),
        patch("ogham.service.is_ordering_query", return_value=False),
        patch("ogham.service.is_multi_hop_temporal", return_value=True),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=False),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_query_anchors", return_value=["alpha", "beta"]),
        patch("ogham.service.hybrid_search_memories", return_value=[bridge_result]),
        patch("ogham.service._merge_bridge_results", return_value=[bridge_result]),
        patch("ogham.service._entity_thread", side_effect=lambda results, *_: results),
        patch("ogham.service.record_access"),
        patch("ogham.service.emit_audit_event") as audit,
    ):
        results = search_memories_enriched("When did alpha and beta happen?", profile="default")

    assert results == [bridge_result]
    audit.assert_called_once()
    assert audit.call_args.kwargs["embedding_model"] == "openai:text-embedding-3-small"
    assert audit.call_args.kwargs["tokens_used"] == 60
    assert audit.call_args.kwargs["cost_usd"] == pytest.approx(0.0000012)


def test_search_precomputed_embedding_leaves_usage_null_without_extra_embeds():
    from ogham.service import search_memories_enriched

    result = {"id": "mem-1", "content": "result", "relevance": 0.8}

    with (
        patch("ogham.service.generate_embedding") as generate,
        patch("ogham.service.is_ordering_query", return_value=False),
        patch("ogham.service.is_multi_hop_temporal", return_value=False),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=False),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.hybrid_search_memories", return_value=[result]),
        patch("ogham.service.record_access"),
        patch("ogham.service.emit_audit_event") as audit,
    ):
        results = search_memories_enriched(
            "query",
            profile="default",
            embedding=FAKE_EMBEDDING,
        )

    assert results == [result]
    generate.assert_not_called()
    assert "embedding_model" not in audit.call_args.kwargs
    assert "tokens_used" not in audit.call_args.kwargs
    assert "cost_usd" not in audit.call_args.kwargs


def test_postgres_audit_insert_includes_cost_columns():
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()

    with patch.object(backend, "_execute") as execute:
        backend.emit_audit_event(
            profile="default",
            operation="search",
            embedding_model="openai:text-embedding-3-small",
            tokens_used=123,
            cost_usd=0.00000246,
        )

    sql = execute.call_args.args[0]
    params = execute.call_args.args[1]
    assert "embedding_model" in sql
    assert "tokens_used" in sql
    assert "cost_usd" in sql
    assert params["embedding_model"] == "openai:text-embedding-3-small"
    assert params["tokens_used"] == 123
    assert params["cost_usd"] == pytest.approx(0.00000246)
    assert execute.call_args.kwargs["fetch"] == "none"
