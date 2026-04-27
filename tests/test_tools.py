import hashlib
from unittest.mock import patch

import pytest

FAKE_ID = "a1b2c3d4-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")
    # Tests assume the resolution order falls through to settings.default_profile
    # = "default". The OGHAM_PROFILE env var would short-circuit that.
    monkeypatch.delenv("OGHAM_PROFILE", raising=False)


@pytest.fixture(autouse=True)
def reset_profile(monkeypatch):
    """Reset profile to default between tests.

    get_active_profile() reads four sources in order:
      OGHAM_PROFILE env > ~/.ogham/active_profile sentinel > in-memory > settings
    Tests want the in-memory branch to win, so we both reset the in-memory
    flag AND stub the sentinel file reader (so Kevin's actual ~/.ogham
    state, which says "work", doesn't bleed into tests). settings is also
    pinned to "default" because it can drift with config.env.
    """
    import ogham.tools.memory as mem

    monkeypatch.setattr(mem, "_read_active_profile_sentinel", lambda: None)
    monkeypatch.setattr("ogham.config.settings.default_profile", "default")
    mem._active_profile = "default"
    yield
    mem._active_profile = "default"


@pytest.fixture
def mock_embedding():
    with (
        patch("ogham.tools.memory.generate_embedding") as mock,
        patch("ogham.embeddings.generate_embedding", mock),
        patch("ogham.service.generate_embedding", mock),
    ):
        mock.return_value = [0.1] * 1024
        yield mock


@pytest.fixture
def mock_db():
    # delete/update/reinforce/contradict pre-fetch tags via get_memory_by_id
    # (the backend facade) before mutating. The pre-fetch needs a fixture
    # answer; otherwise the tools crash on get_memory_by_id returning None.
    # Patch targets follow the actual import locations:
    #   - get_memory_by_id, emit_audit_event: imported inline from ogham.database,
    #     so the module-level name to patch is `ogham.database.<name>`.
    #   - enqueue_for_tags: imported inline from ogham.recompute_executor.
    #   - db_update_confidence: imported at the top of tools/memory.py.
    with (
        patch("ogham.service.db_store") as store,
        patch("ogham.service.hybrid_search_memories") as search,
        patch("ogham.service.record_access") as rec_access,
        patch("ogham.tools.memory.record_access", new=rec_access),
        patch("ogham.service.db_get_profile_ttl") as get_ttl,
        patch("ogham.service.db_auto_link") as auto_link,
        patch("ogham.tools.memory.list_recent_memories") as list_recent,
        patch("ogham.tools.memory.db_delete") as delete,
        patch("ogham.tools.memory.db_update") as update,
        patch("ogham.database.get_memory_by_id") as get_by_id,
        patch("ogham.database.emit_audit_event"),
        patch("ogham.recompute_executor.enqueue_for_tags") as enqueue,
        patch("ogham.tools.memory.db_update_confidence") as update_conf,
    ):
        store.return_value = {
            "id": FAKE_ID,
            "created_at": "2026-01-01T00:00:00Z",
        }
        search.return_value = [
            {
                "id": FAKE_ID,
                "content": "test",
                "similarity": 0.95,
                "keyword_rank": 0.3,
                "relevance": 0.66,
                "tags": [],
                "source": None,
                "profile": "default",
                "access_count": 0,
                "last_accessed_at": None,
                "confidence": 0.5,
            }
        ]
        list_recent.return_value = [
            {
                "id": FAKE_ID,
                "content": "test",
                "tags": [],
                "source": None,
                "profile": "default",
            }
        ]
        delete.return_value = True
        update.return_value = {
            "id": FAKE_ID,
            "updated_at": "2026-01-02T00:00:00Z",
        }
        get_ttl.return_value = None
        auto_link.return_value = 0
        get_by_id.return_value = {"id": FAKE_ID, "tags": [], "profile": "default"}
        update_conf.return_value = 0.85
        yield {
            "store": store,
            "search": search,
            "list_recent": list_recent,
            "delete": delete,
            "update": update,
            "get_ttl": get_ttl,
            "record_access": rec_access,
            "auto_link": auto_link,
            "get_by_id": get_by_id,
            "enqueue": enqueue,
            "update_conf": update_conf,
        }


# --- Profile tools ---


def test_switch_profile():
    from ogham.tools.memory import current_profile, switch_profile

    assert current_profile()["profile"] == "default"
    result = switch_profile(profile="work")
    assert result["from"] == "default"
    assert result["to"] == "work"
    assert current_profile()["profile"] == "work"


def test_list_profiles():
    from ogham.tools.memory import list_profiles

    with patch("ogham.tools.memory.db_list_profiles") as mock_lp:
        mock_lp.return_value = [
            {"profile": "default", "count": 5},
            {"profile": "work", "count": 3},
        ]
        result = list_profiles()

    assert len(result) == 2
    assert result[0]["active"] is True  # default is active
    assert "active" not in result[1]


# --- Memory tools with profile ---


def test_store_memory(mock_embedding, mock_db):
    from ogham.tools.memory import store_memory

    result = store_memory(content="test memory", source="test", tags=["tag1"])
    assert result["status"] == "stored"
    assert result["profile"] == "default"
    mock_db["store"].assert_called_once()
    call_kwargs = mock_db["store"].call_args[1]
    assert call_kwargs["profile"] == "default"


def test_store_memory_uses_active_profile(mock_embedding, mock_db):
    from ogham.tools.memory import store_memory, switch_profile

    switch_profile(profile="personal")
    result = store_memory(content="secret stuff")
    assert result["profile"] == "personal"
    call_kwargs = mock_db["store"].call_args[1]
    assert call_kwargs["profile"] == "personal"


def test_hybrid_search(mock_embedding, mock_db):
    """v0.12.1 split shape: hybrid_search returns dict with results +
    wiki_preamble keys. wiki_preamble defaults to [] when injection is
    off (which it is here, since the fixture doesn't enable it)."""
    from ogham.tools.memory import hybrid_search

    out = hybrid_search(query="test query")
    assert isinstance(out, dict)
    results = out["results"]
    assert out["wiki_preamble"] == []
    assert len(results) == 1
    assert results[0]["similarity"] == 0.95
    assert results[0]["keyword_rank"] == 0.3
    assert "relevance" in results[0]
    assert "confidence" in results[0]
    mock_db["search"].assert_called_once()
    call_kwargs = mock_db["search"].call_args[1]
    assert call_kwargs["query_text"] == "test query"
    assert call_kwargs["profile"] == "default"
    mock_db["record_access"].assert_called_once_with([FAKE_ID])


def test_list_recent(mock_embedding, mock_db):
    from ogham.tools.memory import list_recent

    results = list_recent(limit=5)
    assert len(results) == 1
    mock_db["list_recent"].assert_called_once_with(
        profile="default", limit=5, source=None, tags=None
    )


def test_delete_memory(mock_embedding, mock_db):
    from ogham.tools.memory import delete_memory

    result = delete_memory(memory_id=FAKE_ID)
    assert result["status"] == "deleted"
    mock_db["delete"].assert_called_once_with(FAKE_ID, profile="default")


def test_delete_memory_wrong_profile(mock_embedding, mock_db):
    """delete_memory should only delete from active profile"""
    from ogham.tools.memory import delete_memory, switch_profile

    switch_profile(profile="work")
    mock_db["delete"].return_value = False

    result = delete_memory(memory_id=FAKE_ID)
    assert result["status"] == "not_found"

    mock_db["delete"].assert_called_once_with(FAKE_ID, profile="work")


def test_update_memory_with_content(mock_embedding, mock_db):
    from ogham.tools.memory import update_memory

    result = update_memory(memory_id=FAKE_ID, content="updated content")
    assert result["status"] == "updated"
    mock_embedding.assert_called_once_with("updated content")
    mock_db["update"].assert_called_once()
    call_args = mock_db["update"].call_args
    assert call_args[0][0] == FAKE_ID
    assert "embedding" in call_args[0][1]
    assert call_args[1]["profile"] == "default"


def test_update_memory_wrong_profile(mock_embedding, mock_db):
    """update_memory should only update in active profile"""
    from unittest.mock import MagicMock

    from ogham.tools.memory import switch_profile, update_memory

    switch_profile(profile="work")
    mock_result = MagicMock()
    mock_result.data = []
    mock_db["update"].side_effect = KeyError("not found in profile 'work'")

    with pytest.raises(KeyError, match="not found in profile"):
        update_memory(memory_id=FAKE_ID, content="updated content for this memory")

    mock_db["update"].assert_called_once()
    call_args = mock_db["update"].call_args
    assert call_args[0][0] == FAKE_ID
    assert call_args[1]["profile"] == "work"


def test_update_memory_no_changes(mock_embedding, mock_db):
    from ogham.tools.memory import update_memory

    result = update_memory(memory_id=FAKE_ID)
    assert result["status"] == "no_changes"
    mock_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_re_embed_all(mock_embedding, mock_db):
    from unittest.mock import AsyncMock, MagicMock

    from ogham.tools.memory import re_embed_all

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.report_progress = AsyncMock()

    with (
        patch("ogham.tools.memory.get_all_memories_content") as mock_get_all,
        patch("ogham.tools.memory.generate_embeddings_batch") as mock_gen_batch,
        patch("ogham.tools.memory.batch_update_embeddings") as mock_batch,
    ):
        mock_get_all.return_value = [
            {"id": FAKE_ID, "content": "memory one"},
            {"id": "b2c3d4e5-0000-0000-0000-000000000002", "content": "memory two"},
        ]
        mock_gen_batch.return_value = [[0.1] * 512, [0.2] * 512]
        mock_batch.return_value = 2
        result = await re_embed_all(mock_ctx)

    assert result["status"] == "complete"
    assert result["profile"] == "default"
    assert result["total"] == 2
    assert result["succeeded"] == 2
    assert result["failed"] == 0
    mock_get_all.assert_called_once_with(profile="default")
    mock_ctx.report_progress.assert_called_once_with(2, 2)
    mock_batch.assert_called_once()


@pytest.mark.asyncio
async def test_re_embed_all_empty(mock_embedding, mock_db):
    from unittest.mock import AsyncMock, MagicMock

    from ogham.tools.memory import re_embed_all

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.report_progress = AsyncMock()

    with patch("ogham.tools.memory.get_all_memories_content") as mock_get_all:
        mock_get_all.return_value = []
        result = await re_embed_all(mock_ctx)

    assert result["status"] == "nothing_to_do"
    assert result["total"] == 0
    mock_embedding.assert_not_called()
    mock_ctx.report_progress.assert_not_called()


def test_update_memory_not_found(mock_embedding):
    """update_memory should raise KeyError when memory_id doesn't exist"""
    from unittest.mock import MagicMock

    from ogham.database import update_memory

    mock_result = MagicMock()
    mock_result.data = []

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_update = mock_client.return_value.from_.return_value.update
        mock_update.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        with pytest.raises(KeyError, match="not found in profile"):
            update_memory(
                memory_id="nonexistent-uuid", updates={"content": "test"}, profile="default"
            )


def test_store_memory_insert_failure(mock_embedding):
    """store_memory should raise RuntimeError when insert returns no data"""
    from unittest.mock import MagicMock

    from ogham.database import store_memory

    mock_result = MagicMock()
    mock_result.data = []

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.from_.return_value.insert.return_value.execute.return_value = (
            mock_result
        )

        with pytest.raises(RuntimeError, match="Insert returned no data"):
            store_memory(content="test", embedding=[0.1] * 1024, profile="default")


# --- Input validation tests ---


def test_store_memory_empty_content():
    from ogham.tools.memory import store_memory

    with pytest.raises(ValueError, match="non-empty string"):
        store_memory(content="")


def test_store_memory_whitespace_only():
    from ogham.tools.memory import store_memory

    with pytest.raises(ValueError, match="non-empty string"):
        store_memory(content="   \n\t  ")


def test_store_memory_too_long():
    from ogham.tools.memory import store_memory

    long_content = "x" * 100_001
    with pytest.raises(ValueError, match="exceeds maximum length"):
        store_memory(content=long_content)


def test_store_memory_too_short():
    from ogham.tools.memory import store_memory

    with pytest.raises(ValueError, match="too short"):
        store_memory(content="hello")


def test_store_memory_diff_noise_rejected():
    from ogham.tools.memory import store_memory

    diff_content = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+import os\n"
        " import sys\n"
    )
    with pytest.raises(ValueError, match="diff"):
        store_memory(content=diff_content)


def test_store_memory_valid_content_passes():
    """Content that mentions diffs in prose should NOT be rejected."""
    from ogham.tools.memory import _require_content

    # This mentions "diff" but isn't a raw diff dump
    _require_content("We ran git diff and found that the migration was missing the new column.")


def test_hybrid_search_no_threshold_param(mock_embedding, mock_db):
    """hybrid_search should not accept a threshold parameter (hybrid search uses RRF ranking)"""
    import inspect

    from ogham.tools.memory import hybrid_search

    sig = inspect.signature(hybrid_search)
    assert "threshold" not in sig.parameters


def test_list_recent_limit_too_large():
    from ogham.tools.memory import list_recent

    with pytest.raises(ValueError, match="limit must be between"):
        list_recent(limit=10000)


def test_list_recent_limit_zero():
    from ogham.tools.memory import list_recent

    with pytest.raises(ValueError, match="limit must be between"):
        list_recent(limit=0)


# --- Config validation tests ---


def test_config_validation_invalid_provider():
    """Config should reject invalid embedding_provider at load time"""
    import os

    from pydantic import ValidationError

    old_val = os.environ.get("EMBEDDING_PROVIDER")
    os.environ["EMBEDDING_PROVIDER"] = "invalid_provider"

    try:
        from ogham.config import Settings

        with pytest.raises(ValidationError, match="embedding_provider must be one of"):
            Settings()
    finally:
        if old_val is not None:
            os.environ["EMBEDDING_PROVIDER"] = old_val
        else:
            os.environ.pop("EMBEDDING_PROVIDER", None)


# --- Health check tests ---


def test_health_check():
    from ogham.tools.memory import health_check

    with patch("ogham.health.check_database") as mock_database:
        with patch("ogham.health.check_embedding_provider") as mock_embedding:
            mock_database.return_value = {"status": "ok", "connected": True}
            mock_embedding.return_value = {"status": "ok", "provider": "ollama"}

            result = health_check()

            assert result["database"]["status"] == "ok"
            assert result["embedding"]["status"] == "ok"


def test_check_database_success():
    from unittest.mock import MagicMock

    from ogham.health import check_database

    mock_backend = MagicMock()
    chain = mock_backend._get_client.return_value.from_.return_value
    chain.select.return_value.limit.return_value.execute.return_value = MagicMock()

    with patch("ogham.health.get_backend", return_value=mock_backend):
        result = check_database()

        assert result["status"] == "ok"
        assert result["connected"] is True


def test_check_database_failure():
    from ogham.health import check_database

    with patch("ogham.health.get_backend", side_effect=Exception("Connection refused")):
        result = check_database()

        assert result["status"] == "error"
        assert result["connected"] is False
        assert isinstance(result["error"], str)
        assert "Connection refused" in result["error"]


def test_check_embedding_provider_ollama_success():
    from unittest.mock import MagicMock

    from ogham.health import check_embedding_provider

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.embedding_provider = "ollama"
        mock_settings.ollama_url = "http://localhost:11434"

        with patch("ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = []
            mock_ollama.return_value = mock_client

            result = check_embedding_provider()

            assert result["status"] == "ok"
            assert result["provider"] == "ollama"


def test_check_embedding_provider_ollama_failure():
    from ogham.health import check_embedding_provider

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.embedding_provider = "ollama"
        mock_settings.ollama_url = "http://localhost:11434"

        with patch("ollama.Client") as mock_ollama:
            mock_ollama.side_effect = Exception("Connection refused")

            result = check_embedding_provider()

            assert result["status"] == "error"
            assert result["provider"] == "ollama"
            assert "hint" in result


def test_check_embedding_provider_openai_success():
    from ogham.health import check_embedding_provider

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "sk-test-key"

        result = check_embedding_provider()

        assert result["status"] == "ok"
        assert result["provider"] == "openai"


def test_check_embedding_provider_openai_no_key():
    from ogham.health import check_embedding_provider

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = None

        result = check_embedding_provider()

        assert result["status"] == "error"
        assert isinstance(result["error"], str)
        assert "OPENAI_API_KEY not set" in result["error"]


def test_check_config_valid():
    from ogham.health import check_config

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.database_backend = "supabase"
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_key = "test-key"
        mock_settings.embedding_dim = 1024

        result = check_config()

        assert result["status"] == "ok"
        assert result["issues"] == []


def test_check_config_missing_url():
    from ogham.health import check_config

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.database_backend = "supabase"
        mock_settings.supabase_url = ""
        mock_settings.supabase_key = "test-key"
        mock_settings.embedding_dim = 1024

        result = check_config()

        assert result["status"] == "warning"
        assert isinstance(result["issues"], list)
        assert "SUPABASE_URL not set" in result["issues"]


def test_check_config_unusual_embedding_dim():
    from ogham.health import check_config

    with patch("ogham.health.settings") as mock_settings:
        mock_settings.database_backend = "supabase"
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_key = "test-key"
        mock_settings.embedding_dim = 999

        result = check_config()

        assert result["status"] == "warning"
        assert isinstance(result["issues"], list)
        assert any("Unusual embedding_dim" in issue for issue in result["issues"])


# --- Embedding cache tests ---


def test_embedding_cache():
    """Identical content should use cached embedding"""
    from ogham.embeddings import clear_embedding_cache, generate_embedding

    clear_embedding_cache()

    with patch("ogham.embeddings._generate_uncached") as mock_gen:
        mock_gen.return_value = [0.1] * 1024

        result1 = generate_embedding("test content")
        assert mock_gen.call_count == 1

        result2 = generate_embedding("test content")
        assert mock_gen.call_count == 1

        assert result1 == result2


def test_embedding_cache_different_content():
    """Different content should generate new embedding"""
    from ogham.embeddings import clear_embedding_cache, generate_embedding

    clear_embedding_cache()

    with patch("ogham.embeddings._generate_uncached") as mock_gen:
        mock_gen.side_effect = [[0.1] * 1024, [0.2] * 1024]

        result1 = generate_embedding("content one")
        result2 = generate_embedding("content two")

        assert mock_gen.call_count == 2
        assert result1 != result2


# --- Cache size limit tests ---


def test_cache_respects_max_size(tmp_path):
    """Cache should evict oldest entries when max size exceeded"""
    import ogham.embeddings as emb
    from ogham.embedding_cache import EmbeddingCache
    from ogham.embeddings import generate_embedding

    old_cache = emb._cache
    emb._cache = EmbeddingCache(cache_dir=str(tmp_path), max_size=2)

    try:
        with patch("ogham.embeddings._generate_uncached") as mock_gen:
            mock_gen.side_effect = lambda t, usage_out=None: [float(hash(t) % 100)] * 1024

            generate_embedding("first")
            generate_embedding("second")
            assert len(emb._cache) == 2

            generate_embedding("third")
            assert len(emb._cache) == 2

            first_key = hashlib.sha256("first".encode()).hexdigest()
            assert emb._cache.get(first_key) is None
    finally:
        emb._cache = old_cache


def test_cache_eviction_counter(tmp_path):
    """Cache stats should track size after eviction"""
    import ogham.embeddings as emb
    from ogham.embedding_cache import EmbeddingCache
    from ogham.embeddings import generate_embedding, get_cache_stats

    old_cache = emb._cache
    emb._cache = EmbeddingCache(cache_dir=str(tmp_path), max_size=1)

    try:
        with patch("ogham.embeddings._generate_uncached") as mock_gen:
            mock_gen.side_effect = lambda t, usage_out=None: [float(hash(t) % 100)] * 1024

            generate_embedding("first")
            generate_embedding("second")

            stats = get_cache_stats()
            assert stats["size"] == 1
            assert stats["max_size"] == 1
    finally:
        emb._cache = old_cache


def test_get_cache_stats_tool():
    """get_cache_stats tool should return cache statistics"""
    from ogham.embeddings import clear_embedding_cache
    from ogham.tools.stats import get_cache_stats

    clear_embedding_cache()
    result = get_cache_stats()

    assert "size" in result
    assert "max_size" in result
    assert "hits" in result
    assert "misses" in result
    assert "evictions" in result
    assert "hit_rate" in result
    assert result["size"] == 0
    assert result["hit_rate"] == 0.0


# --- Expiration database tests ---


def test_get_profile_ttl():
    """get_profile_ttl should return ttl_days for a profile"""
    from unittest.mock import MagicMock

    from ogham.database import get_profile_ttl

    mock_result = MagicMock()
    mock_result.data = [{"profile": "work", "ttl_days": 90}]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        chain = mock_client.return_value.from_.return_value.select.return_value
        chain.eq.return_value.execute.return_value = mock_result
        result = get_profile_ttl("work")

    assert result == 90


def test_get_profile_ttl_not_set():
    """get_profile_ttl should return None if no TTL configured"""
    from unittest.mock import MagicMock

    from ogham.database import get_profile_ttl

    mock_result = MagicMock()
    mock_result.data = []

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        chain = mock_client.return_value.from_.return_value.select.return_value
        chain.eq.return_value.execute.return_value = mock_result
        result = get_profile_ttl("personal")

    assert result is None


def test_set_profile_ttl():
    """set_profile_ttl should upsert TTL for a profile"""
    from unittest.mock import MagicMock

    from ogham.database import set_profile_ttl

    mock_result = MagicMock()
    mock_result.data = [{"profile": "work", "ttl_days": 90}]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        chain = mock_client.return_value.from_.return_value
        chain.upsert.return_value.execute.return_value = mock_result
        result = set_profile_ttl("work", 90)

    assert result["ttl_days"] == 90


def test_cleanup_expired():
    """cleanup_expired should call the RPC and return deleted count"""
    from unittest.mock import MagicMock

    from ogham.database import cleanup_expired

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = MagicMock(data=5)
        result = cleanup_expired("work")

    assert result == 5
    mock_client.return_value.rpc.assert_called_once_with(
        "cleanup_expired_memories", {"target_profile": "work"}
    )


def test_count_expired():
    """count_expired should return number of expired memories"""
    from unittest.mock import MagicMock

    from ogham.database import count_expired

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = MagicMock(data=3)
        result = count_expired("work")

    assert result == 3


def test_store_memory_computes_expires_at(mock_embedding, mock_db):
    """store_memory should set expires_at when profile has TTL"""
    from ogham.tools.memory import store_memory

    with patch("ogham.service.db_get_profile_ttl") as mock_ttl:
        mock_ttl.return_value = 90
        store_memory(content="work note about the project architecture", source="test")

    call_kwargs = mock_db["store"].call_args[1]
    assert "expires_at" in call_kwargs
    assert call_kwargs["expires_at"] is not None


def test_store_memory_no_expires_at_without_ttl(mock_embedding, mock_db):
    """store_memory should not set expires_at when profile has no TTL"""
    from ogham.tools.memory import store_memory

    with patch("ogham.service.db_get_profile_ttl") as mock_ttl:
        mock_ttl.return_value = None
        store_memory(content="personal note about a side project", source="test")

    call_kwargs = mock_db["store"].call_args[1]
    assert call_kwargs.get("expires_at") is None


# --- TTL and cleanup tool tests ---


def test_set_profile_ttl_tool():
    """set_profile_ttl tool should upsert TTL"""
    from ogham.tools.memory import set_profile_ttl

    with patch("ogham.tools.memory.db_set_profile_ttl") as mock_set:
        mock_set.return_value = {"profile": "work", "ttl_days": 90}
        result = set_profile_ttl(profile="work", ttl_days=90)

    assert result["status"] == "configured"
    assert result["ttl_days"] == 90


def test_set_profile_ttl_remove():
    """set_profile_ttl with ttl_days=None should remove TTL"""
    from ogham.tools.memory import set_profile_ttl

    with patch("ogham.tools.memory.db_set_profile_ttl") as mock_set:
        mock_set.return_value = {"profile": "work", "ttl_days": None}
        result = set_profile_ttl(profile="work", ttl_days=None)

    assert result["status"] == "configured"
    assert result["ttl_days"] is None


def test_set_profile_ttl_invalid():
    """set_profile_ttl should reject ttl_days < 1"""
    from ogham.tools.memory import set_profile_ttl

    with pytest.raises(ValueError, match="ttl_days must be at least 1"):
        set_profile_ttl(profile="work", ttl_days=0)


def test_cleanup_expired_tool():
    """cleanup_expired tool should delete expired memories"""
    from ogham.tools.memory import cleanup_expired

    with (
        patch("ogham.tools.memory.db_cleanup_expired") as mock_cleanup,
        patch("ogham.tools.memory.db_count_expired") as mock_count,
    ):
        mock_count.return_value = 5
        mock_cleanup.return_value = 5
        result = cleanup_expired()

    assert result["status"] == "cleaned"
    assert result["deleted"] == 5


def test_cleanup_expired_nothing_to_do():
    """cleanup_expired should report zero when nothing expired"""
    from ogham.tools.memory import cleanup_expired

    with (
        patch("ogham.tools.memory.db_cleanup_expired") as mock_cleanup,
        patch("ogham.tools.memory.db_count_expired") as mock_count,
    ):
        mock_count.return_value = 0
        mock_cleanup.return_value = 0
        result = cleanup_expired()

    assert result["status"] == "nothing_to_clean"
    assert result["deleted"] == 0


# --- Confidence scoring tests ---


def test_update_confidence_db():
    """update_confidence should call the RPC and return new confidence"""
    from unittest.mock import MagicMock

    from ogham.database import update_confidence

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = MagicMock(data=0.72)
        result = update_confidence("some-uuid", 0.85, "default")

    assert result == 0.72
    mock_client.return_value.rpc.assert_called_once_with(
        "update_confidence",
        {"memory_id": "some-uuid", "signal": 0.85, "memory_profile": "default"},
    )


def test_reinforce_memory(mock_embedding, mock_db):
    from ogham.tools.memory import reinforce_memory

    with patch("ogham.tools.memory.db_update_confidence") as mock_conf:
        mock_conf.return_value = 0.72
        result = reinforce_memory(memory_id=FAKE_ID)

    assert result["status"] == "reinforced"
    assert result["confidence"] == 0.72
    mock_conf.assert_called_once_with(FAKE_ID, 0.85, "default")


def test_contradict_memory(mock_embedding, mock_db):
    from ogham.tools.memory import contradict_memory

    with patch("ogham.tools.memory.db_update_confidence") as mock_conf:
        mock_conf.return_value = 0.32
        result = contradict_memory(memory_id=FAKE_ID)

    assert result["status"] == "contradicted"
    assert result["confidence"] == 0.32
    mock_conf.assert_called_once_with(FAKE_ID, 0.15, "default")


def test_reinforce_memory_custom_strength(mock_embedding, mock_db):
    from ogham.tools.memory import reinforce_memory

    with patch("ogham.tools.memory.db_update_confidence") as mock_conf:
        mock_conf.return_value = 0.65
        result = reinforce_memory(memory_id=FAKE_ID, strength=0.7)

    assert result["confidence"] == 0.65
    mock_conf.assert_called_once_with(FAKE_ID, 0.7, "default")


def test_reinforce_memory_invalid_strength(mock_embedding, mock_db):
    from ogham.tools.memory import reinforce_memory

    with pytest.raises(ValueError, match="strength must be between"):
        reinforce_memory(memory_id=FAKE_ID, strength=1.5)


def test_hybrid_search_memories_db():
    """hybrid_search_memories should call the RPC with correct params"""
    from unittest.mock import MagicMock

    from ogham.database import hybrid_search_memories

    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": FAKE_ID,
            "content": "test",
            "similarity": 0.9,
            "keyword_rank": 0.5,
            "relevance": 0.7,
            "tags": [],
            "source": None,
            "profile": "default",
            "access_count": 0,
            "last_accessed_at": None,
            "confidence": 0.5,
        }
    ]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = mock_result
        results = hybrid_search_memories(
            query_text="test query",
            query_embedding=[0.1] * 768,
            profile="default",
            limit=5,
            tags=["tag1"],
            source="test",
        )

    assert len(results) == 1
    assert results[0]["similarity"] == 0.9
    assert results[0]["keyword_rank"] == 0.5
    mock_client.return_value.rpc.assert_called_once_with(
        "hybrid_search_memories",
        {
            "query_text": "test query",
            "query_embedding": str([0.1] * 768),
            "match_count": 5,
            "filter_profile": "default",
            "filter_tags": ["tag1"],
            "filter_source": "test",
        },
    )


# --- Auto-link database tests ---


def test_auto_link_memory_db():
    """auto_link_memory should call the RPC with correct params"""
    from unittest.mock import MagicMock

    from ogham.database import auto_link_memory

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = MagicMock(data=3)
        result = auto_link_memory(
            memory_id="some-uuid",
            embedding=[0.1] * 768,
            profile="default",
            threshold=0.85,
            max_links=5,
        )

    assert result == 3
    mock_client.return_value.rpc.assert_called_once_with(
        "auto_link_memory",
        {
            "new_memory_id": "some-uuid",
            "new_embedding": str([0.1] * 768),
            "link_threshold": 0.85,
            "max_links": 5,
            "filter_profile": "default",
        },
    )


def test_link_unlinked_memories_db():
    """link_unlinked_memories should call the RPC and return processed count"""
    from unittest.mock import MagicMock

    from ogham.database import link_unlinked_memories

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = MagicMock(data=42)
        result = link_unlinked_memories(
            profile="default", threshold=0.85, max_links=5, batch_size=100
        )

    assert result == 42


# --- Auto-link store_memory integration tests ---


def test_store_memory_auto_links(mock_embedding, mock_db):
    """store_memory should call auto_link_memory by default"""
    from ogham.tools.memory import store_memory

    mock_db["auto_link"].return_value = 3
    result = store_memory(content="test memory", source="test")

    assert result["status"] == "stored"
    assert result["links_created"] == 3
    mock_db["auto_link"].assert_called_once()


def test_store_memory_auto_link_disabled(mock_embedding, mock_db):
    """store_memory with auto_link=False should skip auto-linking"""
    from ogham.tools.memory import store_memory

    result = store_memory(content="test memory", auto_link=False)

    assert result["status"] == "stored"
    assert "links_created" not in result
    mock_db["auto_link"].assert_not_called()


# --- link_unlinked tool tests ---


def test_link_unlinked_memories_tool(mock_embedding, mock_db):
    """link_unlinked tool should call the RPC and return count"""
    from ogham.tools.memory import link_unlinked

    with patch("ogham.tools.memory.db_link_unlinked") as mock_link:
        mock_link.return_value = 25
        result = link_unlinked(batch_size=50)

    assert result["status"] == "linked"
    assert result["processed"] == 25


def test_link_unlinked_nothing_to_do(mock_embedding, mock_db):
    """link_unlinked should report zero when all memories already linked"""
    from ogham.tools.memory import link_unlinked

    with patch("ogham.tools.memory.db_link_unlinked") as mock_link:
        mock_link.return_value = 0
        result = link_unlinked()

    assert result["status"] == "nothing_to_link"
    assert result["processed"] == 0


# --- Graph explorer database tests ---


def test_explore_memory_graph_db():
    """explore_memory_graph should call the RPC with correct params"""
    from unittest.mock import MagicMock

    from ogham.database import explore_memory_graph

    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": FAKE_ID,
            "content": "test",
            "metadata": {},
            "source": None,
            "tags": [],
            "relevance": 0.9,
            "depth": 0,
            "relationship": None,
            "edge_strength": None,
            "connected_from": None,
        }
    ]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = mock_result
        results = explore_memory_graph(
            query_text="test query",
            query_embedding=[0.1] * 768,
            profile="default",
            limit=5,
            depth=1,
            min_strength=0.5,
        )

    assert len(results) == 1
    assert results[0]["depth"] == 0


# --- explore_knowledge tool tests ---


def test_explore_knowledge(mock_embedding, mock_db):
    """explore_knowledge should search and traverse graph"""
    from ogham.tools.memory import explore_knowledge

    with patch("ogham.tools.memory.db_explore_graph") as mock_explore:
        mock_explore.return_value = [
            {
                "id": FAKE_ID,
                "content": "direct match",
                "metadata": {},
                "source": None,
                "tags": [],
                "relevance": 0.9,
                "depth": 0,
                "relationship": None,
                "edge_strength": None,
                "connected_from": None,
            },
            {
                "id": "b2c3d4e5-0000-0000-0000-000000000002",
                "content": "related memory",
                "metadata": {},
                "source": None,
                "tags": [],
                "relevance": 0.7,
                "depth": 1,
                "relationship": "similar",
                "edge_strength": 0.85,
                "connected_from": FAKE_ID,
            },
        ]
        results = explore_knowledge(query="test query")

    assert len(results) == 2
    assert results[0]["depth"] == 0
    assert results[1]["depth"] == 1
    assert results[1]["relationship"] == "similar"
    mock_explore.assert_called_once()


# --- store_decision tool tests ---


def test_store_decision(mock_embedding, mock_db):
    """store_decision should store with type:decision tag and structured metadata"""
    from ogham.tools.memory import store_decision

    mock_db["auto_link"].return_value = 2
    result = store_decision(
        decision="Use UUID PKs",
        rationale="Supabase recommends it for distributed systems",
        alternatives=["bigint", "ULID"],
        tags=["project:openbrain"],
    )

    assert result["status"] == "stored"
    call_kwargs = mock_db["store"].call_args[1]
    assert "Decision: Use UUID PKs" in call_kwargs["content"]
    assert "Rationale:" in call_kwargs["content"]
    assert "type:decision" in call_kwargs["tags"]
    assert "project:openbrain" in call_kwargs["tags"]
    assert call_kwargs["metadata"]["type"] == "decision"
    assert call_kwargs["metadata"]["alternatives"] == ["bigint", "ULID"]


def test_store_decision_minimal(mock_embedding, mock_db):
    """store_decision with only required fields"""
    from ogham.tools.memory import store_decision

    result = store_decision(
        decision="Switch to Postgres",
        rationale="Need SQL queries",
    )

    assert result["status"] == "stored"
    call_kwargs = mock_db["store"].call_args[1]
    assert "type:decision" in call_kwargs["tags"]
    assert "Alternatives considered" not in call_kwargs["content"]


def test_store_decision_with_related_memories(mock_embedding, mock_db):
    """store_decision should create supports edges for related_memories"""
    from ogham.tools.memory import store_decision

    related_id = "b2c3d4e5-0000-0000-0000-000000000002"

    with patch("ogham.tools.memory.db_create_relationship") as mock_rel:
        result = store_decision(
            decision="Use RRF for hybrid search",
            rationale="No score normalization needed",
            related_memories=[related_id],
        )

    assert result["status"] == "stored"
    mock_rel.assert_called_once_with(
        source_id=FAKE_ID,
        target_id=related_id,
        relationship="supports",
        strength=1.0,
        created_by="user",
        metadata={},
    )


# --- get_related_memories database tests ---


def test_get_related_memories_db():
    """get_related_memories should call the RPC with correct params"""
    from unittest.mock import MagicMock

    from ogham.database import get_related_memories

    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": "b2c3d4e5-0000-0000-0000-000000000002",
            "content": "related",
            "metadata": {},
            "source": None,
            "tags": [],
            "confidence": 0.5,
            "depth": 1,
            "relationship": "similar",
            "edge_strength": 0.9,
            "connected_from": FAKE_ID,
        }
    ]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_client.return_value.rpc.return_value.execute.return_value = mock_result
        results = get_related_memories(
            memory_id=FAKE_ID,
            depth=1,
            min_strength=0.5,
            relationship_types=["similar"],
            limit=20,
        )

    assert len(results) == 1
    assert results[0]["relationship"] == "similar"


# --- find_related tool tests ---


def test_find_related(mock_embedding, mock_db):
    """find_related should traverse graph from a memory"""
    from ogham.tools.memory import find_related

    with patch("ogham.tools.memory.db_get_related") as mock_related:
        mock_related.return_value = [
            {
                "id": "b2c3d4e5-0000-0000-0000-000000000002",
                "content": "related memory",
                "metadata": {},
                "source": None,
                "tags": [],
                "confidence": 0.5,
                "depth": 1,
                "relationship": "similar",
                "edge_strength": 0.9,
                "connected_from": FAKE_ID,
            }
        ]
        results = find_related(memory_id=FAKE_ID)

    assert len(results) == 1
    assert results[0]["relationship"] == "similar"
    mock_related.assert_called_once_with(
        memory_id=FAKE_ID,
        depth=1,
        min_strength=0.5,
        relationship_types=None,
        limit=20,
    )


def test_find_related_with_type_filter(mock_embedding, mock_db):
    """find_related should pass relationship_types filter"""
    from ogham.tools.memory import find_related

    with patch("ogham.tools.memory.db_get_related") as mock_related:
        mock_related.return_value = []
        find_related(memory_id=FAKE_ID, relationship_types=["supports"])

    call_kwargs = mock_related.call_args[1]
    assert call_kwargs["relationship_types"] == ["supports"]
