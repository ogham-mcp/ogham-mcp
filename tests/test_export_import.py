import json
from unittest.mock import patch

import pytest

FAKE_MEMORIES = [
    {
        "id": "a1b2c3d4-0000-0000-0000-000000000001",
        "content": "first memory",
        "metadata": {},
        "source": "test",
        "profile": "default",
        "tags": ["tag1"],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    },
    {
        "id": "a1b2c3d4-0000-0000-0000-000000000002",
        "content": "second memory",
        "metadata": {"key": "value"},
        "source": "test",
        "profile": "default",
        "tags": ["tag2"],
        "created_at": "2026-01-02T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
    },
]


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")


@pytest.fixture(autouse=True)
def reset_profile():
    import ogham.tools.memory as mem

    mem._active_profile = "default"
    yield
    mem._active_profile = "default"


def test_export_json():
    """export should produce valid JSON with memory data"""
    from ogham.export_import import export_memories

    with patch("ogham.export_import.get_all_memories_full") as mock_get:
        mock_get.return_value = FAKE_MEMORIES
        result = export_memories("default", "json")

    data = json.loads(result)
    assert data["profile"] == "default"
    assert len(data["memories"]) == 2
    assert data["memories"][0]["content"] == "first memory"


def test_export_markdown():
    """export should produce readable markdown"""
    from ogham.export_import import export_memories

    with patch("ogham.export_import.get_all_memories_full") as mock_get:
        mock_get.return_value = FAKE_MEMORIES
        result = export_memories("default", "markdown")

    assert "# Ogham Memory Export" in result
    assert "first memory" in result
    assert "second memory" in result


def test_export_empty():
    """export with no memories should still produce valid output"""
    from ogham.export_import import export_memories

    with patch("ogham.export_import.get_all_memories_full") as mock_get:
        mock_get.return_value = []
        result = export_memories("default", "json")

    data = json.loads(result)
    assert data["count"] == 0
    assert data["memories"] == []


def test_import_json():
    """import should parse JSON and store each memory"""
    from ogham.export_import import import_memories

    export_data = json.dumps({
        "profile": "default",
        "memories": FAKE_MEMORIES,
    })

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
    ):
        mock_embed.return_value = [[0.1] * 768] * 2
        mock_store.return_value = [{"id": "new-id"}] * 2
        mock_ttl.return_value = None
        result = import_memories(export_data, "default")

    assert result["imported"] == 2
    assert result["skipped"] == 0


def test_import_with_dedup():
    """import should skip memories with matching content when dedup enabled"""
    from ogham.export_import import import_memories

    export_data = json.dumps({
        "profile": "default",
        "memories": FAKE_MEMORIES,
    })

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
        patch("ogham.export_import.batch_check_duplicates") as mock_dedup,
    ):
        mock_embed.return_value = [[0.1] * 768] * 2
        mock_store.return_value = [{"id": "new-id"}]
        mock_ttl.return_value = None
        mock_dedup.return_value = [True, False]
        result = import_memories(export_data, "default", dedup_threshold=0.9)

    assert result["imported"] == 1
    assert result["skipped"] == 1


def test_import_with_ttl():
    """import should compute expires_at when profile has TTL"""
    from ogham.export_import import import_memories

    export_data = json.dumps({
        "profile": "work",
        "memories": [FAKE_MEMORIES[0]],
    })

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
    ):
        mock_embed.return_value = [[0.1] * 768]
        mock_store.return_value = [{"id": "new-id"}]
        mock_ttl.return_value = 90
        result = import_memories(export_data, "work")

    assert result["imported"] == 1
    inserted_rows = mock_store.call_args[0][0]
    assert inserted_rows[0].get("expires_at") is not None


def test_store_memories_batch():
    """store_memories_batch should insert multiple rows in a single call"""
    from ogham.database import store_memories_batch

    rows = [
        {
            "content": "memory one",
            "embedding": str([0.1] * 768),
            "profile": "default",
            "metadata": {},
            "source": "test",
            "tags": ["t1"],
        },
        {
            "content": "memory two",
            "embedding": str([0.2] * 768),
            "profile": "default",
            "metadata": {},
            "source": "test",
            "tags": ["t2"],
        },
    ]

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_table = mock_client.return_value.table.return_value
        mock_insert = mock_table.insert.return_value
        mock_insert.execute.return_value.data = [
            {"id": "id-1", "content": "memory one"},
            {"id": "id-2", "content": "memory two"},
        ]

        result = store_memories_batch(rows)

    assert len(result) == 2
    mock_table.insert.assert_called_once_with(rows)


def test_store_memories_batch_empty():
    """store_memories_batch with empty list should return empty without calling DB"""
    from ogham.database import store_memories_batch

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        result = store_memories_batch([])

    assert result == []
    mock_client.return_value.table.return_value.insert.assert_not_called()


def test_build_row():
    """_build_row should create a properly formatted row dict for insertion"""
    from ogham.export_import import _build_row

    mem = {
        "content": "test content",
        "metadata": {"key": "val"},
        "source": "cli",
        "tags": ["t1"],
    }
    embedding = [0.1] * 768

    row = _build_row(mem, embedding, profile="work", expires_at="2026-06-01T00:00:00Z")

    assert row["content"] == "test content"
    assert row["embedding"] == str([0.1] * 768)
    assert row["profile"] == "work"
    assert row["metadata"] == {"key": "val"}
    assert row["source"] == "cli"
    assert row["tags"] == ["t1"]
    assert row["expires_at"] == "2026-06-01T00:00:00Z"


def test_build_row_no_expires():
    """_build_row should omit expires_at when None"""
    from ogham.export_import import _build_row

    mem = {"content": "test", "metadata": None, "source": None, "tags": None}
    row = _build_row(mem, [0.1] * 768, profile="default", expires_at=None)

    assert "expires_at" not in row
    assert row["metadata"] == {}
    assert row["tags"] == []


def test_import_uses_batch_insert():
    """import_memories should use store_memories_batch instead of individual inserts"""
    from ogham.export_import import import_memories

    export_data = json.dumps({
        "memories": [
            {"content": "mem 1", "source": "test", "tags": ["t1"], "metadata": {}},
            {"content": "mem 2", "source": "test", "tags": ["t2"], "metadata": {}},
            {"content": "mem 3", "source": "test", "tags": ["t3"], "metadata": {}},
        ],
    })

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_batch_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
    ):
        mock_embed.return_value = [[0.1] * 768] * 3
        mock_batch_store.return_value = [{"id": f"id-{i}"} for i in range(3)]
        mock_ttl.return_value = None

        result = import_memories(export_data, "default")

    assert result["imported"] == 3
    assert result["skipped"] == 0
    mock_batch_store.assert_called_once()
    assert len(mock_batch_store.call_args[0][0]) == 3


def test_import_batch_insert_with_dedup():
    """import should batch-insert only non-duplicate memories"""
    from ogham.export_import import import_memories

    export_data = json.dumps({
        "memories": [
            {"content": "mem 1", "source": "test", "tags": [], "metadata": {}},
            {"content": "mem 2 (dup)", "source": "test", "tags": [], "metadata": {}},
            {"content": "mem 3", "source": "test", "tags": [], "metadata": {}},
        ],
    })

    emb_1 = [0.1] * 768
    emb_2 = [0.2] * 768  # this one will be the "duplicate"
    emb_3 = [0.3] * 768

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_batch_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
        patch("ogham.export_import.batch_check_duplicates") as mock_dedup,
    ):
        mock_embed.return_value = [emb_1, emb_2, emb_3]
        mock_batch_store.return_value = [{"id": "id-1"}, {"id": "id-3"}]
        mock_ttl.return_value = None
        mock_dedup.return_value = [False, True, False]

        result = import_memories(export_data, "default", dedup_threshold=0.8)

    assert result["imported"] == 2
    assert result["skipped"] == 1
    inserted_rows = mock_batch_store.call_args[0][0]
    assert len(inserted_rows) == 2
    contents = {row["content"] for row in inserted_rows}
    assert contents == {"mem 1", "mem 3"}


def test_import_batch_dedup():
    """import should correctly handle batch dedup results"""
    from ogham.export_import import import_memories

    memories_data = [
        {"content": f"memory {i}", "source": "test", "tags": [], "metadata": {}}
        for i in range(20)
    ]
    export_data = json.dumps({"memories": memories_data})

    # Every 3rd memory (index 2,5,8,11,14,17) is a duplicate
    dedup_results = [((i + 1) % 3 == 0) for i in range(20)]

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
        patch("ogham.export_import.batch_check_duplicates") as mock_dedup,
    ):
        mock_embed.return_value = [[0.1] * 768] * 20
        mock_store.return_value = [{"id": f"id-{i}"} for i in range(14)]
        mock_ttl.return_value = None
        # 20 memories fits in one batch of 50, so one call
        mock_dedup.return_value = dedup_results

        result = import_memories(export_data, "default", dedup_threshold=0.8)

    # 20 memories, every 3rd is dup (indices 2,5,8,11,14,17) = 6 skipped, 14 imported
    assert result["skipped"] == 6
    assert result["imported"] == 14
    assert result["total"] == 20
    mock_dedup.assert_called_once()


def test_import_uses_batch_dedup():
    """import should use batch_check_duplicates instead of individual search_memories calls"""
    from ogham.export_import import import_memories

    memories_data = [
        {"content": f"memory {i}", "source": "test", "tags": [], "metadata": {}}
        for i in range(5)
    ]
    export_data = json.dumps({"memories": memories_data})

    embeddings = [[0.1 * (i + 1)] * 768 for i in range(5)]

    with (
        patch("ogham.export_import.generate_embeddings_batch") as mock_embed,
        patch("ogham.export_import.store_memories_batch") as mock_store,
        patch("ogham.export_import.get_profile_ttl") as mock_ttl,
        patch("ogham.export_import.batch_check_duplicates") as mock_dedup,
    ):
        mock_embed.return_value = embeddings
        mock_store.return_value = [{"id": f"id-{i}"} for i in range(3)]
        mock_ttl.return_value = None
        # Mark indices 1 and 3 as duplicates
        mock_dedup.return_value = [False, True, False, True, False]

        result = import_memories(export_data, "default", dedup_threshold=0.8)

    assert result["imported"] == 3
    assert result["skipped"] == 2
    # 5 memories fits in one batch of 50, so one call
    mock_dedup.assert_called_once()


def test_batch_check_duplicates():
    """batch_check_duplicates should call the RPC and return a list of booleans"""
    from ogham.database import batch_check_duplicates

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        mock_rpc = mock_client.return_value.rpc.return_value
        mock_rpc.execute.return_value.data = [True, False, True]

        result = batch_check_duplicates(
            query_embeddings=[[0.1] * 768, [0.2] * 768, [0.3] * 768],
            profile="work",
            threshold=0.8,
        )

    assert result == [True, False, True]
    mock_client.return_value.rpc.assert_called_once_with(
        "batch_check_duplicates",
        {
            "query_embeddings": [str([0.1] * 768), str([0.2] * 768), str([0.3] * 768)],
            "match_threshold": 0.8,
            "filter_profile": "work",
        },
    )


def test_batch_check_duplicates_empty():
    """batch_check_duplicates with empty list should return empty without calling DB"""
    from ogham.database import batch_check_duplicates

    with patch("ogham.backends.supabase.SupabaseBackend._get_client") as mock_client:
        result = batch_check_duplicates(
            query_embeddings=[],
            profile="work",
            threshold=0.8,
        )

    assert result == []
    mock_client.return_value.rpc.assert_not_called()
