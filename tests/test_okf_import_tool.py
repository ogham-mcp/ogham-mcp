"""Tests for import_memories_tool with OKF bundle inputs."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_backend(monkeypatch):
    """Capture inserted memories instead of writing to a real DB."""
    inserted: list[dict] = []

    def fake_import(data, profile, dedup_threshold=0.8, **kwargs):
        # data is either a JSON string or an OKF bundle path string
        if isinstance(data, str) and Path(data).is_dir():
            from ogham.okf import import_okf_bundle

            memories, stats = import_okf_bundle(Path(data))
            inserted.extend(memories)
            return {
                "status": "complete",
                "imported": len(memories),
                "skipped": 0,
                "total": stats["total"],
                "missing_id_count": stats["missing_id_count"],
            }
        # JSON path falls through to real logic; skip in this test
        return {"status": "complete", "imported": 0, "skipped": 0, "total": 0}

    import ogham.tools.memory as _memory_mod

    monkeypatch.setattr(_memory_mod, "_import_memories", fake_import)
    return inserted


def test_import_memories_tool_detects_okf_directory(fake_backend, tmp_path):
    from ogham.okf import export_okf_bundle

    memories = [
        {
            "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
            "content": "Test memory",
            "tags": ["type:decision"],
            "source": "test",
            "created_at": "2026-06-17T00:00:00Z",
            "metadata": {},
        }
    ]
    bundle = tmp_path / "test-bundle"
    export_okf_bundle(memories, bundle, {"producer": "t", "exported_at": "t", "profile": "p"})

    from ogham.tools.memory import import_memories_tool

    result = import_memories_tool(data=str(bundle))
    assert result["status"] == "complete"
    assert result["imported"] == 1
    assert len(fake_backend) == 1
    assert fake_backend[0]["id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"


def test_import_memories_tool_returns_missing_id_count(fake_backend, tmp_path):
    from ogham.okf.serialization import write_concept

    (tmp_path / "memories").mkdir()
    write_concept(
        tmp_path / "memories" / "no-id.md",
        {"type": "Memory", "tags": []},
        "missing id",
    )
    write_concept(
        tmp_path / "memories" / "with-id.md",
        {"type": "Memory", "id": "11111111-2222-3333-4444-555555555555", "tags": []},
        "has id",
    )

    from ogham.tools.memory import import_memories_tool

    result = import_memories_tool(data=str(tmp_path))
    assert result["missing_id_count"] == 1
    assert result["total"] == 2


def test_import_memories_tool_still_handles_json_string(fake_backend):
    # Existing v0.9.1 behaviour must keep working
    from ogham.tools.memory import import_memories_tool

    data = json.dumps({"memories": []})
    result = import_memories_tool(data=data)
    assert result["status"] == "complete"


def test_import_okf_upserts_existing_memory_by_uuid(tmp_path, monkeypatch):
    """Round-trip: import a bundle that contains a memory whose UUID already
    exists in the profile. The existing record should be UPDATED, not duplicated.

    This test calls export_import.import_memories directly (bypassing the tool
    layer) so we exercise the real OKF routing logic without a DB connection.
    The backend upsert is stubbed at the export_import level.
    """
    existing_id = "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
    upsert_calls: list[str] = []

    def fake_upsert(memory_dict):
        upsert_calls.append(memory_dict["id"])

    monkeypatch.setattr("ogham.export_import._upsert_memory", fake_upsert, raising=False)

    # Also stub out embedding generation (not needed for upsert path test)
    monkeypatch.setattr(
        "ogham.export_import.generate_embeddings_batch",
        lambda texts, **kw: [[0.0]] * len(texts),
    )
    # Stub get_profile_ttl to avoid DB call
    monkeypatch.setattr("ogham.export_import.get_profile_ttl", lambda profile: None)

    from ogham.okf import export_okf_bundle

    memories = [
        {
            "id": existing_id,
            "content": "Round-trip content",
            "tags": ["type:decision"],
            "source": "test",
            "created_at": "2026-06-17T00:00:00Z",
            "metadata": {},
        }
    ]
    bundle = tmp_path / "rt-bundle"
    export_okf_bundle(memories, bundle, {"producer": "t", "exported_at": "t", "profile": "p"})

    from ogham.export_import import import_memories

    import_memories(data=str(bundle), profile="test")
    # _upsert_memory called once for the round-tripped UUID
    assert existing_id in upsert_calls


# ---------------------------------------------------------------------------
# Fix 1: Supabase upsert_memory includes importance + surprise defaults
# ---------------------------------------------------------------------------


def test_supabase_upsert_includes_importance_surprise_defaults():
    """Upsert payload must always carry importance + surprise so a fresh INSERT
    does not rely on PostgREST applying Postgres column defaults."""
    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend()

    mock_client = MagicMock()
    mock_upsert_chain = mock_client.from_.return_value.upsert.return_value
    mock_upsert_chain.execute.return_value.data = [{"id": "abc", "content": "test content"}]
    # A fresh INSERT: the existence lookup finds no row (TBU-253 -- on an
    # existing row an absent importance is left out of the payload instead).
    lookup = mock_client.from_.return_value.select.return_value.eq.return_value.limit.return_value
    lookup.execute.return_value.data = []

    with patch.object(backend, "_get_client", return_value=mock_client):
        # Memory dict deliberately omits importance + surprise (simulates OKF
        # round-trip where those fields may not be in frontmatter)
        memory = {
            "id": "abc-123",
            "content": "test content",
            "profile": "default",
            "embedding": [0.1, 0.2],
        }
        backend.upsert_memory(memory)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, call_kwargs = mock_client.from_.return_value.upsert.call_args
        row_arg = mock_client.from_.return_value.upsert.call_args[0][0]

    assert "importance" in row_arg, "upsert row must include importance"
    assert "surprise" in row_arg, "upsert row must include surprise"
    assert row_arg["importance"] == 0.5
    assert row_arg["surprise"] == 0.5


def test_supabase_upsert_preserves_explicit_importance_surprise():
    """Explicit importance/surprise values in the memory dict must not be overridden."""
    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend()
    mock_client = MagicMock()
    mock_client.from_.return_value.upsert.return_value.execute.return_value.data = [
        {"id": "xyz", "content": "hi"}
    ]

    with patch.object(backend, "_get_client", return_value=mock_client):
        memory = {
            "id": "xyz",
            "content": "hi",
            "profile": "default",
            "embedding": [0.3],
            "importance": 0.9,
            "surprise": 0.7,
        }
        backend.upsert_memory(memory)

    row_arg = mock_client.from_.return_value.upsert.call_args[0][0]
    assert row_arg["importance"] == 0.9
    assert row_arg["surprise"] == 0.7


# ---------------------------------------------------------------------------
# Fix 3: skipped_count propagates through import_memories return dict
# ---------------------------------------------------------------------------


def test_import_memories_okf_surfaces_skipped_count(tmp_path, monkeypatch):
    """import_memories with an OKF bundle must expose skipped_count in the
    return dict when malformed concept files are silently dropped."""
    from ogham.okf import export_okf_bundle

    bundle = tmp_path / "bundle"
    memories = [
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "content": "Good memory",
            "tags": [],
            "source": None,
            "created_at": "2026-06-17T00:00:00Z",
            "metadata": {},
        }
    ]
    export_okf_bundle(memories, bundle, {"producer": "t", "exported_at": "t", "profile": "p"})
    # Inject a malformed concept file (no frontmatter) so skipped_count > 0
    (bundle / "memories" / "bad.md").write_text("no frontmatter\n", encoding="utf-8")

    monkeypatch.setattr("ogham.export_import._upsert_memory", lambda m: None)
    monkeypatch.setattr(
        "ogham.export_import.generate_embeddings_batch",
        lambda texts, **kw: [[0.0]] * len(texts),
    )
    monkeypatch.setattr("ogham.export_import.get_profile_ttl", lambda profile: None)

    from ogham.export_import import import_memories

    result = import_memories(data=str(bundle), profile="test")
    assert "skipped_count" in result, "skipped_count must be present in return dict"
    assert result["skipped_count"] == 1


# ---------------------------------------------------------------------------
# Fix 5: preflight rejects non-OKF directories
# ---------------------------------------------------------------------------


def test_import_rejects_non_okf_directory(tmp_path):
    """A directory without an OKF index.md should raise ValueError rather than
    silently importing every .md file inside it as memories."""
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "notes.md").write_text("# random notes\n", encoding="utf-8")
    # No index.md -- not a bundle.

    from ogham.tools.memory import import_memories_tool

    with pytest.raises(ValueError, match="OKF bundle"):
        import_memories_tool(data=str(tmp_path))


def test_import_rejects_directory_with_an_index_but_no_concepts(tmp_path):
    """OKF v0.3 §11.2/§12 forbid refusing a bundle over a missing okf_version,
    so the version is no longer the gate. What still refuses is a directory
    with no concept at all -- the "pointed at the wrong path" case the old
    index check existed to catch."""
    (tmp_path / "memories").mkdir()
    # An index.md without okf_version, and a README that is not a concept.
    (tmp_path / "index.md").write_text(
        "---\nproducer: test\n---\n# Not a real bundle\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Notes\n", encoding="utf-8")
    (tmp_path / "memories" / "note.md").write_text(
        "---\ntitle: frontmatter but no type\n---\nbody\n", encoding="utf-8"
    )

    from ogham.tools.memory import import_memories_tool

    with pytest.raises(ValueError, match="no OKF concept files"):
        import_memories_tool(data=str(tmp_path))
