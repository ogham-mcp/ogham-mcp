"""Unit tests for ogham.database.find_by_metadata_kv facade."""

from unittest.mock import MagicMock


def _fake_backend(memories):
    backend = MagicMock()
    backend.get_all_memories_full.return_value = memories
    return backend


def test_find_by_metadata_kv_returns_first_match(monkeypatch):
    import ogham.database as db

    memories = [
        {"id": "m1", "metadata": {"tracker_external_id": "u1"}},
        {"id": "m2", "metadata": {"tracker_external_id": "u2"}},
    ]
    monkeypatch.setattr(db, "get_backend", lambda: _fake_backend(memories))

    result = db.find_by_metadata_kv("tracker_external_id", "u2", "work")
    assert result is not None
    assert result["id"] == "m2"


def test_find_by_metadata_kv_returns_none_when_absent(monkeypatch):
    import ogham.database as db

    memories = [{"id": "m1", "metadata": {"tracker_external_id": "u1"}}]
    monkeypatch.setattr(db, "get_backend", lambda: _fake_backend(memories))

    result = db.find_by_metadata_kv("tracker_external_id", "notfound", "work")
    assert result is None


def test_find_by_metadata_kv_survives_none_metadata_row(monkeypatch):
    """Regression guard: a memory row with metadata=None must not crash the scan."""
    import ogham.database as db

    memories = [
        {"id": "m1", "metadata": None},
        {"id": "m2", "metadata": {"tracker_external_id": "u2"}},
    ]
    monkeypatch.setattr(db, "get_backend", lambda: _fake_backend(memories))

    result = db.find_by_metadata_kv("tracker_external_id", "u2", "work")
    assert result is not None
    assert result["id"] == "m2"
