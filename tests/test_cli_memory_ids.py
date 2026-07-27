"""Tests for memory-ID handling in the CLI (ogham-mcp#70).

`ogham delete --help` promised "full UUID or prefix", but the backends passed
the argument straight at a `uuid` column, so a prefix raised a raw
InvalidTextRepresentation. Compounding it, `list` and `search` render only the
first 8 characters, so the reporter had to query Postgres by hand to obtain an
ID the CLI would accept.

Postgres resolves a prefix natively (`id::text LIKE ...`). PostgreSQL will not
implicitly cast uuid to text for LIKE, so the PostgREST/Supabase path cannot
express it without a SQL function (a migration); there it must say so plainly
rather than crash.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()

FULL_ID = "bf85ee48-3c21-4f0a-9d17-2b8e5c11aa04"
OTHER_ID = "bf85a1c9-77d0-4e33-8a01-1f0c9b6d2e55"


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")


def _row(**over):
    row = {
        "id": FULL_ID,
        "created_at": "2026-07-26T15:36:48",
        "content": "config.env values invisible to ingestion adapters",
        "tags": ["type:gotcha"],
        "source": "telegram",
        "relevance": 0.42,
    }
    row.update(over)
    return row


# --- delete: resolve a prefix to a full UUID ------------------------------


def test_delete_resolves_unique_prefix():
    """A prefix matching exactly one memory is resolved, then deleted."""
    from ogham.cli import app

    with (
        patch("ogham.database.find_memory_ids_by_prefix", return_value=[FULL_ID]) as mock_find,
        patch("ogham.database.delete_memory", return_value=True) as mock_delete,
    ):
        result = runner.invoke(app, ["delete", "bf85ee48", "--yes"])

    assert result.exit_code == 0
    mock_find.assert_called_once()
    # The backend must receive the resolved UUID, never the prefix.
    assert mock_delete.call_args[0][0] == FULL_ID


def test_delete_refuses_ambiguous_prefix_and_lists_candidates():
    """An ambiguous prefix must never guess -- it lists and exits nonzero."""
    from ogham.cli import app

    with (
        patch("ogham.database.find_memory_ids_by_prefix", return_value=[FULL_ID, OTHER_ID]),
        patch("ogham.database.delete_memory") as mock_delete,
    ):
        result = runner.invoke(app, ["delete", "bf85", "--yes"])

    assert result.exit_code == 1
    mock_delete.assert_not_called()
    flat = result.output.replace("\n", "")
    assert FULL_ID in flat and OTHER_ID in flat


def test_delete_prefix_with_no_match_reports_not_found():
    from ogham.cli import app

    with (
        patch("ogham.database.find_memory_ids_by_prefix", return_value=[]),
        patch("ogham.database.delete_memory") as mock_delete,
    ):
        result = runner.invoke(app, ["delete", "deadbeef", "--yes"])

    assert result.exit_code == 1
    mock_delete.assert_not_called()
    assert "not found" in result.output.lower()


def test_delete_full_uuid_skips_prefix_lookup():
    """A full UUID goes straight to the backend -- no extra round trip."""
    from ogham.cli import app

    with (
        patch("ogham.database.find_memory_ids_by_prefix") as mock_find,
        patch("ogham.database.delete_memory", return_value=True) as mock_delete,
    ):
        result = runner.invoke(app, ["delete", FULL_ID, "--yes"])

    assert result.exit_code == 0
    mock_find.assert_not_called()
    assert mock_delete.call_args[0][0] == FULL_ID


def test_delete_reports_backend_without_prefix_support():
    """Backends that cannot do prefix lookup say so instead of crashing."""
    from ogham.cli import app

    with (
        patch("ogham.database.find_memory_ids_by_prefix", side_effect=NotImplementedError),
        patch("ogham.database.delete_memory") as mock_delete,
    ):
        result = runner.invoke(app, ["delete", "bf85ee48", "--yes"])

    assert result.exit_code == 1
    mock_delete.assert_not_called()
    assert "--full-id" in result.output


def test_delete_rejects_garbage_that_is_not_hex():
    """A non-hex argument is neither a UUID nor a valid prefix."""
    from ogham.cli import app

    with patch("ogham.database.delete_memory") as mock_delete:
        result = runner.invoke(app, ["delete", "not-an-id!", "--yes"])

    assert result.exit_code == 1
    mock_delete.assert_not_called()


# --- postgres backend: the actual prefix query ----------------------------


def test_postgres_prefix_query_casts_uuid_to_text():
    """The SQL must cast id to text; a bare uuid LIKE would error in PG."""
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend.__new__(PostgresBackend)
    captured = {}

    def fake_execute(query, params=None, *, fetch="all"):
        captured["sql"] = query
        captured["params"] = params
        return [{"id": FULL_ID}]

    # Patch the instance rather than assigning to the overloaded method.
    with patch.object(backend, "_execute", fake_execute):
        ids = backend.find_memory_ids_by_prefix("bf85ee48", "work")

    assert ids == [FULL_ID]
    assert "id::text" in captured["sql"]
    assert captured["params"]["prefix"] == "bf85ee48"


def test_supabase_prefix_lookup_raises_not_implemented():
    """PostgREST cannot express uuid LIKE; the backend must be explicit."""
    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend.__new__(SupabaseBackend)
    with pytest.raises(NotImplementedError):
        backend.find_memory_ids_by_prefix("bf85ee48", "work")


# --- list / search: make the full ID obtainable ---------------------------


def test_list_truncates_id_by_default():
    from ogham.cli import app

    with patch("ogham.database.list_recent_memories", return_value=[_row()]):
        result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert FULL_ID not in result.output


def test_list_full_id_flag_shows_whole_uuid():
    from ogham.cli import app

    with patch("ogham.database.list_recent_memories", return_value=[_row()]):
        result = runner.invoke(app, ["list", "--full-id"])

    assert result.exit_code == 0
    assert FULL_ID in result.output.replace("\n", "")


def test_search_full_id_flag_shows_whole_uuid():
    from ogham.cli import app

    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=[_row()]),
    ):
        result = runner.invoke(app, ["search", "adapters", "--full-id"])

    assert result.exit_code == 0
    assert FULL_ID in result.output.replace("\n", "")
