"""Tests for the v0.14 entities backfill loop.

The pure-Python orchestration in entity_backfill.py is tested with a
fake backend so we don't need a live database. The SQL helper
(link_memory_entities RPC + the schema retrofit) is exercised by the
postgres_integration tier in test_migration_036.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch


class _FakePostgresBackend:
    """Stand-in for PostgresBackend.

    The class name contains 'postgres' so _select_memory_rows takes the
    SQL branch. Captures link_memory_entities calls for assertion.
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.linked: list[tuple[str, str, list[str]]] = []

    def _execute(self, sql: str, params: dict[str, Any], fetch: str) -> list[dict[str, Any]]:
        profile = params.get("profile")
        if profile is None:
            return list(self._rows)
        return [r for r in self._rows if r.get("profile") == profile]

    def link_memory_entities(
        self,
        memory_id: str,
        profile: str,
        entity_tags: list[str],
    ) -> int:
        self.linked.append((memory_id, profile, list(entity_tags)))
        return len(entity_tags)


def test_backfill_walks_rows_and_links_entities():
    rows = [
        {"id": "m-1", "profile": "work", "content": "Cemre opened PR #50"},
        {"id": "m-2", "profile": "work", "content": "Bug in src/ogham/llm.py"},
        {"id": "m-3", "profile": "work", "content": "the and a or"},  # likely no entities
    ]
    backend = _FakePostgresBackend(rows)

    from ogham import entity_backfill

    with patch.object(entity_backfill, "get_backend", return_value=backend):
        out = entity_backfill.backfill_entities(profile="work")

    assert out["status"] == "complete"
    assert out["total"] == 3
    assert out["processed"] == 3
    # At least one row should produce entities (the file path).
    assert out["memories_with_entities"] >= 1
    assert len(backend.linked) >= 1


def test_backfill_skips_rows_with_no_extracted_entities():
    rows = [{"id": "m-1", "profile": "work", "content": "the and a or"}]
    backend = _FakePostgresBackend(rows)

    from ogham import entity_backfill

    with patch.object(entity_backfill, "get_backend", return_value=backend):
        out = entity_backfill.backfill_entities(profile="work")

    assert out["processed"] == 1
    assert out["memories_with_entities"] == 0
    assert backend.linked == []


def test_backfill_continues_after_link_failure(caplog):
    """A bad row logs a warning but doesn't abort the whole run."""
    rows = [
        {"id": "m-1", "profile": "work", "content": "Bug in src/ogham/llm.py"},
        {"id": "m-2", "profile": "work", "content": "Bug in src/ogham/recompute.py"},
    ]
    backend = _FakePostgresBackend(rows)

    original = backend.link_memory_entities
    calls = {"n": 0}

    def flaky(memory_id: str, profile: str, entity_tags: list[str]) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("RPC missing -- migration 036 not applied")
        return original(memory_id, profile, entity_tags)

    backend.link_memory_entities = flaky  # type: ignore[method-assign]

    from ogham import entity_backfill

    with (
        patch.object(entity_backfill, "get_backend", return_value=backend),
        caplog.at_level("WARNING"),
    ):
        out = entity_backfill.backfill_entities(profile="work")

    assert out["processed"] == 2
    assert "RPC missing" in caplog.text


def test_backfill_progress_callback_fires_per_row():
    rows = [
        {"id": f"m-{i}", "profile": "work", "content": f"src/ogham/file{i}.py touched"}
        for i in range(5)
    ]
    backend = _FakePostgresBackend(rows)
    calls: list[tuple[int, int, int]] = []

    from ogham import entity_backfill

    with patch.object(entity_backfill, "get_backend", return_value=backend):
        entity_backfill.backfill_entities(
            profile="work", on_progress=lambda p, e, t: calls.append((p, e, t))
        )

    assert len(calls) == 5
    assert calls[-1][0] == 5
    assert calls[-1][2] == 5


def test_backfill_no_profile_walks_every_row():
    rows = [
        {"id": "m-1", "profile": "work", "content": "src/ogham/llm.py"},
        {"id": "m-2", "profile": "personal", "content": "src/ogham/recompute.py"},
    ]
    backend = _FakePostgresBackend(rows)

    from ogham import entity_backfill

    with patch.object(entity_backfill, "get_backend", return_value=backend):
        out = entity_backfill.backfill_entities(profile=None)

    assert out["total"] == 2
    assert out["profile"] is None


def test_backfill_unknown_backend_raises():
    """Backends other than postgres/supabase aren't implemented."""

    class _GatewayBackend:
        pass

    from ogham import entity_backfill

    with patch.object(entity_backfill, "get_backend", return_value=_GatewayBackend()):
        try:
            entity_backfill.backfill_entities(profile="work")
        except NotImplementedError as exc:
            assert "_GatewayBackend" in str(exc)
        else:
            raise AssertionError("expected NotImplementedError")
