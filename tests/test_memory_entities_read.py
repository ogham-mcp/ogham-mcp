"""``get_memory_entities`` -- the read behind the D7 MENTIONS bridge.

``memory_entities`` had a write path (``link_memory_entities``) and no read.
OKF export needs the join table whole: memory id -> the entities that memory
mentions, which becomes ``MENTIONS`` on the memory concept.

Two things here are behaviour, not plumbing, and each has a test:

* the key is a **string** memory id. psycopg hands back ``uuid.UUID`` for a
  uuid column while PostgREST hands back a string; a caller looking up
  ``str(memory["id"])`` would silently miss every row on the postgres backend.
* a pre-036 install (no entities layer) must yield ``{}``, not an exception.
  The bridge is additive -- a bundle without MENTIONS is still a valid bundle,
  so failing the whole export over a missing table would be the wrong trade.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ogham import database as db
from ogham.backends.gateway import GatewayBackend
from ogham.backends.postgres import PostgresBackend
from ogham.backends.protocol import DatabaseBackend
from ogham.backends.supabase import SupabaseBackend

MEM_A = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
MEM_B = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _reset_backend():
    db._backend = None
    yield
    db._backend = None


# ── protocol ──────────────────────────────────────────────────────────────


def test_protocol_declares_get_memory_entities():
    assert hasattr(DatabaseBackend, "get_memory_entities")


@pytest.mark.parametrize("backend", [PostgresBackend, SupabaseBackend, GatewayBackend])
def test_every_backend_implements_it(backend):
    """DatabaseBackend is runtime_checkable, so widening it un-conforms any
    implementer that does not follow."""
    assert callable(getattr(backend, "get_memory_entities", None))


# ── postgres backend ──────────────────────────────────────────────────────


def _pg(rows: list[dict[str, Any]], monkeypatch) -> tuple[PostgresBackend, list[tuple]]:
    backend = PostgresBackend()
    log: list[tuple] = []

    def fake_execute(query: str, params: Any = None, *, fetch: str = "all") -> Any:
        log.append((" ".join(query.split()), params, fetch))
        return rows

    monkeypatch.setattr(backend, "_execute", fake_execute)
    return backend, log


def test_pg_groups_entity_ids_by_memory(monkeypatch):
    backend, log = _pg(
        [
            {"memory_id": MEM_A, "entity_id": 7},
            {"memory_id": MEM_A, "entity_id": 42},
            {"memory_id": MEM_B, "entity_id": 42},
        ],
        monkeypatch,
    )
    assert backend.get_memory_entities("work") == {MEM_A: [7, 42], MEM_B: [42]}
    sql, params, _fetch = log[0]
    assert "memory_entities" in sql
    assert params == {"p": "work"}


def test_pg_orders_for_a_deterministic_bundle(monkeypatch):
    """Two exports of an unchanged profile must produce identical MENTIONS
    lists, so the ordering is the server's job and not the dict's."""
    backend, log = _pg([], monkeypatch)
    backend.get_memory_entities("work")
    sql, _params, _fetch = log[0]
    assert "ORDER BY memory_id, entity_id" in sql


def test_pg_stringifies_uuid_keys(monkeypatch):
    """psycopg returns uuid.UUID for a uuid column. A UUID key never matches a
    str lookup, so MENTIONS would come out empty on every postgres export."""
    backend, _log = _pg([{"memory_id": UUID(MEM_A), "entity_id": 7}], monkeypatch)
    result = backend.get_memory_entities("work")
    assert result == {MEM_A: [7]}
    assert all(isinstance(k, str) for k in result)


# ── supabase backend ──────────────────────────────────────────────────────


class _FakeQuery:
    def __init__(self, rows, log):
        self._rows = rows
        self._log = log

    def select(self, *cols, **_k):
        self._log.append(("select", cols))
        return self

    def eq(self, col, val):
        self._log.append(("eq", col, val))
        return self

    def order(self, col, **_k):
        self._log.append(("order", col))
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.log: list[tuple] = []

    def table(self, name):
        self.log.append(("table", name))
        return _FakeQuery(self.rows, self.log)


def _sb(rows, monkeypatch) -> tuple[SupabaseBackend, _FakeClient]:
    backend = SupabaseBackend()
    client = _FakeClient(rows)
    monkeypatch.setattr(backend, "_get_client", lambda: client)
    return backend, client


def test_sb_groups_entity_ids_by_memory(monkeypatch):
    backend, client = _sb(
        [
            {"memory_id": MEM_A, "entity_id": 7},
            {"memory_id": MEM_A, "entity_id": 42},
            {"memory_id": MEM_B, "entity_id": 42},
        ],
        monkeypatch,
    )
    assert backend.get_memory_entities("work") == {MEM_A: [7, 42], MEM_B: [42]}
    assert ("table", "memory_entities") in client.log
    assert ("eq", "profile", "work") in client.log


def test_sb_orders_by_memory_then_entity(monkeypatch):
    """A stub cannot prove PostgREST sorts -- only that we asked it to."""
    backend, client = _sb([], monkeypatch)
    backend.get_memory_entities("work")
    assert ("order", "memory_id") in client.log
    assert ("order", "entity_id") in client.log


# ── gateway backend ───────────────────────────────────────────────────────


def test_gateway_refuses_rather_than_pretending_the_profile_is_empty():
    """Same shape as every other graph op on the gateway: an empty dict here is
    indistinguishable from a profile with no entity links. The facade is what
    turns the refusal into a degraded-but-valid export."""
    with pytest.raises(NotImplementedError, match="get_memory_entities"):
        GatewayBackend(url="https://example.invalid", api_key="k").get_memory_entities("work")


# ── database facade ───────────────────────────────────────────────────────


def test_facade_delegates_to_the_backend():
    mock = MagicMock()
    mock.get_memory_entities.return_value = {MEM_A: [7]}
    db._backend = mock

    assert db.get_memory_entities("work") == {MEM_A: [7]}
    mock.get_memory_entities.assert_called_once_with("work")


def test_facade_returns_empty_when_the_entities_layer_is_absent():
    """Pre-036 installs have no memory_entities table. MENTIONS is additive, so
    the export degrades instead of failing -- mirrors service.py's
    link_memory_entities guard on the write side."""
    mock = MagicMock()
    mock.get_memory_entities.side_effect = Exception('relation "memory_entities" does not exist')
    db._backend = mock

    assert db.get_memory_entities("work") == {}
