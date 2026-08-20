"""Unit tests for SupabaseEntityGraph -- mocked at the postgrest client
boundary (no real network / no real DB). Integration coverage against a
live Supabase project lives in TBU-122/TBU-123 (out of scope here).

Test double shape: a fake ``postgrest.SyncPostgrestClient`` whose
``.table(name)`` returns a fluent builder recording ``.eq()``/``.is_()``/
``.not_.is_()``/``.limit()`` filters, then dispatches ``.execute()``
against a small in-memory model. Mirrors the fluent-builder
mocking style implied by the existing ``tests/test_supabase_upsert_prefer_header.py``
(same client library), adapted here for a fully offline double instead of
a captured-request assertion, since SupabaseEntityGraph takes the client
via constructor injection (see ``inspect.signature`` test in
``tests/test_entity_graph_backend_protocol.py``).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from postgrest import SyncPostgrestClient

from ogham.entity_graph import Predicate
from ogham.supabase.entity_graph import SupabaseEntityGraph


class _Result:
    def __init__(self, data: Any):
        self.data = data


class _FakeBuilder:
    """Records the filter chain for one PostgREST call, then dispatches
    to the harness's handler for (table, op) on ``.execute()``."""

    def __init__(self, harness: "_Harness", table: str):
        self._harness = harness
        self._table = table
        self._op: str | None = None
        self._payload: dict[str, Any] | None = None
        self._on_conflict: str | None = None
        self._ignore_duplicates: bool = False
        self._filters: dict[str, Any] = {}
        self._not_filters: dict[str, Any] = {}
        self._order: tuple[str, bool] | None = None

    # -- operation entrypoints --
    def select(self, columns: str):
        self._op = "select"
        return self

    def insert(self, payload: dict[str, Any]):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict[str, Any]):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(
        self,
        payload: dict[str, Any],
        on_conflict: str | None = None,
        ignore_duplicates: bool = False,
    ):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        self._ignore_duplicates = ignore_duplicates
        return self

    # -- filters --
    def eq(self, col: str, val: Any):
        self._filters[col] = val
        return self

    def is_(self, col: str, val: Any):
        self._filters[col] = ("is", val)
        return self

    def limit(self, n):
        return self

    def order(self, col: str, desc: bool = False):
        # Mirrors postgrest-py's chainable .order(). The real client pushes the
        # sort to the database; recording it here keeps the double faithful to
        # the surface _resolve_to_id actually uses (TBU-245: two sides agreeing
        # by different mechanisms is how parity bugs hide).
        self._order = (col, desc)
        return self

    def contains(self, col: str, val: Any):
        self._filters[f"{col}__contains"] = val
        return self

    @property
    def not_(self):
        return _NotProxy(self)

    def execute(self):
        self._harness.calls.append(
            (self._table, self._op, dict(self._filters), self._payload, self._ignore_duplicates)
        )
        return self._harness.dispatch(self)


class _NotProxy:
    """Supports ``.not_.is_(col, val)`` -- records a negated IS filter."""

    def __init__(self, builder: _FakeBuilder):
        self._builder = builder

    def is_(self, col, val):
        self._builder._not_filters[col] = ("not_is", val)
        return self._builder


class _Harness:
    """In-memory model backing the fake postgrest client.

    ``entities``: canonical_name -> row dict
    ``entities_by_id``: id -> row dict
    ``aliases``: (alias, profile) -> entity_id
    ``edges``: list of edge row dicts (mutable -- store_triple mutates
    ``valid_to``/``superseded_by`` on matching rows, same as the real
    two-step PostgREST UPDATE-then-INSERT flow).
    """

    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.entities_by_id: dict[int, dict[str, Any]] = {}
        self.aliases: dict[tuple[Any, Any], int] = {}
        self.edges: list[dict[str, Any]] = []
        self.aliases_upserted: list[dict[str, Any]] = []
        self._next_edge_id = 1000
        self.calls: list[tuple[Any, ...]] = []

    def add_entity(self, entity_id: int, name: str, entity_type: str = "person") -> None:
        row: dict[str, Any] = {"id": entity_id, "canonical_name": name, "entity_type": entity_type}
        self.entities[name] = row
        self.entities_by_id[entity_id] = row

    def table(self, name: str) -> _FakeBuilder:
        return _FakeBuilder(self, name)

    def dispatch(self, b: _FakeBuilder) -> _Result:
        if b._table == "entities" and b._op == "select":
            name = b._filters.get("canonical_name")
            if name is not None:
                row = self.entities.get(name)
                return _Result([row] if row else [])
            eid = b._filters.get("id")
            row = self.entities_by_id.get(eid) if eid is not None else None
            return _Result([row] if row else [])

        if b._table == "entity_aliases" and b._op == "select":
            key = (b._filters.get("alias"), b._filters.get("profile"))
            eid = self.aliases.get(key)
            return _Result([{"entity_id": eid}] if eid is not None else [])

        if b._table == "entity_aliases" and b._op == "upsert":
            payload = b._payload
            assert payload is not None
            self.aliases_upserted.append(payload)
            key = (payload["alias"], payload["profile"])
            if b._ignore_duplicates and key in self.aliases:
                # ON CONFLICT DO NOTHING semantics: existing mapping wins.
                pass
            else:
                self.aliases[key] = payload["entity_id"]
            return _Result([payload])

        if b._table == "entity_edges" and b._op == "update":
            payload = b._payload
            assert payload is not None
            if "superseded_by" not in payload:
                # Supersession: stamp valid_to on the matching current row(s).
                for row in self.edges:
                    if self._matches_triple_filter(row, b._filters) and row["valid_to"] is None:
                        row["valid_to"] = payload["valid_to"]
            else:
                for row in self.edges:
                    if (
                        self._matches_triple_filter(row, b._filters)
                        and row["valid_to"] is not None
                        and row["superseded_by"] is None
                    ):
                        row["superseded_by"] = payload["superseded_by"]
            return _Result([])

        if b._table == "entity_edges" and b._op == "insert":
            payload = b._payload
            assert payload is not None
            new_id = self._next_edge_id
            self._next_edge_id += 1
            row = {
                "id": new_id,
                "subject_id": payload["subject_id"],
                "predicate": payload["predicate"],
                "object_id": payload["object_id"],
                "profile": payload["profile"],
                "fact_id": payload["fact_id"],
                "strength": payload["strength"],
                "metadata": payload["metadata"],
                "derived_from": payload.get("derived_from", []),
                "valid_from": "2026-07-02T00:00:00+00:00",
                "valid_to": None,
                "superseded_by": None,
            }
            self.edges.append(row)
            return _Result([row])

        if (
            b._table == "entity_edges"
            and b._op == "select"
            and "derived_from__contains" in b._filters
        ):
            needle = b._filters["derived_from__contains"]
            profile = b._filters.get("profile")
            rows = [
                r
                for r in self.edges
                if r["profile"] == profile
                and self._derived_from_contains(r["derived_from"], needle)
            ]
            return _Result(rows)

        if b._table == "entity_edges" and b._op == "select" and "id" in b._filters:
            eid = b._filters.get("id")
            profile = b._filters.get("profile")
            rows = [r for r in self.edges if r["id"] == eid and r["profile"] == profile]
            return _Result(rows)

        if b._table == "entity_edges" and b._op == "select":
            subj_id = b._filters.get("subject_id")
            obj_id = b._filters.get("object_id")
            predicate = b._filters.get("predicate")
            profile = b._filters.get("profile")
            rows = [
                r
                for r in self.edges
                if r["predicate"] == predicate
                and r["profile"] == profile
                and r["valid_to"] is None
                and (subj_id is None or r["subject_id"] == subj_id)
                and (obj_id is None or r["object_id"] == obj_id)
            ]
            return _Result(rows)

        raise AssertionError(
            f"unrouted call: table={b._table!r} op={b._op!r} filters={b._filters!r}"
        )

    @staticmethod
    def _derived_from_contains(derived_from: list[dict[str, Any]], needle: Any) -> bool:
        # Production code sends the needle JSON-encoded (a str) so postgrest-py's
        # .contains() takes its string branch instead of ",".join()-ing a list of
        # dicts (that TypeError is exactly what Fix 1 patches). Decode it here the
        # same way Postgres would interpret the `cs.<value>` filter server-side.
        parsed = json.loads(needle) if isinstance(needle, str) else needle
        (want,) = parsed
        return any(all(el.get(k) == v for k, v in want.items()) for el in derived_from)

    @staticmethod
    def _matches_triple_filter(row: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key in ("subject_id", "predicate", "object_id", "profile"):
            if row.get(key) != filters.get(key):
                return False
        return True


@pytest.fixture
def harness() -> _Harness:
    h = _Harness()
    h.add_entity(1, "Alice")
    h.add_entity(2, "Bob")
    h.add_entity(3, "Carol")
    return h


@pytest.fixture
def graph(harness: _Harness) -> SupabaseEntityGraph:
    # _Harness duck-types SyncPostgrestClient's `.table()` surface; cast
    # so pyright accepts the constructor call without SupabaseEntityGraph
    # having to widen to a Protocol just for tests.
    return SupabaseEntityGraph(
        cast(SyncPostgrestClient, harness), allowed_predicates={"KNOWS", "WORKS_WITH"}
    )


# ── store_triple ─────────────────────────────────────────────────────


def test_store_triple_new_edge_returns_id(harness, graph):
    new_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    assert new_id == 1000
    assert len(harness.edges) == 1
    assert harness.edges[0]["subject_id"] == 1
    assert harness.edges[0]["object_id"] == 2
    assert harness.edges[0]["valid_to"] is None


def test_store_triple_supersedes_prior_current_edge(harness, graph):
    first_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")
    second_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    assert second_id != first_id
    old_row = next(r for r in harness.edges if r["id"] == first_id)
    new_row = next(r for r in harness.edges if r["id"] == second_id)
    assert old_row["valid_to"] is not None
    # Regression guard: valid_to must be a parseable ISO-8601 timestamp,
    # not the literal string 'now()' -- Postgres does not auto-cast that
    # string when it comes from a PostgREST JSON payload (only bare
    # 'now', no parens, is a recognised special date/time literal). See
    # the supersede path in SupabaseEntityGraph.store_triple.
    datetime.fromisoformat(old_row["valid_to"])
    assert old_row["superseded_by"] == second_id
    assert new_row["valid_to"] is None


def test_store_triple_self_referential_raises(harness, graph):
    with pytest.raises(ValueError, match="self-referential"):
        graph.store_triple("Alice", Predicate("KNOWS"), "Alice", None, "work")


def test_store_triple_unresolvable_subject_raises(harness, graph):
    with pytest.raises(ValueError, match="cannot resolve"):
        graph.store_triple("Ghost", Predicate("KNOWS"), "Bob", None, "work")


def test_store_triple_accepts_int_ids_directly(harness, graph):
    new_id = graph.store_triple(1, Predicate("KNOWS"), 2, None, "work")
    assert new_id == 1000


def test_store_triple_carries_fact_id_and_metadata(harness, graph):
    fact_id = uuid4()
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", fact_id, "work", {"source": "test"})

    row = harness.edges[0]
    assert row["fact_id"] == str(fact_id)
    assert row["metadata"] == {"source": "test"}


# ── query_join ───────────────────────────────────────────────────────


def test_query_join_single_hop_outgoing_success(harness, graph):
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)

    assert result is not None
    assert {e.canonical_name for e in result.entities} == {"Alice", "Bob"}
    assert len(result.edges) == 1


def test_query_join_no_matching_edges_returns_none(harness, graph):
    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)
    assert result is None


def test_query_join_hop_limit_below_path_length_raises(harness, graph):
    with pytest.raises(ValueError, match="hop_limit"):
        graph.query_join(
            "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=1
        )


def test_query_join_unresolvable_start_returns_none(harness, graph):
    result = graph.query_join("Ghost", [Predicate("KNOWS")], "work", hop_limit=1)
    assert result is None


def test_query_join_incoming_direction(harness, graph):
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    result = graph.query_join(
        "Bob", [Predicate("KNOWS")], "work", hop_limit=1, direction="incoming"
    )

    assert result is not None
    assert {e.canonical_name for e in result.entities} == {"Alice", "Bob"}


def test_query_join_collects_citations_from_fact_id(harness, graph):
    fact_id = uuid4()
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", fact_id, "work")

    result = graph.query_join("Alice", [Predicate("KNOWS")], "work", hop_limit=1)

    assert result is not None
    assert result.citations == [fact_id]


def test_query_join_two_hop_traversal_succeeds(harness, graph):
    """A -KNOWS-> B -WORKS_WITH-> C: the BFS loop must accumulate both
    hops' edges/entities across separate ``cur_id`` iterations, not just
    the first."""
    fact_1, fact_2 = uuid4(), uuid4()
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", fact_1, "work")
    graph.store_triple("Bob", Predicate("WORKS_WITH"), "Carol", fact_2, "work")

    result = graph.query_join(
        "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is not None
    assert {e.id for e in result.entities} == {1, 2, 3}
    assert len(result.edges) == 2
    assert {e.predicate for e in result.edges} == {"KNOWS", "WORKS_WITH"}
    assert set(result.citations) == {fact_1, fact_2}


def test_query_join_two_hop_dead_end_mid_path_returns_none(harness, graph):
    """Only the first hop exists (A -KNOWS-> B); no WORKS_WITH edge out
    of B. The BFS must return None on the SECOND hop's dead end, not
    just the first."""
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    result = graph.query_join(
        "Alice", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is None


def test_query_join_entities_returned_in_bfs_insertion_order(harness, graph):
    """Regression guard for TBU-150 -- entities must be in traversal order,
    NOT sorted by id. Carol(id=3) -KNOWS-> Alice(id=1) -WORKS_WITH-> Bob(id=2):
    traversal order [3, 1, 2] does NOT match id-sorted order [1, 2, 3], so a
    lurking ``sorted(entities_by_id)`` would produce [1, 2, 3] and fail this
    test."""
    graph.store_triple("Carol", Predicate("KNOWS"), "Alice", None, "work")
    graph.store_triple("Alice", Predicate("WORKS_WITH"), "Bob", None, "work")

    result = graph.query_join(
        "Carol", [Predicate("KNOWS"), Predicate("WORKS_WITH")], "work", hop_limit=2
    )

    assert result is not None
    assert [e.id for e in result.entities] == [3, 1, 2]


# ── aliases ──────────────────────────────────────────────────────────


def test_add_alias_upserts_on_conflict_alias_profile(harness, graph):
    graph.add_alias(1, "Al", "work")

    assert len(harness.aliases_upserted) == 1
    assert harness.aliases_upserted[0] == {"entity_id": 1, "alias": "Al", "profile": "work"}
    assert harness.aliases[("Al", "work")] == 1


def test_add_alias_uses_ignore_duplicates_not_merge(harness, graph):
    """Must send ignore_duplicates=True (-> resolution=ignore-duplicates,
    i.e. ON CONFLICT DO NOTHING) -- NOT postgrest-py's default
    ignore_duplicates=False (-> resolution=merge-duplicates, ON CONFLICT
    DO UPDATE), which would silently repoint an existing alias. This
    keeps parity with the Postgres backend's
    `ON CONFLICT (alias, profile) DO NOTHING`."""
    graph.add_alias(1, "Al", "work")

    call = next(c for c in harness.calls if c[0] == "entity_aliases" and c[1] == "upsert")
    _, _, _, payload, ignore_duplicates = call
    assert payload == {"entity_id": 1, "alias": "Al", "profile": "work"}
    assert ignore_duplicates is True


def test_add_alias_first_write_wins_on_duplicate(harness, graph):
    """Cross-backend contract: a duplicate (alias, profile) must NOT
    repoint an existing alias to a new entity_id -- first-write-wins,
    matching the Postgres backend's ON CONFLICT DO NOTHING."""
    graph.add_alias(1, "AAPL", "work")
    graph.add_alias(2, "AAPL", "work")

    assert harness.aliases[("AAPL", "work")] == 1
    entity = graph.resolve_alias("AAPL", "work")
    assert entity is not None
    assert entity.id == 1


def test_resolve_alias_by_canonical_name(harness, graph):
    entity = graph.resolve_alias("Alice", "work")
    assert entity is not None
    assert entity.id == 1


def test_resolve_alias_by_alias_fallback(harness, graph):
    graph.add_alias(1, "Al", "work")

    entity = graph.resolve_alias("Al", "work")

    assert entity is not None
    assert entity.id == 1


def test_resolve_alias_not_found_returns_none(harness, graph):
    assert graph.resolve_alias("Ghost", "work") is None


def test_resolve_alias_by_int_id_skips_name_lookup(harness, graph):
    entity = graph.resolve_alias(2, "work")
    assert entity is not None
    assert entity.canonical_name == "Bob"


# ── provenance (TBU-124/125/126) ────────────────────────────────────


def test_store_triple_writes_derived_from(harness, graph):
    graph.store_triple(
        "Alice", Predicate("KNOWS"), "Bob", None, "work", derived_from=[{"source_edge_id": 5}]
    )

    assert harness.edges[0]["derived_from"] == [{"source_edge_id": 5}]


def test_store_triple_defaults_derived_from_to_empty_list(harness, graph):
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")

    assert harness.edges[0]["derived_from"] == []


def test_fetch_edge_returns_edge_with_derived_from(harness, graph):
    new_id = graph.store_triple(
        "Alice", Predicate("KNOWS"), "Bob", None, "work", derived_from=[{"source_edge_id": 5}]
    )

    edge = graph.fetch_edge(new_id, "work")

    assert edge is not None
    assert edge.id == new_id
    assert edge.derived_from == [{"source_edge_id": 5}]

    call = next(
        c for c in harness.calls if c[0] == "entity_edges" and c[1] == "select" and c[2].get("id")
    )
    assert call[2]["id"] == new_id
    assert call[2]["profile"] == "work"


def test_fetch_edge_missing_returns_none(harness, graph):
    assert graph.fetch_edge(9999, "work") is None


def test_fetch_edge_ignores_valid_to(harness, graph):
    """Provenance is historical -- fetch_edge must return a superseded edge too."""
    first_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")
    graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")  # supersedes first_id

    edge = graph.fetch_edge(first_id, "work")

    assert edge is not None
    assert edge.valid_to is not None


def test_find_citing_edges_by_source_edge_id(harness, graph):
    parent_id = graph.store_triple("Alice", Predicate("KNOWS"), "Bob", None, "work")
    child_id = graph.store_triple(
        "Bob",
        Predicate("WORKS_WITH"),
        "Carol",
        None,
        "work",
        derived_from=[{"source_edge_id": parent_id}],
    )

    edges = graph.find_citing_edges(source_edge_id=parent_id, source_memory_id=None, profile="work")

    assert [e.id for e in edges] == [child_id]

    call = next(
        c
        for c in harness.calls
        if c[0] == "entity_edges" and c[1] == "select" and "derived_from__contains" in c[2]
    )
    # JSON-encoded (str), not a raw list -- see Fix 1: postgrest-py's
    # .contains() only json.dumps() a dict; a list gets ",".join()-ed, which
    # crashes on a list of dicts. The needle is pre-encoded to take the
    # string branch instead.
    needle = call[2]["derived_from__contains"]
    assert isinstance(needle, str)
    assert json.loads(needle) == [{"source_edge_id": parent_id}]
    assert call[2]["profile"] == "work"


def test_find_citing_edges_by_source_memory_id(harness, graph):
    graph.store_triple(
        "Alice",
        Predicate("KNOWS"),
        "Bob",
        None,
        "work",
        derived_from=[{"source_memory_id": "mem-1"}],
    )

    edges = graph.find_citing_edges(source_edge_id=None, source_memory_id="mem-1", profile="work")

    assert len(edges) == 1
    assert edges[0].derived_from == [{"source_memory_id": "mem-1"}]


def test_find_citing_edges_no_source_returns_empty(harness, graph):
    assert graph.find_citing_edges(source_edge_id=None, source_memory_id=None, profile="work") == []


def test_real_postgrest_contains_accepts_json_encoded_needle():
    """Regression guard for Fix 1 -- uses the REAL postgrest-py library, not
    the mocked fake (which stores the raw value and would mask this).

    postgrest-py v2.31.0's `.contains()` only `json.dumps()`s a `dict`
    value; for a `list` it assumes a native Postgres array column and does
    `",".join(value)`, which raises `TypeError: sequence item 0: expected
    str instance, dict found` on a list of dicts like
    `[{"source_edge_id": 5}]`. `.contains()` builds the query string before
    any network call, so this needs no real server -- a dummy URL is enough
    to prove the call constructs without raising.
    """
    needle = json.dumps([{"source_edge_id": 5}])
    client = SyncPostgrestClient("http://localhost")

    # Must not raise -- this is exactly the call SupabaseEntityGraph.find_citing_edges
    # issues. Before Fix 1 (passing the raw list), this raised TypeError.
    client.table("entity_edges").select("*").eq("profile", "work").contains("derived_from", needle)
