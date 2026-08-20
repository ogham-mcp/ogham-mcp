"""Applying a parsed OKF graph to the database (TBU-219, write half).

The load-bearing test here is idempotency. The design council's finding was that
per-edge writes made retry destructive: a failure partway left a half-merged
graph, and re-running re-superseded every edge already written and re-stamped
``valid_from``, so the natural operator reflex -- run it again -- degraded data
on each attempt.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from ogham.okf.entities import ParsedEntityConcept
from ogham.okf.graph_import import apply_okf_graph
from ogham.postgres.entity_graph import PostgresEntityGraph

pytestmark = pytest.mark.postgres_integration

V1_VOCAB = ["OWNS", "MENTIONS", "PART_OF"]


@pytest.fixture
def graph(pg_url):
    pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        pg_url, min_size=1, max_size=2, open=True, kwargs={"row_factory": dict_row}
    )
    yield PostgresEntityGraph(pool, V1_VOCAB)
    pool.close()


def _concepts(suffix: str) -> list[ParsedEntityConcept]:
    """Two concepts and one edge, shaped exactly as import_okf_graph returns them."""
    a = ParsedEntityConcept(
        f"Ogham{suffix}", "project", 42, [], [("OWNS", f"graph{suffix}-e88")], f"ogham{suffix}-e42"
    )
    b = ParsedEntityConcept(f"Graph{suffix}", "component", 88, [], [], f"graph{suffix}-e88")
    return [a, b]


class TestApply:
    def test_entities_and_the_edge_land(self, graph):
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        stats = apply_okf_graph(_concepts(profile[-4:]), profile, graph)
        assert stats["entities"] == 2
        assert stats["edges_written"] == 1
        assert stats["unresolved_edges"] == 0
        edges = graph.list_edges(profile, current_only=True)
        assert len(edges) == 1
        assert str(edges[0].predicate) == "OWNS"

    def test_running_twice_writes_nothing_the_second_time(self, graph):
        """Retry must converge, not churn.

        Before this, the second run re-superseded the edge and re-stamped
        valid_from, so an operator retrying after a partial failure quietly
        rewrote history every attempt.
        """
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        concepts = _concepts(profile[-4:])
        first = apply_okf_graph(concepts, profile, graph)
        edge_id_before = graph.list_edges(profile, current_only=True)[0].id

        second = apply_okf_graph(concepts, profile, graph)

        assert first["edges_written"] == 1
        assert second["edges_written"] == 0, "a retry must not write the same edge again"
        assert second["edges_already_present"] == 1
        current = graph.list_edges(profile, current_only=True)
        assert len(current) == 1, "no duplicate, and nothing superseded"
        assert current[0].id == edge_id_before, "the original edge row must survive untouched"

    def test_an_edge_whose_object_is_absent_is_counted_not_dropped(self, graph):
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        orphan = ParsedEntityConcept(
            f"Lonely{profile[-4:]}",
            "project",
            1,
            [],
            [("OWNS", "missing-e999")],
            f"lonely{profile[-4:]}-e1",
        )
        stats = apply_okf_graph([orphan], profile, graph)
        assert stats["entities"] == 1
        assert stats["edges_written"] == 0
        assert stats["unresolved_edges"] == 1

    def test_an_unknown_predicate_is_refused_even_if_the_caller_supplies_it(self, graph):
        """The parser allowlists, but apply does not trust its caller."""
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        s = profile[-4:]
        a = ParsedEntityConcept(f"A{s}", "project", 1, [], [("INVENTED", f"b{s}-e2")], f"a{s}-e1")
        b = ParsedEntityConcept(f"B{s}", "project", 2, [], [], f"b{s}-e2")
        stats = apply_okf_graph([a, b], profile, graph)
        assert stats["edges_written"] == 0
        assert stats["unresolved_edges"] == 1

    def test_a_self_edge_is_counted_rather_than_raising_mid_import(self, graph):
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        s = profile[-4:]
        me = ParsedEntityConcept(
            f"Self{s}", "project", 1, [], [("OWNS", f"self{s}-e1")], f"self{s}-e1"
        )
        stats = apply_okf_graph([me], profile, graph)
        assert stats["edges_written"] == 0
        assert stats["unresolved_edges"] == 1


class TestUpsertEntity:
    def test_upsert_is_get_or_create_on_the_natural_key(self, graph):
        name = f"Upsert{uuid.uuid4().hex[:6]}"
        first = graph.upsert_entity(name, "project")
        again = graph.upsert_entity(name, "project")
        assert first == again, "same natural key must return the same id"

    def test_same_name_different_type_is_a_different_entity(self, graph):
        """The whole reason a bare name is not a key (TBU-274)."""
        name = f"Dual{uuid.uuid4().hex[:6]}"
        as_entity = graph.upsert_entity(name, "entity")
        as_error = graph.upsert_entity(name, "error")
        assert as_entity != as_error

    def test_upsert_does_not_inflate_mention_count(self, graph, pg_url):
        """An import is not a mention -- mention_count feeds ranking everywhere."""
        name = f"Mention{uuid.uuid4().hex[:6]}"
        eid = graph.upsert_entity(name, "project")
        for _ in range(3):
            graph.upsert_entity(name, "project")
        with ConnectionPool(
            pg_url, min_size=1, max_size=1, open=True, kwargs={"row_factory": dict_row}
        ) as p:
            with p.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT mention_count FROM entities WHERE id = %s", (eid,))
                row = cur.fetchone()
        assert row is not None
        assert row["mention_count"] == 0, "import must not look like four mentions"


class TestImportMemoriesWiring:
    """The graph layer is opt-in, and off means genuinely off."""

    def test_graph_import_is_off_by_default(self, tmp_path, monkeypatch):
        """`entities` is global, so a default-on graph import would mutate rows
        every profile reads, for callers that never asked."""
        from ogham import export_import

        called: list[str] = []
        monkeypatch.setattr(
            export_import,
            "_looks_like_okf_bundle_dir",
            lambda d: False,
        )
        monkeypatch.setattr(
            "ogham.okf.graph_import.apply_okf_graph",
            lambda *a, **k: called.append("applied") or {},
        )
        export_import.import_memories('{"memories": []}', "t")
        assert called == [], "nothing may touch the graph unless import_graph=True"

    def test_signature_exposes_the_flag(self):
        import inspect

        from ogham.export_import import import_memories

        sig = inspect.signature(import_memories)
        assert "import_graph" in sig.parameters
        assert sig.parameters["import_graph"].default is False


class TestDryRun:
    """A dry run exists because the undo does not.

    Importing into a populated profile merges rather than restores, and the
    obvious safety net -- snapshot first -- is built from the same export path,
    so restoring from it would flatten `strength` and re-stamp `valid_from`.
    When there is no reliable way back, a reliable way to look first is what is
    left.
    """

    def test_dry_run_writes_nothing(self, graph):
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        concepts = _concepts(profile[-4:])
        stats = apply_okf_graph(concepts, profile, graph, dry_run=True)
        assert stats["dry_run"] is True
        assert graph.list_edges(profile, current_only=True) == []
        assert stats["entities_new"] == 2, "both are new, and neither was created"

    def test_dry_run_predicts_the_real_run_exactly(self, graph):
        """The only property worth having. An indicative preview is worse than
        none -- it would be trusted and wrong."""
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        concepts = _concepts(profile[-4:])

        predicted = apply_okf_graph(concepts, profile, graph, dry_run=True)
        actual = apply_okf_graph(concepts, profile, graph)

        for key in ("entities", "edges_written", "edges_already_present", "unresolved_edges"):
            assert predicted[key] == actual[key], f"dry run mispredicted {key}"

    def test_dry_run_sees_what_already_exists_on_a_second_pass(self, graph):
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        concepts = _concepts(profile[-4:])
        apply_okf_graph(concepts, profile, graph)

        predicted = apply_okf_graph(concepts, profile, graph, dry_run=True)
        assert predicted["entities_existing"] == 2, "both entities already exist"
        assert predicted["entities_new"] == 0
        assert predicted["edges_written"] == 0
        assert predicted["edges_already_present"] == 1

    def test_dry_run_resolves_on_the_qualified_key(self, graph):
        """A bare-name lookup could find the wrong type and report a false
        'already exists' (TBU-274)."""
        s = uuid.uuid4().hex[:6]
        profile = f"okf-{uuid.uuid4().hex[:8]}"
        graph.upsert_entity(f"Dual{s}", "error")  # same name, DIFFERENT type
        concept = ParsedEntityConcept(f"Dual{s}", "project", 1, [], [], f"dual{s}-e1")

        stats = apply_okf_graph([concept], profile, graph, dry_run=True)
        assert stats["entities_new"] == 1, "the error: row must not count as this project: entity"


class TestEndToEndThroughImportMemories:
    """The path a user actually takes, not just the function underneath it.

    A bundle with entities and no memories, so this exercises the graph branch
    without needing an embedding provider.
    """

    @staticmethod
    def _bundle(tmp_path, suffix):
        from ogham.entity_graph import Entity, EntityEdge, Predicate
        from ogham.okf.bundle import export_okf_bundle

        entities = [
            Entity(id=42, canonical_name=f"Ogham{suffix}", entity_type="project"),
            Entity(id=88, canonical_name=f"Graph{suffix}", entity_type="component"),
        ]
        edges = [
            EntityEdge(
                id=1,
                subject_id=42,
                predicate=Predicate("OWNS"),
                object_id=88,
                profile="default",
                fact_id=None,
                strength=1.0,
                metadata={},
                valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_to=None,
            )
        ]
        out = tmp_path / "bundle"
        export_okf_bundle([], out, {"producer": "t"}, entities=entities, edges=edges)
        return out

    def test_graph_is_untouched_without_the_flag(self, tmp_path, graph):
        from ogham.export_import import import_memories

        profile = f"okf-{uuid.uuid4().hex[:8]}"
        out = self._bundle(tmp_path, profile[-4:])
        result = import_memories(str(out), profile=profile)
        assert "graph" not in result
        assert graph.list_edges(profile, current_only=True) == []

    def test_dry_run_reports_without_writing(self, tmp_path, graph):
        from ogham.export_import import import_memories

        profile = f"okf-{uuid.uuid4().hex[:8]}"
        out = self._bundle(tmp_path, profile[-4:])
        result = import_memories(str(out), profile=profile, import_graph=True, graph_dry_run=True)
        assert result["graph"]["dry_run"] is True
        assert result["graph"]["entities_new"] == 2
        assert graph.list_edges(profile, current_only=True) == [], "dry run must write nothing"

    def test_the_flag_imports_the_graph(self, tmp_path, graph):
        from ogham.export_import import import_memories

        profile = f"okf-{uuid.uuid4().hex[:8]}"
        out = self._bundle(tmp_path, profile[-4:])
        result = import_memories(str(out), profile=profile, import_graph=True)
        assert result["graph"]["edges_written"] == 1
        edges = graph.list_edges(profile, current_only=True)
        assert len(edges) == 1
        assert str(edges[0].predicate) == "OWNS"
