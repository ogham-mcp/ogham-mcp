"""Reading a bundle's entities/ layer back (TBU-219, the reader for
``ogham_graph_version: 1``).

Scope decided 2026-08-20: this imports YOUR OWN bundles. Profiles are a
convenience namespace rather than a trust boundary, and bundles are not treated
as untrusted input -- the caps in ``bundle.py`` are corruption guards, not an
adversarial threat model. Importing a third-party bundle is not supported.

The tests that matter here are the allowlist ones. The drafted design used a
DENYLIST of non-triple keys, which meant any unknown list-valued key holding
wiki-link-shaped values silently became an edge -- and vault-ld SPEC 4.3 names
three host-tool keys a conforming tool MUST NOT emit as triples, of which the
draft covered one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ogham.entity_graph import Entity, EntityEdge, Predicate
from ogham.okf.bundle import export_okf_bundle, import_okf_graph
from ogham.okf.entities import entity_id_from_note_name, frontmatter_to_entity

ENTITIES = [
    Entity(id=42, canonical_name="Ogham", entity_type="project"),
    Entity(id=88, canonical_name="Entity Graph", entity_type="component"),
]
EDGES = [
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
MEMORIES = [{"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "content": "hello", "tags": []}]


class TestRoundTrip:
    def test_every_exported_entity_comes_back(self, tmp_path):
        out = tmp_path / "bundle"
        export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
        concepts, stats = import_okf_graph(out)
        assert stats["total"] == 2
        assert stats["graph_present"] is True
        by_name = {c.canonical_name: c for c in concepts}
        assert set(by_name) == {"Ogham", "Entity Graph"}
        assert by_name["Ogham"].entity_type == "project"
        assert by_name["Ogham"].source_entity_id == 42

    def test_the_edge_comes_back_pointing_at_the_right_note(self, tmp_path):
        out = tmp_path / "bundle"
        export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
        concepts, stats = import_okf_graph(out)
        subject = next(c for c in concepts if c.canonical_name == "Ogham")
        assert subject.edges == [("OWNS", "entity-graph-e88")]
        assert stats["edge_count"] == 1

    def test_no_inverse_is_invented_on_import(self, tmp_path):
        """D4 holds in both directions -- export emits one row, import reads one."""
        out = tmp_path / "bundle"
        export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
        concepts, _ = import_okf_graph(out)
        obj = next(c for c in concepts if c.canonical_name == "Entity Graph")
        assert obj.edges == []

    def test_a_bundle_with_no_graph_is_not_an_error(self, tmp_path):
        out = tmp_path / "bundle"
        export_okf_bundle(MEMORIES, out, {"producer": "t"})
        concepts, stats = import_okf_graph(out)
        assert concepts == []
        assert stats["graph_present"] is False


class TestAllowlist:
    """An allowlist of the 16 known predicates, not a denylist of known non-triples."""

    BASE = {"type": "Entity", "entity_id": 7, "canonical_name": "X", "entity_type": "entity"}

    @pytest.mark.parametrize("host_key", ["tags", "aliases", "cssclasses"])
    def test_host_tool_keys_never_become_edges(self, host_key):
        """SPEC 4.3 names these three. The drafted denylist covered one."""
        fm = dict(self.BASE, **{host_key: ["[[something-e9]]"]})
        parsed = frontmatter_to_entity(fm, "x-e7")
        assert parsed is not None
        assert parsed.edges == []

    def test_an_unknown_predicate_is_not_an_edge(self):
        fm = dict(self.BASE, INVENTED_PREDICATE=["[[target-e9]]"])
        parsed = frontmatter_to_entity(fm, "x-e7")
        assert parsed is not None and parsed.edges == []

    def test_a_known_predicate_is_an_edge(self):
        fm = dict(self.BASE, OWNS=["[[target-e9]]"])
        parsed = frontmatter_to_entity(fm, "x-e7")
        assert parsed is not None and parsed.edges == [("OWNS", "target-e9")]

    def test_dangling_records_are_not_resurrected_as_edges(self):
        """The exporter writes ogham_dangling BECAUSE the object was absent."""
        fm = dict(self.BASE, ogham_dangling=[{"predicate": "OWNS", "object_id": 999}])
        parsed = frontmatter_to_entity(fm, "x-e7")
        assert parsed is not None and parsed.edges == []

    def test_aliases_are_read_as_aliases_not_as_triples(self):
        fm = dict(self.BASE, aliases=["Ex", "Ecks"])
        parsed = frontmatter_to_entity(fm, "x-e7")
        assert parsed is not None
        assert parsed.aliases == ["Ex", "Ecks"]
        assert parsed.edges == []


class TestMalformedInput:
    def test_a_non_entity_concept_is_rejected(self):
        assert frontmatter_to_entity({"type": "Memory", "id": "x"}, "x") is None

    @pytest.mark.parametrize(
        "fm",
        [
            {"type": "Entity", "entity_type": "entity"},
            {"type": "Entity", "canonical_name": "X"},
            {"type": "Entity", "canonical_name": "", "entity_type": "entity"},
        ],
    )
    def test_a_concept_missing_its_natural_key_is_rejected(self, fm):
        """canonical_name + entity_type IS the key; neither half is optional."""
        assert frontmatter_to_entity(fm, "x-e1") is None

    def test_a_malformed_file_is_counted_not_crashed(self, tmp_path):
        out = tmp_path / "bundle"
        export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
        (out / "entities" / "broken.md").write_text("no frontmatter here", encoding="utf-8")
        concepts, stats = import_okf_graph(out)
        assert stats["total"] == 2
        assert stats["skipped_count"] == 1
        assert len(concepts) == 2

    def test_source_id_falls_back_to_the_filename(self):
        """entity_id is an extension; the -e{id} suffix is the spec-level carrier."""
        fm = {"type": "Entity", "canonical_name": "X", "entity_type": "entity"}
        parsed = frontmatter_to_entity(fm, "x-e123")
        assert parsed is not None and parsed.source_entity_id == 123

    def test_a_note_without_the_suffix_has_no_source_id(self):
        assert entity_id_from_note_name("plain-note") is None
