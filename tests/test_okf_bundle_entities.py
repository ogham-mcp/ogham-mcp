"""Bundle-level export: entities/ + context.jsonld, both additive.

The compatibility claim rests on this file: an OKF-only consumer must be able
to read every .md we write, and must not trip over what it does not know.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

from ogham.entity_graph import Entity, EntityEdge, Predicate
from ogham.okf import bundle as bundle_mod
from ogham.okf.bundle import export_okf_bundle, import_okf_bundle
from ogham.okf.serialization import read_concept

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
        # A real datetime, not the ISO string the plan text used: EntityEdge
        # annotates valid_from as datetime, and both backends hand back one.
        valid_from=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
        valid_to=None,
        derived_from=[],
    )
]
MEMORIES = [{"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "content": "hello", "tags": []}]


def _tree(root):
    """Bundle contents as {relative path: bytes} -- the byte-identity oracle."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_memories_only_export_is_byte_identical_to_before(tmp_path):
    """The regression guard: adding the graph must not perturb the old path.

    Empty entity/edge lists are the same case as None -- a caller whose profile
    simply has no graph yet must not get a different bundle from one on an
    install that predates the graph entirely.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    export_okf_bundle(MEMORIES, a, {"producer": "t"})
    export_okf_bundle(MEMORIES, b, {"producer": "t"}, entities=[], edges=[])
    assert not (a / "entities").exists()
    assert not (b / "entities").exists()
    assert not (a / "context.jsonld").exists()
    assert _tree(a) == _tree(b)


def test_export_writes_one_concept_per_entity(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    assert (out / "entities" / "ogham-e42.md").is_file()
    assert (out / "entities" / "entity-graph-e88.md").is_file()


def test_exported_entity_carries_the_edge_as_a_frontmatter_triple(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    fm, body = read_concept(out / "entities" / "ogham-e42.md")
    assert fm["OWNS"] == ["[[entities/entity-graph-e88]]"]
    assert body.strip() == ""


def test_an_edge_appears_only_on_its_subject(tmp_path):
    """D4: edges are emitted once, on the subject, and no inverse is synthesised."""
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    fm, _ = read_concept(out / "entities" / "entity-graph-e88.md")
    assert "OWNS" not in fm
    assert "OWNED_BY" not in fm


def test_aliases_reach_the_entity_concept(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(
        MEMORIES,
        out,
        {"producer": "t"},
        entities=ENTITIES,
        edges=EDGES,
        aliases={42: ["OpenBrain", "ogham-mcp"]},
    )
    fm, _ = read_concept(out / "entities" / "ogham-e42.md")
    assert fm["aliases"] == ["OpenBrain", "ogham-mcp"]
    other, _ = read_concept(out / "entities" / "entity-graph-e88.md")
    assert "aliases" not in other


def test_export_writes_a_parseable_context(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    doc = json.loads((out / "context.jsonld").read_text())
    ctx = doc["@context"]
    assert ctx["type"] == "@type"
    assert "id" not in ctx
    # The Schema.org alignments are RDF assertions, so they live in @graph --
    # writing only the @context sub-dict would silently drop all five.
    assert doc["@graph"]


def test_context_base_is_overridable(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(
        MEMORIES,
        out,
        {"producer": "t"},
        entities=ENTITIES,
        edges=EDGES,
        base="https://example.org/mine/",
    )
    ctx = json.loads((out / "context.jsonld").read_text())["@context"]
    assert ctx["@base"] == "https://example.org/mine/"


def test_index_declares_the_graph_layer_version(tmp_path):
    """D11. okf_version covers the container; nothing covered our entities/
    layer. A future importer needs something to branch on."""
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    fm, _ = read_concept(out / "index.md")
    assert fm["ogham_graph_version"] == 1
    assert fm["okf_version"] == "0.1"


def test_memories_only_index_declares_no_graph_version(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"})
    fm, _ = read_concept(out / "index.md")
    assert "ogham_graph_version" not in fm


def test_the_index_is_written_exactly_once(tmp_path):
    """The manifest has to be finalised before write_index, not patched after.

    A second call would leave the bundle correct by luck of ordering while the
    graph version was absent from the first write -- so count the calls rather
    than trusting the file.
    """
    out = tmp_path / "bundle"
    with patch.object(bundle_mod, "write_index", wraps=bundle_mod.write_index) as spy:
        export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    assert spy.call_count == 1
    assert spy.call_args.args[1]["ogham_graph_version"] == 1


def test_the_caller_manifest_is_not_mutated(tmp_path):
    """The manifest belongs to the caller; export_memories reuses it."""
    manifest = {"producer": "t"}
    export_okf_bundle(MEMORIES, tmp_path / "b", manifest, entities=ENTITIES, edges=EDGES)
    assert manifest == {"producer": "t"}


def test_no_edge_qualifiers_reach_the_bundle(tmp_path):
    """D8 reversed. Guards the whole-file level, not just the marshaller."""
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    for md in (out / "entities").glob("*.md"):
        fm, _ = read_concept(md)
        assert "ogham_edges" not in fm
        assert "strength" not in fm


def test_every_written_md_file_has_the_type_field_okf_requires(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    for md in out.rglob("*.md"):
        if md.name == "index.md":
            continue
        fm, _ = read_concept(md)
        assert isinstance(fm.get("type"), str) and fm["type"], f"{md} has no OKF type"


def test_memory_concepts_link_to_entities_when_the_bridge_is_supplied(tmp_path):
    out = tmp_path / "bundle"
    export_okf_bundle(
        MEMORIES,
        out,
        {"producer": "t"},
        entities=ENTITIES,
        edges=EDGES,
        memory_entities={"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": [42]},
    )
    fm, _ = read_concept(next((out / "memories").glob("*.md")))
    assert fm["MENTIONS"] == ["[[entities/ogham-e42]]"]


def test_a_memory_linked_to_an_unexported_entity_emits_no_broken_link(tmp_path):
    """The path map is the single table both sides resolve against. An entity id
    that is not in it has no concept file, so a link to it would dangle."""
    out = tmp_path / "bundle"
    export_okf_bundle(
        MEMORIES,
        out,
        {"producer": "t"},
        entities=ENTITIES,
        edges=EDGES,
        memory_entities={"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": [42, 777]},
    )
    fm, _ = read_concept(next((out / "memories").glob("*.md")))
    assert fm["MENTIONS"] == ["[[entities/ogham-e42]]"]


def test_entity_concepts_are_not_imported_as_memories(tmp_path):
    """The regression this task exists to prevent. `rglob("*.md")` sweeps every
    Markdown file into the memory list, and _RESERVED_FILENAMES only filters by
    file name -- so entities/ would arrive as junk memories.
    """
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    memories, stats = import_okf_bundle(out)
    assert len(memories) == 1
    assert stats["total"] == 1


def test_the_memory_import_signature_is_unchanged(tmp_path):
    """Nine call sites across four files depend on the 2-tuple. Widening it was
    considered and rejected -- the graph gets its own reader in a later release.
    """
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    result = import_okf_bundle(out)
    assert isinstance(result, tuple) and len(result) == 2


def test_a_legacy_memories_only_bundle_imports_exactly_as_before(tmp_path):
    """The skip must be inert on a bundle with no entities/ at all -- that is
    every bundle any prior release ever wrote."""
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"})
    memories, stats = import_okf_bundle(out)
    assert len(memories) == 1
    assert stats["total"] == 1
    assert stats["missing_id_count"] == 0
    assert stats["skipped_count"] == 0


def test_a_nested_entities_directory_is_also_skipped(tmp_path):
    """The skip is on the directory, not on a filename pattern -- rglob
    descends, so a nested layout must be excluded too."""
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    nested = out / "entities" / "sub"
    nested.mkdir()
    (nested / "extra.md").write_text("---\ntype: Entity\n---\nbody\n")
    memories, _ = import_okf_bundle(out)
    assert len(memories) == 1


def test_skipped_entities_are_not_counted_as_malformed(tmp_path):
    """A skip that ran as a parse failure would still keep entities out of the
    memory list, so len(memories) alone cannot tell the two apart. skipped_count
    is what the import tool warns the operator about; a deliberate omission must
    not show up there as data loss.
    """
    out = tmp_path / "bundle"
    export_okf_bundle(MEMORIES, out, {"producer": "t"}, entities=ENTITIES, edges=EDGES)
    _, stats = import_okf_bundle(out)
    assert stats["skipped_count"] == 0


def test_export_is_byte_stable_across_two_runs_of_the_same_graph(tmp_path):
    """D9: fidelity is asserted on the export half. A second export of an
    unchanged graph must diff clean, or bundles cannot be version-controlled."""
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        export_okf_bundle(
            MEMORIES,
            out,
            {"producer": "t"},
            entities=ENTITIES,
            edges=EDGES,
            aliases={42: ["OpenBrain"]},
            memory_entities={"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": [42]},
        )
    assert _tree(a) == _tree(b)
