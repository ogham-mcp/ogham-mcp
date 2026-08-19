"""End-to-end round-trip test: export -> import -> verify identity preservation.

Round-trip definition (locked in v0.15 design):
  - UUID survives byte-identically
  - content survives byte-identically
  - tags survive (modulo type:X re-derivation)
  - source survives
  - metadata extension fields survive (spec §4.1 round-trip preservation)

Fields that DO NOT survive (and should not):
  - embedding (regenerated)
  - access_count, last_accessed_at (runtime state)
  - created_at may be re-stamped depending on backend behaviour
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ogham.entity_graph import Entity, EntityEdge, Predicate
from ogham.okf.bundle import export_okf_bundle, import_okf_bundle
from ogham.okf.entities import resolve_wiki_link
from ogham.okf.serialization import read_concept


def _make_memory(id_: str, content: str, tags: list[str], metadata: dict | None = None) -> dict:
    return {
        "id": id_,
        "content": content,
        "tags": tags,
        "source": "claude-code",
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": metadata or {},
    }


def test_okf_roundtrip_preserves_identity(tmp_path: Path):
    """Export then re-import a fixed set of memories; assert identity preservation."""
    from ogham.okf import export_okf_bundle, import_okf_bundle

    original = [
        _make_memory(
            "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
            "Use UUID PKs because Supabase recommends",
            ["type:decision", "project:ogham"],
        ),
        _make_memory(
            "d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88",
            "Gemini batch returns nulls under load",
            ["type:gotcha"],
            metadata={"language": "en"},
        ),
        _make_memory(
            "2bf662d8-1869-48e1-bd9e-1e0eaae162af",
            "Generic memory with no type",
            ["project:ogham"],
        ),
    ]
    bundle_dir = tmp_path / "rt-bundle"
    manifest = {
        "producer": "ogham-mcp/test",
        "exported_at": "2026-06-17T00:00:00Z",
        "profile": "test",
    }

    export_okf_bundle(original, bundle_dir, manifest)
    imported, stats = import_okf_bundle(bundle_dir)

    assert stats["total"] == 3
    assert stats["missing_id_count"] == 0

    # Index by id for comparison
    imported_by_id = {m["id"]: m for m in imported}
    for orig in original:
        rt = imported_by_id[orig["id"]]
        assert rt["content"] == orig["content"], f"content drift for {orig['id']}"
        assert rt["source"] == orig["source"]
        # Tags: original type:X tags should be preserved (winner becomes OKF type
        # and is re-derived to tag on import; losers stay as tags throughout)
        assert sorted(rt["tags"]) == sorted(orig["tags"]), f"tag drift for {orig['id']}"
        # Metadata extension fields preserved per spec §4.1
        for k, v in orig["metadata"].items():
            assert rt["metadata"].get(k) == v, f"metadata drift for {orig['id']} key {k}"


def test_okf_roundtrip_handles_default_type_memory(tmp_path: Path):
    """A memory with no type:X tag round-trips as type=Memory and tag-namespace
    is NOT polluted by an injected `type:memory` tag.
    """
    from ogham.okf import export_okf_bundle, import_okf_bundle

    original = [_make_memory("11111111-2222-3333-4444-555555555555", "no type tag", ["project:x"])]
    bundle_dir = tmp_path / "rt-default"
    export_okf_bundle(original, bundle_dir, {"producer": "p", "exported_at": "t", "profile": "p"})
    imported, _ = import_okf_bundle(bundle_dir)
    assert imported[0]["tags"] == ["project:x"]
    assert "type:memory" not in imported[0]["tags"]


def test_okf_roundtrip_index_md_declares_okf_version(tmp_path: Path):
    """Exported bundle is conformant: index.md declares okf_version: 0.1."""
    import yaml

    from ogham.okf import export_okf_bundle

    export_okf_bundle(
        [_make_memory("11111111-2222-3333-4444-555555555555", "x", [])],
        tmp_path,
        {"producer": "p", "exported_at": "t", "profile": "p"},
    )
    index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
    yaml_block = index_text.split("---\n", 2)[1]
    parsed = yaml.safe_load(yaml_block)
    assert parsed["okf_version"] == "0.1"


# --- graph export fidelity (TBU-130) ---------------------------------------
# SPEC §5.6 puts fidelity on the GRAPH, not on either file. v0.18 exports but
# does not import the graph (D10), so what is assertable here is the EXPORT
# half of that contract: the bundle faithfully and deterministically represents
# the database's graph. Round-trip fidelity is an acceptance criterion of the
# deferred graph-import issue, not of this release.
#
# No production code belongs in this section. A red test here is a defect in
# the module that owns the behaviour.

ENTITIES = [
    Entity(id=42, canonical_name="Ogham", entity_type="project"),
    Entity(id=88, canonical_name="Entity Graph", entity_type="component"),
    Entity(id=99, canonical_name="Supabase", entity_type="service"),
]
# strength/valid_from are populated here precisely so the "no qualifiers reach
# the bundle" assertion below has something it could leak. valid_from is a real
# datetime, not the ISO string the plan text used -- EntityEdge annotates it as
# datetime and both backends hand one back.
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
        valid_from=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
        valid_to=None,
        derived_from=[],
    ),
    EntityEdge(
        id=2,
        subject_id=42,
        predicate=Predicate("DEPENDS_ON"),
        object_id=99,
        profile="default",
        fact_id=None,
        strength=0.8,
        metadata={},
        valid_from=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
        valid_to=None,
        derived_from=[],
    ),
]
GRAPH_MEMORIES = [
    {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "content": "Ogham owns the entity graph.\n\nSecond paragraph.",
        "tags": ["type:decision", "project:ogham"],
        "created_at": "2026-08-03T09:00:00+00:00",
        "source": "claude-code",
    }
]
ALIASES = {42: ["OpenBrain", "ogham-mcp"]}
BRIDGE = {"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": [42, 99]}


def _export_graph(path: Path) -> None:
    export_okf_bundle(
        GRAPH_MEMORIES,
        path,
        {"producer": "test"},
        entities=ENTITIES,
        edges=EDGES,
        aliases=ALIASES,
        memory_entities=BRIDGE,
    )


def _entity_fm(bundle: Path, name: str) -> dict:
    fm, _ = read_concept(bundle / "entities" / name)
    return fm


def _wiki_targets(fm: dict):
    """Every wiki-link target in a frontmatter dict, scalar or list-valued.

    Deliberately not keyed on a known predicate list: a link that appears under
    an unexpected key still has to point at a real file, and hard-coding the
    vocabulary here would let a new predicate escape the dangling check.
    """
    for value in fm.values():
        for item in value if isinstance(value, list) else [value]:
            target = resolve_wiki_link(item)
            if target is not None:
                yield target


def test_every_entity_becomes_a_concept(tmp_path: Path):
    out = tmp_path / "b"
    _export_graph(out)
    assert {p.name for p in (out / "entities").glob("*.md")} == {
        "ogham-e42.md",
        "entity-graph-e88.md",
        "supabase-e99.md",
    }


def test_every_edge_becomes_a_triple_on_its_subject(tmp_path: Path):
    out = tmp_path / "b"
    _export_graph(out)
    fm = _entity_fm(out, "ogham-e42.md")
    assert resolve_wiki_link(fm["OWNS"][0]) == "entity-graph-e88"
    assert resolve_wiki_link(fm["DEPENDS_ON"][0]) == "supabase-e99"


def test_edges_are_not_duplicated_onto_the_object(tmp_path: Path):
    """D4: emit the stored rows, synthesise no inverses. entity-graph is the
    OBJECT of OWNS and must carry no OWNED_BY it was never given -- `inverse` is
    metadata on entity_edge_predicates and neither backend writes the
    reciprocal row, so emitting one would put a fact in the bundle that is not
    in the database."""
    out = tmp_path / "b"
    _export_graph(out)
    fm = _entity_fm(out, "entity-graph-e88.md")
    assert "OWNED_BY" not in fm
    assert "OWNS" not in fm
    # The object of DEPENDS_ON gets the same treatment; asserting only one
    # direction on one predicate would pass against a partial inverse synthesis.
    assert "DEPENDED_ON_BY" not in _entity_fm(out, "supabase-e99.md")


def test_aliases_are_carried(tmp_path: Path):
    out = tmp_path / "b"
    _export_graph(out)
    assert _entity_fm(out, "ogham-e42.md")["aliases"] == ["OpenBrain", "ogham-mcp"]


def test_memory_concepts_link_to_their_entities(tmp_path: Path):
    out = tmp_path / "b"
    _export_graph(out)
    fm, _ = read_concept(next((out / "memories").glob("*.md")))
    assert {resolve_wiki_link(v) for v in fm["MENTIONS"]} == {"ogham-e42", "supabase-e99"}


def test_no_edge_survives_without_its_endpoint_concept(tmp_path: Path):
    """Every wiki link anywhere in the bundle must resolve to a file in the
    bundle -- entity triples and memory MENTIONS alike.

    Dangling is structurally impossible on the read side (list_entities is the
    union of memory_entities with every edge endpoint) and filtered on the
    write side (both link emitters resolve through the same path table). This
    asserts the outcome rather than trusting either mechanism, and it is the
    only check that covers memories/ and entities/ with one rule.
    """
    out = tmp_path / "b"
    _export_graph(out)
    present = {p.stem for p in out.rglob("*.md")}
    seen = 0
    for md in sorted(out.rglob("*.md")):
        if md.name == "index.md":
            continue
        fm, _ = read_concept(md)
        for target in _wiki_targets(fm):
            seen += 1
            assert target in present, f"{md.name}: [[{target}]] is dangling"
    # A bundle whose links all vanished would satisfy the loop vacuously.
    assert seen == 4, "expected 2 edge triples + 2 MENTIONS links"


def test_memories_still_import_from_a_graph_bearing_bundle(tmp_path: Path):
    """The graph is export-only, but the memory half must keep round-tripping.

    D10's one non-optional import-side change is the entities/ skip; without it
    this would report three extra bodyless "memories".
    """
    out = tmp_path / "b"
    _export_graph(out)
    memories, stats = import_okf_bundle(out)
    assert stats["total"] == 1
    assert memories[0]["content"] == GRAPH_MEMORIES[0]["content"]
    assert memories[0]["id"] == GRAPH_MEMORIES[0]["id"]


def test_a_second_export_of_the_same_graph_is_byte_identical(tmp_path: Path):
    """Determinism (D9): no timestamps, no set-iteration order, no dict churn in
    the concept files, or a bundle cannot be version-controlled and a diff stops
    meaning the graph changed. index.md carries exported_at and is excluded."""
    a, b = tmp_path / "a", tmp_path / "b"
    _export_graph(a)
    _export_graph(b)
    left_files = sorted(p for p in a.rglob("*") if p.is_file())
    assert {p.relative_to(a) for p in left_files} == {
        p.relative_to(b) for p in b.rglob("*") if p.is_file()
    }
    for left in left_files:
        if left.name == "index.md":
            continue
        right = b / left.relative_to(a)
        assert left.read_bytes() == right.read_bytes(), f"{left.name} is not deterministic"


def test_context_is_stable_across_exports(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    _export_graph(a)
    _export_graph(b)
    assert json.loads((a / "context.jsonld").read_text()) == json.loads(
        (b / "context.jsonld").read_text()
    )


def test_no_edge_qualifier_reaches_any_file_in_the_bundle(tmp_path: Path):
    """D8, reversed by the 2026-08-04 council, asserted on raw bytes.

    store_triple hardcodes strength = 1.0 and valid_from = now(), so a bundle
    carrying either would advertise fidelity the write path cannot deliver. The
    fixture edges deliberately set strength=0.8 and a fixed valid_from so a leak
    has a distinctive value to leak. Scanning bytes rather than parsed
    frontmatter also catches a sidecar file that no read_concept call would
    ever open.
    """
    out = tmp_path / "b"
    _export_graph(out)
    forbidden = ("ogham_edges", "strength", "valid_from", "valid_to", "fact_id", "derived_from")
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} carries edge qualifier {token!r}"
    assert "0.8" not in (out / "entities" / "ogham-e42.md").read_text(encoding="utf-8")


@pytest.mark.postgres_integration
def test_export_matches_the_database_graph(tmp_path, monkeypatch, pg_test_profile, pg_client):
    """Seed entities + edges in a real profile, export, assert the bundle
    matches what the database holds.

    This is the class of test TBU-214 exists to make CI actually run: the SQL
    layer is verified to install, never to behave. list_entities' union over
    memory_entities and entity_edges, and the ORDER BY that makes export
    deterministic, are only stubbed elsewhere.

    Comparison is on natural keys -- (canonical_name, entity_type) and
    (subject_name, predicate, object_name) -- never on ids. Ids are
    install-specific: `entities.id` is GENERATED ALWAYS AS IDENTITY, so the same
    graph seeded on two machines carries different numbers and an id-keyed
    assertion would only be testing this one database.

    The profile is shared and long-lived, so this asserts bundle == database
    rather than bundle == a fixed expected set; rows left by earlier runs are
    then a valid part of the graph under test instead of a source of flakes.
    """
    import uuid

    from ogham.database import get_entity_graph_and_vocab
    from ogham.export_import import export_memories

    graph = get_entity_graph_and_vocab()[0]
    uid = uuid.uuid4().hex[:8]

    def _seed(name: str, type_: str) -> int:
        # entities has UNIQUE(canonical_name, entity_type) and the scratch DB is
        # shared, so names are uuid-suffixed rather than fixed.
        row = pg_client.fetchone(
            "INSERT INTO entities(canonical_name, entity_type) VALUES (%(n)s, %(t)s) RETURNING id",
            {"n": f"{name}-{uid}", "t": type_},
        )
        return int(row["id"])

    subject = _seed("Ogham", "project")
    object_ = _seed("EntityGraph", "component")
    graph.store_triple(subject, Predicate("OWNS"), object_, None, pg_test_profile)

    monkeypatch.chdir(tmp_path)
    bundle = Path(export_memories(pg_test_profile, format="okf", include_viewer=False))

    db_entities = graph.list_entities(pg_test_profile)
    db_edges = graph.list_edges(pg_test_profile)
    name_by_id = {e.id: e.canonical_name for e in db_entities}
    assert (f"Ogham-{uid}", "project") in {(e.canonical_name, e.entity_type) for e in db_entities}

    bundle_entities = {}
    for md in (bundle / "entities").glob("*.md"):
        fm, _ = read_concept(md)
        bundle_entities[md.stem] = fm

    assert {(fm["canonical_name"], fm["entity_type"]) for fm in bundle_entities.values()} == {
        (e.canonical_name, e.entity_type) for e in db_entities
    }

    # Resolve each triple's object through the bundle's own files, so the
    # comparison never borrows an id from the database to interpret the bundle.
    bundle_triples = set()
    for stem, fm in bundle_entities.items():
        for key, value in fm.items():
            if not isinstance(value, list) or key == "aliases":
                continue
            for item in value:
                target = resolve_wiki_link(item)
                if target is None:
                    continue
                assert target in bundle_entities, f"{stem}: [[{target}]] is dangling"
                object_name = bundle_entities[target]["canonical_name"]
                bundle_triples.add((fm["canonical_name"], key, object_name))

    assert bundle_triples == {
        (name_by_id[e.subject_id], str(e.predicate), name_by_id[e.object_id]) for e in db_edges
    }
    assert (f"Ogham-{uid}", "OWNS", f"EntityGraph-{uid}") in bundle_triples


@pytest.mark.postgres_integration
def test_export_writes_memory_concepts_from_a_real_backend(
    tmp_path, monkeypatch, pg_test_profile, pg_client
):
    """Export a profile that actually CONTAINS memories, against real Postgres.

    This exists because its absence hid a crash. `test_export_matches_the_database_graph`
    seeds entities and edges but no memories, so the memory-concept loop never
    ran and the test passed in 0.29s while `ogham export --format okf` was
    unusable on Postgres: psycopg returns `memories.id` as `uuid.UUID`, and both
    `make_filename` (`.replace` on it) and `memory_to_frontmatter` (yaml cannot
    represent it) blew up on the first row. Broken since v0.15.0; every unit
    fixture uses a string id, and PostgREST returns strings, so only a real
    Postgres export with at least one memory surfaces it.

    Seeds the embedding as a zero vector so no provider is needed.
    """
    import uuid

    from ogham.export_import import export_memories

    uid = uuid.uuid4().hex[:8]
    memory_id = str(uuid.uuid4())
    pg_client.execute(
        "INSERT INTO memories(id, content, embedding, tags, profile, source) "
        "VALUES (%(id)s::uuid, %(c)s, %(e)s::vector, %(t)s, %(p)s, %(s)s)",
        {
            "id": memory_id,
            "c": f"Smoke memory {uid}",
            "e": "[" + ",".join(["0"] * 512) + "]",
            "t": ["type:decision"],
            "p": pg_test_profile,
            "s": "integration",
        },
    )

    monkeypatch.chdir(tmp_path)
    bundle = Path(export_memories(pg_test_profile, format="okf", include_viewer=False))

    concepts = list((bundle / "memories").glob("*.md"))
    assert concepts, "no memory concepts written"

    stems = {p.stem for p in concepts}
    assert any(s.endswith(memory_id.replace("-", "")[:8]) for s in stems), stems

    fm, _ = next(
        (read_concept(p) for p in concepts if p.stem.endswith(memory_id.replace("-", "")[:8]))
    )
    assert fm["id"] == memory_id
    assert isinstance(fm["id"], str)
