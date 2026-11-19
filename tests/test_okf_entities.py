"""Entity concepts and the wiki-link grammar of vault-ld SPEC §4.4.1.

Two spec rules drive every decision here.

§4.5 -- identity mints from the FILE NAME alone; folders never enter the IRI.
So file names must be unique across the whole bundle, not per directory. Memory
concepts already end `-{uuid8}`; entity concepts get `-e{entity_id}`.

§4.4.1 -- a wiki link's alias is display-only and MUST be ignored, a fragment
addresses a location rather than a resource, and a path disambiguates only:
resolution uses the final segment. Generation and resolution are two halves of
one contract, so both live here and are tested against each other.

Pure marshalling -- no filesystem, no SQL. bundle.py owns IO.
"""

from datetime import datetime, timezone

import pytest

from ogham.entity_graph import V1_PREDICATES, Entity, EntityEdge, Predicate
from ogham.okf.entities import (
    ENTITY_OKF_TYPE,
    entity_to_body,
    entity_to_frontmatter,
    make_entity_filename,
    predicate_key,
    resolve_wiki_link,
    wiki_link,
)
from ogham.okf.links import resolve_link_note_name

OGHAM = Entity(id=42, canonical_name="Ogham", entity_type="project")
GRAPH = Entity(id=88, canonical_name="Entity Graph", entity_type="component")
CLI = Entity(id=99, canonical_name="CLI", entity_type="component")

PATHS = {
    42: "entities/ogham-e42",
    88: "entities/entity-graph-e88",
    99: "entities/cli-e99",
}


def _edge(subject_id, predicate, object_id, *, edge_id=1, **kw):
    return EntityEdge(
        id=edge_id,
        subject_id=subject_id,
        predicate=Predicate(predicate),
        object_id=object_id,
        profile="work",
        fact_id=kw.get("fact_id"),
        strength=kw.get("strength", 1.0),
        metadata=kw.get("metadata", {}),
        valid_from=kw.get("valid_from", datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)),
        valid_to=None,
        derived_from=kw.get("derived_from", []),
    )


# ── filenames and identity (§4.5) ─────────────────────────────────────────


def test_entity_filename_carries_the_id_so_it_cannot_collide_with_a_memory():
    assert make_entity_filename(OGHAM) == "ogham-e42.md"
    assert make_entity_filename(GRAPH) == "entity-graph-e88.md"


def test_two_entities_sharing_a_name_across_types_get_distinct_filenames():
    """`entities` is UNIQUE (canonical_name, entity_type), so one name can exist
    under two types. Folders never enter the IRI (§4.5), so distinct file names
    are the only thing keeping them from colliding on one identity."""
    a = Entity(id=1, canonical_name="Ogham", entity_type="project")
    b = Entity(id=2, canonical_name="Ogham", entity_type="product")
    assert make_entity_filename(a) != make_entity_filename(b)


def test_an_entity_named_like_a_uuid8_does_not_collide_with_a_memory():
    """The `e` prefix is load-bearing, not decoration: without it an entity
    named "1a2b3c4d" would mint the same IRI as a memory whose uuid8 is
    "1a2b3c4d"."""
    weird = Entity(id=7, canonical_name="1a2b3c4d", entity_type="thing")
    assert make_entity_filename(weird) == "1a2b3c4d-e7.md"


def test_unnameable_entity_still_produces_a_usable_filename():
    blank = Entity(id=5, canonical_name="!!!", entity_type="thing")
    assert make_entity_filename(blank) == "untitled-e5.md"


# ── wiki-link grammar (§4.4.1) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[[entities/ogham-e42]]", "ogham-e42"),  # path disambiguates only
        ("[[ogham-e42]]", "ogham-e42"),  # bare link still resolves
        ("[[entities/ogham-e42|Ogham]]", "ogham-e42"),  # alias ignored
        ("[[ogham-e42|Ogham]]", "ogham-e42"),
        ("[[entities/ogham-e42#Heading]]", "ogham-e42"),  # fragment discarded
        ("  [[entities/ogham-e42]]  ", "ogham-e42"),  # surrounding space
        ("not a link", None),
        ("[[]]", None),
        ("", None),
        (None, None),
        (42, None),
    ],
)
def test_wiki_link_resolution_follows_the_spec_grammar(raw, expected):
    assert resolve_wiki_link(raw) == expected


def test_wiki_link_round_trips_through_resolution():
    assert resolve_wiki_link(wiki_link("entities/ogham-e42")) == "ogham-e42"


# ── edges as frontmatter triples (Appendix B §5, D4) ──────────────────────


def test_edges_become_frontmatter_properties_keyed_by_predicate():
    fm = entity_to_frontmatter(OGHAM, [_edge(42, "OWNS", 88)], [], PATHS)

    assert fm["type"] == ENTITY_OKF_TYPE
    # OKF v0.3 §4.1: producer keys are x-<producer>-<name> (TBU-297).
    assert fm["x-ogham-entity-id"] == 42
    assert fm["x-ogham-canonical-name"] == "Ogham"
    assert fm["x-ogham-entity-type"] == "project"
    # A path relative to entities/ (OKF §6 rule 3), not a v0.1 wiki link.
    assert fm["x-ogham-owns"] == ["entity-graph-e88.md"]
    for legacy in ("entity_id", "canonical_name", "entity_type", "OWNS"):
        assert legacy not in fm


def test_multiple_objects_under_one_predicate_are_a_list_in_edge_order():
    """entity_edges_current_uq keys on the OBJECT too, so one subject can hold
    several current objects under one predicate. list_edges returns them
    ordered by id and that order is preserved."""
    fm = entity_to_frontmatter(
        OGHAM,
        [_edge(42, "OWNS", 88, edge_id=1), _edge(42, "OWNS", 99, edge_id=2)],
        [],
        PATHS,
    )
    assert fm["x-ogham-owns"] == ["entity-graph-e88.md", "cli-e99.md"]


def test_distinct_predicates_get_distinct_keys():
    fm = entity_to_frontmatter(
        OGHAM,
        [_edge(42, "OWNS", 88, edge_id=1), _edge(42, "DEPENDS_ON", 99, edge_id=2)],
        [],
        PATHS,
    )
    assert fm["x-ogham-owns"] == ["entity-graph-e88.md"]
    assert fm["x-ogham-depends-on"] == ["cli-e99.md"]


def test_only_the_subjects_edges_are_emitted():
    """D4: emit the stored rows, synthesise no inverses. An edge whose subject
    is someone else must not appear on this concept even if it is passed in."""
    fm = entity_to_frontmatter(GRAPH, [_edge(42, "OWNS", 88)], [], PATHS)
    assert "x-ogham-owns" not in fm
    assert entity_to_body(GRAPH, [_edge(42, "OWNS", 88)], PATHS, {}) == ""


def test_written_links_resolve_back_to_the_note_they_name():
    """Generation and resolution are two halves of one contract (§4.4.1).
    Asserted even though nothing imports in v0.18 -- it is what stops the two
    halves drifting before the graph-import issue picks them up."""
    fm = entity_to_frontmatter(OGHAM, [_edge(42, "OWNS", 88)], [], PATHS)
    assert resolve_link_note_name(fm["x-ogham-owns"][0]) == make_entity_filename(GRAPH)[:-3]


def test_the_body_carries_the_same_edges_as_markdown_links():
    """OKF v0.3 §6.1: only [text](target) links are links. The body section is
    what a v0.3 consumer counts; the predicate is the text next to the link."""
    body = entity_to_body(
        OGHAM,
        [_edge(42, "OWNS", 88, edge_id=1), _edge(42, "DEPENDS_ON", 99, edge_id=2)],
        PATHS,
        {88: "Entity Graph", 99: "CLI"},
    )
    assert body == (
        "<!-- x-ogham-relations -->\n"
        "## Relations\n"
        "\n"
        "- DEPENDS_ON: [CLI](cli-e99.md)\n"
        "- OWNS: [Entity Graph](entity-graph-e88.md)\n"
        "<!-- /x-ogham-relations -->"
    )


def test_an_entity_with_no_linked_edges_has_an_empty_body():
    assert entity_to_body(OGHAM, [], PATHS, {}) == ""
    # A dangling edge has no file to link to, so it stays frontmatter-only.
    assert entity_to_body(OGHAM, [_edge(42, "OWNS", 777)], PATHS, {}) == ""


# ── what deliberately does NOT travel (D8) ────────────────────────────────


def test_no_qualifier_sidecar_is_emitted():
    """D8, reversed by design council 2026-08-04.

    store_triple hardcodes strength=1.0 and valid_from=now(), so two of the four
    proposed sidecar fields could never be restored -- and with graph import
    deferred, none of them can. Shipping the sidecar would freeze an invented
    shape for zero present benefit.
    """
    fm = entity_to_frontmatter(
        OGHAM,
        [
            _edge(
                42,
                "OWNS",
                88,
                strength=0.5,
                fact_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                derived_from=[{"source_memory_id": "abc"}],
            )
        ],
        [],
        PATHS,
    )

    assert "ogham_edges" not in fm
    for leaked in ("strength", "valid_from", "valid_to", "fact_id", "derived_from", "metadata"):
        assert leaked not in fm, f"{leaked} leaked into frontmatter"
    # The triple itself is still the point, and still there.
    assert fm["x-ogham-owns"] == ["entity-graph-e88.md"]


# ── dangling (§4.4.1 -- flag, never drop) ─────────────────────────────────


def test_edges_whose_object_is_absent_are_reported_not_dropped():
    fm = entity_to_frontmatter(OGHAM, [_edge(42, "OWNS", 777)], [], PATHS)

    assert fm.get("x-ogham-owns") is None
    assert fm["x-ogham-dangling"] == [{"predicate": "OWNS", "object_id": 777}]


def test_a_dangling_edge_does_not_suppress_its_healthy_siblings():
    fm = entity_to_frontmatter(
        OGHAM,
        [_edge(42, "OWNS", 777, edge_id=1), _edge(42, "OWNS", 88, edge_id=2)],
        [],
        PATHS,
    )
    assert fm["x-ogham-owns"] == ["entity-graph-e88.md"]
    assert len(fm["x-ogham-dangling"]) == 1


def test_no_dangling_key_when_every_endpoint_is_present():
    fm = entity_to_frontmatter(OGHAM, [_edge(42, "OWNS", 88)], [], PATHS)
    assert "x-ogham-dangling" not in fm


# ── aliases and minimal shape ─────────────────────────────────────────────


def test_aliases_are_carried():
    fm = entity_to_frontmatter(OGHAM, [], ["OpenBrain", "ogham-mcp"], PATHS)
    # Unprefixed on purpose (Kevin, 2026-11-19): Obsidian reads `aliases`
    # natively, and Obsidian is a supported source.
    assert fm["aliases"] == ["OpenBrain", "ogham-mcp"]
    assert "x-ogham-aliases" not in fm


def test_no_alias_key_when_there_are_none():
    assert "aliases" not in entity_to_frontmatter(OGHAM, [], [], PATHS)


def test_entity_concept_has_the_one_field_okf_requires():
    """The whole compatibility claim -- an OKF-only consumer reads this bundle
    unchanged -- rests on every concept carrying a non-empty `type`."""
    fm = entity_to_frontmatter(OGHAM, [], [], PATHS)
    assert isinstance(fm.get("type"), str) and fm["type"]


def test_type_comes_first_so_the_file_reads_as_okf_at_a_glance():
    fm = entity_to_frontmatter(OGHAM, [_edge(42, "OWNS", 88)], ["x"], PATHS)
    assert next(iter(fm)) == "type"


def test_no_predicate_can_collide_with_a_reserved_frontmatter_key():
    """Predicates become frontmatter keys, so a 17th predicate whose key
    spelling lands on `x-ogham-aliases` or `x-ogham-entity-id` would silently
    clobber an entity attribute. This fails the moment that becomes true."""
    reserved = {
        "type",
        "x-ogham-entity-id",
        "x-ogham-canonical-name",
        "x-ogham-entity-type",
        "aliases",
        "x-ogham-aliases",
        "x-ogham-dangling",
    }
    assert not ({predicate_key(p) for p in V1_PREDICATES} & reserved)
    # The v0.1 spellings are still read, so they must stay disjoint too.
    legacy = {"type", "entity_id", "canonical_name", "entity_type", "aliases", "ogham_dangling"}
    assert not (V1_PREDICATES & legacy)
