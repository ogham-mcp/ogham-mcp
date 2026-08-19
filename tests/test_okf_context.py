"""The Appendix B lift: a root context.jsonld makes an untouched OKF bundle
valid linked data.

Two things are load-bearing here and both are easy to get wrong in the
"obviously nicer" direction:

* `id` must NOT be aliased onto `@id` -- see test_id_is_NOT_aliased;
* the Schema.org alignments must NOT live inside term definitions -- see
  test_alignments_are_data_not_term_definition_entries.
"""

import json

from ogham.entity_graph import PREDICATE_URIS, V1_PREDICATES
from ogham.okf.context import CONTEXT_FILENAME, build_context

BASE = "https://example.org/bundle/"

# The five WebFetch-verified alignments from TBU-129 (migration 045).
SCHEMA_ORG_ALIGNED = {"OWNS", "OWNED_BY", "MENTIONS", "PART_OF", "CONTAINS"}


def _ctx(base=BASE):
    return build_context(base)["@context"]


def _graph(base=BASE):
    return build_context(base).get("@graph", [])


# ── the lift itself (Appendix B step 1) ───────────────────────────────────


def test_filename_is_the_one_appendix_b_names():
    assert CONTEXT_FILENAME == "context.jsonld"


def test_type_is_aliased_onto_the_json_ld_keyword():
    """Appendix B step 1: keyword aliasing maps OKF's required bare `type` onto
    @type, which is what makes an unmodified OKF concept read as YAML-LD."""
    assert _ctx()["type"] == "@type"


def test_id_is_NOT_aliased():
    """SPEC §4.5: an explicit `id` MUST be a full absolute IRI, and "a relative
    value is non-conforming". Our memory concepts emit a bare UUID.

    Aliasing `id` onto `@id` looks like an obvious improvement and would
    retroactively make every bundle Ogham has ever exported non-conforming.
    Appendix B aliases `type`, and only `type`.
    """
    assert "id" not in _ctx()


def test_base_and_vocab_are_declared():
    ctx = _ctx()
    assert ctx["@base"] == BASE
    assert ctx["@vocab"] == f"{BASE}vocab#"


def test_base_is_normalised_to_end_in_a_slash():
    """Without the trailing slash, @base + "ogham-e42" concatenates into
    ".../bundleogham-e42"."""
    assert build_context("https://example.org/bundle")["@context"]["@base"] == BASE


# ── predicate terms ───────────────────────────────────────────────────────


def test_every_predicate_gets_a_term_pointing_at_its_ogham_uri():
    ctx = _ctx()
    for predicate, uris in PREDICATE_URIS.items():
        assert ctx[predicate]["@id"] == uris["ogham"]


def test_predicate_terms_coerce_their_values_to_iris():
    """Predicates are object properties whose values are wiki links, never
    literals (§4.3). Without "@type": "@id" a processor reads
    `OWNS: "[[entities/x-e88]]"` as a string, not an edge."""
    ctx = _ctx()
    for predicate in V1_PREDICATES:
        assert ctx[predicate]["@type"] == "@id"


def test_the_context_covers_exactly_the_shipped_vocabulary():
    """Ties the bundle's copy of the vocabulary to the Python constant, which
    test_schema_parity ties to the SQL seed. Without this the bundle is a third,
    unreconciled source of truth -- and it is the one frozen on someone else's
    disk."""
    ctx = _ctx()
    terms = {k for k in ctx if k in V1_PREDICATES}
    assert terms == V1_PREDICATES
    assert len(V1_PREDICATES) == 16


# ── Schema.org alignments live in @graph, not in term definitions ─────────


def test_alignments_are_data_not_term_definition_entries():
    """`owl:equivalentProperty` is an RDF assertion ABOUT a property, not a
    name->IRI mapping, so it does not belong in a term definition.

    Expanded term definitions are keyword-keyed (@id, @type, @container,
    @reverse, @language, @direction, @context, @prefix, @protected, @nest). A
    custom key there is at best ignored by a conforming processor -- which is
    worse than an error, because the alignment would look present and emit no
    triples. Carrying it in @graph makes it a real triple instead.
    """
    ctx = _ctx()
    for predicate in V1_PREDICATES:
        assert set(ctx[predicate]) <= {"@id", "@type"}, (
            f"{predicate} term definition carries a non-keyword entry"
        )


def test_exactly_the_five_verified_predicates_are_aligned():
    aligned = {
        node["@id"].rsplit("#", 1)[-1] for node in _graph() if "owl:equivalentProperty" in node
    }
    assert aligned == SCHEMA_ORG_ALIGNED


def test_each_alignment_names_the_verified_schema_org_term():
    by_predicate = {n["@id"].rsplit("#", 1)[-1]: n for n in _graph()}
    expected = {
        "OWNS": "https://schema.org/owns",
        "OWNED_BY": "https://schema.org/owner",
        "MENTIONS": "https://schema.org/mentions",
        "PART_OF": "https://schema.org/isPartOf",
        "CONTAINS": "https://schema.org/hasPart",
    }
    for predicate, schema_uri in expected.items():
        assert by_predicate[predicate]["owl:equivalentProperty"] == {"@id": schema_uri}


def test_alignment_subjects_are_the_ogham_uris():
    """The alignment asserts ogham:OWNS owl:equivalentProperty schema:owns --
    the subject is our IRI, not the bare predicate name."""
    for node in _graph():
        assert node["@id"].startswith("https://ogham-mcp.dev/vocab#")


def test_the_other_eleven_are_not_stretched_onto_terms_they_do_not_mean():
    """v0.17's deliberate decision, and Appendix B step 3 explicitly supports
    leaving a field resolving under @vocab rather than forcing a mapping."""
    aligned = {n["@id"].rsplit("#", 1)[-1] for n in _graph()}
    assert not (aligned - SCHEMA_ORG_ALIGNED)
    assert len(V1_PREDICATES - SCHEMA_ORG_ALIGNED) == 11


def test_owl_prefix_is_declared_so_the_alignments_resolve():
    assert _ctx()["owl"] == "http://www.w3.org/2002/07/owl#"


# ── shape and stability ───────────────────────────────────────────────────


def test_context_is_json_serialisable():
    json.dumps(build_context(BASE))


def test_context_is_deterministic():
    """It is frozen into every bundle; two exports of the same graph must not
    differ because a set iterated differently."""
    assert json.dumps(build_context(BASE)) == json.dumps(build_context(BASE))


def test_predicate_terms_are_emitted_in_sorted_order():
    ctx = _ctx()
    emitted = [k for k in ctx if k in V1_PREDICATES]
    assert emitted == sorted(emitted)
