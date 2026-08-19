"""Build the root context.jsonld -- the vault-ld Appendix B lift.

Appendix B: an OKF bundle plus a root context is already valid linked data
"without modifying a single bundle file". The context aliases OKF's required
bare `type` onto `@type`, declares an `@vocab` default for producer-defined
fields, and gives each predicate a term definition pointing at the `ogham_uri`
shipped in v0.17 (migration 045).

Two decisions here are easy to get wrong in the direction that looks nicer.

**Do NOT add `"id": "@id"`.** SPEC §4.5 requires an explicit `id` to be a full
absolute IRI and calls a relative value non-conforming; our memory concepts emit
a bare UUID. Aliasing it would retroactively make every bundle Ogham has ever
exported non-conforming. Appendix B aliases `type`, and only `type`. Guarded by
tests/test_okf_context.py::test_id_is_NOT_aliased.

**Schema.org alignments are data, not term-definition entries.**
`owl:equivalentProperty` is an RDF assertion *about* a property, not a
name->IRI mapping, so it belongs in `@graph`. Expanded term definitions are
keyword-keyed (`@id`, `@type`, `@container`, `@reverse`, `@language`,
`@direction`, `@context`, `@prefix`, `@protected`, `@nest`); a custom key there
is at best ignored by a conforming processor, which is worse than an error --
the alignment would look present and emit no triples. In `@graph` it becomes a
real triple that a processor actually surfaces.

The whole document is plain JSON we write and (later) read ourselves, so no
JSON-LD processor dependency is needed for a context we control at both ends.
"""

from ogham.entity_graph import PREDICATE_URIS

CONTEXT_FILENAME = "context.jsonld"

_OWL_NS = "http://www.w3.org/2002/07/owl#"


def build_context(base: str) -> dict:
    """Compose the bundle-root JSON-LD document for ``base``.

    Returns `{"@context": {...}, "@graph": [...]}` -- the context maps names to
    IRIs, the graph carries the Schema.org equivalences. A consumer that only
    wants the mappings reads `["@context"]` and ignores the rest.
    """
    if not base.endswith("/"):
        # Without the trailing slash, @base + "ogham-e42" concatenates into
        # ".../bundleogham-e42" rather than resolving as a path segment.
        base = f"{base}/"

    context: dict = {
        "@base": base,
        "@vocab": f"{base}vocab#",
        "owl": _OWL_NS,
        # Appendix B step 1: keyword aliasing does the whole job for `type`.
        # YAML reserves a leading `@`, so without this alias every concept would
        # have to quote "@type": in its frontmatter.
        "type": "@type",
    }

    alignments: list[dict] = []
    # Sorted so the file is byte-stable across exports -- it is frozen into
    # every bundle, and a diff should mean the vocabulary changed.
    for predicate in sorted(PREDICATE_URIS):
        uris = PREDICATE_URIS[predicate]
        # "@type": "@id" coerces values to IRIs. These are object properties
        # whose values are wiki links, never literals (§4.3) -- without the
        # coercion a processor reads `OWNS: "[[entities/x-e88]]"` as a string.
        context[predicate] = {"@id": uris["ogham"], "@type": "@id"}

        schema_org = uris.get("schema_org")
        if schema_org:
            # Only the five WebFetch-verified alignments (TBU-129). The other
            # eleven resolve under @vocab rather than being stretched onto a
            # term they do not mean -- Appendix B step 3 supports exactly this,
            # and an approximate mapping is worse than an absent one.
            alignments.append(
                {
                    "@id": uris["ogham"],
                    "owl:equivalentProperty": {"@id": schema_org},
                }
            )

    return {"@context": context, "@graph": alignments}
