import textwrap
from pathlib import Path

from ogham.okf.viewer.concepts import ViewerConcept, extract_links, parse_bundle


def test_extract_links_intra_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "memories").mkdir()
    doc = bundle / "memories" / "alpha-abc12345.md"
    doc.touch()
    body = (
        "See [beta](beta-def45678.md) and [gamma](../topics/gamma.md).\n"
        "Outside: [google](https://google.com), [mail](mailto:x@y).\n"
        "Anchor: [section](beta-def45678.md#section)."
    )
    links = extract_links(body, doc, bundle)
    assert links == [
        "memories/beta-def45678",
        "topics/gamma",
        "memories/beta-def45678",
    ]


def test_extract_links_skips_external(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    doc = bundle / "x.md"
    doc.touch()
    body = "[a](https://example.com) [b](http://x.y) [c](mailto:a@b)"
    assert extract_links(body, doc, bundle) == []


def test_extract_links_resolves_above_bundle_root(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "sub").mkdir()
    doc = bundle / "sub" / "x.md"
    doc.touch()
    body = "[escape](../../outside.md)"
    assert extract_links(body, doc, bundle) == []


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())


def test_parse_bundle_skips_reserved_and_extracts_links(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write(bundle / "index.md", "---\nokf_version: '0.1'\n---\n# Index\n")
    _write(
        bundle / "memories" / "alpha-abc12345.md",
        """
        ---
        type: Decision
        title: Alpha decision
        tags: [project:demo]
        ---

        Body. See [beta](beta-def45678.md).
        """,
    )
    _write(
        bundle / "memories" / "beta-def45678.md",
        """
        ---
        type: Memory
        ---

        Beta body, no links.
        """,
    )
    _write(bundle / "memories" / "log.md", "Chronology, should be skipped.\n")

    concepts = parse_bundle(bundle)
    by_id = {c.id: c for c in concepts}

    assert set(by_id) == {"memories/alpha-abc12345", "memories/beta-def45678"}
    alpha = by_id["memories/alpha-abc12345"]
    assert alpha.type == "Decision"
    assert alpha.title == "Alpha decision"
    assert alpha.tags == ["project:demo"]
    assert alpha.links_to == ["memories/beta-def45678"]
    assert by_id["memories/beta-def45678"].title == "beta-def45678"
    # Verify ViewerConcept type
    assert isinstance(alpha, ViewerConcept)


def test_entity_concepts_are_not_rendered_as_viewer_nodes(tmp_path):
    """The viewer models edges as markdown links in the BODY (extract_links).

    Entity concept bodies are empty by design -- their triples live in
    frontmatter as wiki links -- so ingesting them would render one grey,
    minimum-size, permanently disconnected node per entity and silently break
    the "one node per memory" reading of the graph. Same rglob shape that
    import_okf_bundle was patched for in daa1087; the viewer shares the pattern
    and was not covered.

    Skipping is the v0.18 answer because the graph is export-only. Teaching the
    viewer the frontmatter wiki-link grammar is a feature, not a fix.
    """
    bundle = tmp_path / "bundle"
    _write(
        bundle / "memories" / "alpha-abc12345.md",
        """---
        type: Decision
        MENTIONS:
          - '[[entities/ogham-e42]]'
        ---

        Body.
        """,
    )
    _write(
        bundle / "entities" / "ogham-e42.md",
        """---
        type: Entity
        entity_id: 42
        canonical_name: Ogham
        OWNS:
          - '[[entities/graph-e88]]'
        ---
        """,
    )
    _write(bundle / "entities" / "nested" / "deep-e99.md", "---\ntype: Entity\n---\n")

    concepts = parse_bundle(bundle)

    assert {c.id for c in concepts} == {"memories/alpha-abc12345"}
    assert not [c for c in concepts if c.type == "Entity"]
