"""OKF v0.3 alignment (TBU-297) and the import fixes on the same path (TBU-253, TBU-277).

What this file pins, in the order a reviewer would ask about it:

* the marked body-links section round-trips memory content byte for byte, and
  never eats content that merely looks like it;
* export -> import -> export is byte-stable across the whole tree, including
  the body links, superseded memories and entity concepts;
* a v0.1 bundle (fixed bytes in tests/fixtures/okf_v01_bundle) still imports
  every field;
* a bundle with no index.md, or no okf_version, imports with a warning;
* counted the way an OKF v0.3 consumer counts (§6.1 body links + §6.2 path
  fields, resolved per §6), an exported bundle has edges -- and with the body
  sections disabled it has none, which is the bite check;
* importance is preserved or computed on import, never left to the default;
* graph import sets evidence_class explicitly.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ogham.entity_graph import Entity, EntityEdge, Predicate
from ogham.okf import bundle as bundle_mod
from ogham.okf.bundle import export_okf_bundle, import_okf_bundle, import_okf_graph
from ogham.okf.concept import (
    MENTIONS_KEY,
    frontmatter_to_memory,
    memory_to_body,
    memory_to_frontmatter,
)
from ogham.okf.entities import entity_id_from_note_name
from ogham.okf.links import (
    MENTIONS_SECTION,
    append_section,
    build_section,
    resolve_link_note_name,
    strip_section,
)
from ogham.okf.serialization import read_concept

FIXTURE_V01 = Path(__file__).parent / "fixtures" / "okf_v01_bundle"

# ── a graph-bearing profile with every v0.3 feature in it ─────────────────

ENTITIES = [
    Entity(id=42, canonical_name="Ogham", entity_type="project"),
    Entity(id=88, canonical_name="Entity Graph", entity_type="component"),
    Entity(id=99, canonical_name="Supabase", entity_type="service"),
]


def _edge(edge_id: int, subject: int, predicate: str, obj: int) -> EntityEdge:
    return EntityEdge(
        id=edge_id,
        subject_id=subject,
        predicate=Predicate(predicate),
        object_id=obj,
        profile="default",
        fact_id=None,
        strength=1.0,
        metadata={},
        valid_from=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
        valid_to=None,
        derived_from=[],
    )


# DEPENDS_ON listed before OWNS on purpose: the exporter orders by predicate,
# and a re-export reads edges back predicate by predicate.
EDGES = [_edge(1, 42, "OWNS", 88), _edge(2, 42, "DEPENDS_ON", 99), _edge(3, 88, "PART_OF", 42)]
ALIASES = {42: ["OpenBrain", "ogham-mcp"]}

OLD = "11111111-1111-4111-8111-111111111111"
NEW = "22222222-2222-4222-8222-222222222222"
ORPHAN = "33333333-3333-4333-8333-333333333333"
TRICKY = "44444444-4444-4444-8444-444444444444"
PLAIN = "55555555-5555-4555-8555-555555555555"

#: Content that ends in a complete, well-formed marked block. Exported with
#: real mentions, the exporter's section follows it; the importer must strip
#: only its own.
TRICKY_CONTENT = "Notes on the format.\n\n## Mentions\n\nnot a section.\n\n" + build_section(
    MENTIONS_SECTION, "Mentions", [("", "Fake", "../entities/fake-e1.md")]
)

MEMORIES: list[dict[str, Any]] = [
    {
        "id": OLD,
        "content": "Ogham runs on Supabase.",
        "tags": ["type:decision", "project:ogham"],
        "source": "claude-code",
        "created_at": "2026-08-01T09:00:00+00:00",
        "importance": 0.7,
        "superseded_by": NEW,  # replacement IS in the bundle
        "metadata": {},
    },
    {
        "id": NEW,
        "content": "Ogham runs on plain Postgres now.\n",
        "tags": ["type:decision"],
        "source": "claude-code",
        "created_at": "2026-08-02T09:00:00+00:00",
        "importance": 0.30000000000000004,
        "metadata": {},
    },
    {
        "id": ORPHAN,
        "content": "Superseded by something not exported.",
        "tags": [],
        "created_at": datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
        "importance": 0.2,
        "superseded_by": "99999999-9999-4999-8999-999999999999",
        "metadata": {},
    },
    {
        "id": TRICKY,
        "content": TRICKY_CONTENT,
        "tags": ["project:ogham"],
        "created_at": "2026-08-03T09:00:00+00:00",
        "importance": 0.5,
        "metadata": {},
    },
    {
        "id": PLAIN,
        "content": "",
        "tags": [],
        "created_at": "2026-08-03T10:00:00+00:00",
        "importance": 0.5,
        "metadata": {},
    },
]
BRIDGE = {OLD: [42, 99], NEW: [42], TRICKY: [88], PLAIN: [99]}
MANIFEST = {"producer": "test", "exported_at": "2026-11-19T00:00:00+00:00"}


def _export(out: Path, memories=MEMORIES, **kw) -> Path:
    export_okf_bundle(
        memories,
        out,
        MANIFEST,
        entities=kw.get("entities", ENTITIES),
        edges=kw.get("edges", EDGES),
        aliases=kw.get("aliases", ALIASES),
        memory_entities=kw.get("memory_entities", BRIDGE),
    )
    return out


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _eid(note_name: str) -> int:
    entity_id = entity_id_from_note_name(note_name)
    assert entity_id is not None, note_name
    return entity_id


def _reexport_inputs(bundle: Path) -> dict[str, Any]:
    """Turn what the importers return back into exporter inputs.

    This is the mapping the database performs between an import and the next
    export: `timestamp` lands in created_at, mentions become memory_entities
    rows, parsed concepts become entities/edges/aliases. Ids are the bundle's
    own (the -e{id} suffix), which a fresh profile would reproduce.
    """
    memories, stats = import_okf_bundle(bundle)
    concepts, _ = import_okf_graph(bundle)
    mems = [
        {
            "id": m["id"],
            "content": m["content"],
            "tags": m["tags"],
            "source": m["source"],
            "created_at": m["timestamp"],
            "importance": m["importance"],
            "superseded_by": m["superseded_by"],
            "metadata": m["metadata"],
        }
        for m in memories
    ]
    bridge = {m["id"]: [_eid(n) for n in m["mentions"]] for m in memories if m["mentions"]}
    entities = [
        Entity(id=_eid(c.note_name), canonical_name=c.canonical_name, entity_type=c.entity_type)
        for c in concepts
    ]
    edges = [
        _edge(i, _eid(c.note_name), predicate, _eid(target))
        for i, (c, (predicate, target)) in enumerate(
            ((c, e) for c in concepts for e in c.edges), start=1
        )
    ]
    aliases = {_eid(c.note_name): c.aliases for c in concepts if c.aliases}
    return {
        "memories": mems,
        "memory_entities": bridge,
        "entities": entities,
        "edges": edges,
        "aliases": aliases,
        "stats": stats,
    }


# ── the marked section ────────────────────────────────────────────────────

SECTION = build_section(MENTIONS_SECTION, "Mentions", [("", "Ogham", "../entities/ogham-e42.md")])


def test_the_marker_names_the_frontmatter_key_it_mirrors():
    assert MENTIONS_SECTION == MENTIONS_KEY == "x-ogham-mentions"


@pytest.mark.parametrize(
    "content",
    ["", "x", "x\n", "\n\n", "line one\n\nline two", TRICKY_CONTENT, SECTION, "## Mentions"],
)
def test_strip_inverts_append_for_any_content(content):
    assert strip_section(append_section(content, SECTION), MENTIONS_SECTION) == content


def test_content_that_merely_contains_a_mentions_heading_is_not_stripped():
    content = "Plan\n\n## Mentions\n\n- [Ogham](../entities/ogham-e42.md)\n"
    assert strip_section(content, MENTIONS_SECTION) == content
    # Without the structured key the importer does not strip at all, so even a
    # complete marked block that a human wrote into content survives.
    memory = frontmatter_to_memory({"type": "Memory"}, SECTION)
    assert memory["content"] == SECTION


def test_a_marked_block_followed_by_more_text_is_content():
    body = SECTION + "\n\nmore text"
    assert strip_section(body, MENTIONS_SECTION) == body


@pytest.mark.parametrize(
    "value,expected",
    [
        ("../entities/ogham-e42.md", "ogham-e42"),  # relative to the file (§6 rule 3)
        ("/entities/ogham-e42.md", "ogham-e42"),  # from the bundle root (§6 rule 2)
        ("ogham-e42.md", "ogham-e42"),
        ("ogham-e42.md#Heading", "ogham-e42"),
        ("[[entities/ogham-e42]]", "ogham-e42"),  # v0.1 wiki link
        ("[[entities/ogham-e42|Ogham]]", "ogham-e42"),
        ("https://example.org/ogham-e42.md", None),  # external (§6 rule 1)
        ("", None),
        (None, None),
    ],
)
def test_both_link_forms_and_legacy_wiki_links_resolve(value, expected):
    assert resolve_link_note_name(value) == expected


# ── memory concepts: lifecycle and importance ─────────────────────────────


def test_a_superseded_memory_is_deprecated_with_a_path_to_its_replacement():
    fm = memory_to_frontmatter(MEMORIES[0], superseded_by_path="memories/x-22222222.md")
    assert fm["status"] == "deprecated"
    assert fm["superseded_by"] == "x-22222222.md"  # relative to memories/
    assert fm["x-ogham-superseded-by"] == NEW


def test_a_replacement_outside_the_bundle_leaves_only_the_status():
    fm = memory_to_frontmatter(MEMORIES[2], superseded_by_path=None)
    assert fm["status"] == "deprecated"
    assert "superseded_by" not in fm
    # expires_at is NOT mapped to stale_after (decision 4).
    assert "stale_after" not in fm


def test_a_current_memory_carries_no_lifecycle_keys():
    fm = memory_to_frontmatter(MEMORIES[1])
    assert "status" not in fm
    assert "superseded_by" not in fm
    # Rounded to 4 dp on export: 0.2 + 0.1 must not reach the bundle as
    # 0.30000000000000004.
    assert fm["x-ogham-importance"] == 0.3


def test_export_resolves_the_replacement_path_only_when_it_was_written(tmp_path):
    out = _export(tmp_path / "b")
    old, _ = read_concept(out / "memories" / "ogham-runs-on-supabase-11111111.md")
    assert old["superseded_by"] == "ogham-runs-on-plain-postgres-now-22222222.md"
    assert (out / "memories" / old["superseded_by"]).is_file()
    orphan, _ = read_concept(out / "memories" / "superseded-by-something-not-exported-33333333.md")
    assert orphan["status"] == "deprecated"
    assert "superseded_by" not in orphan


def test_the_mentions_body_lists_entities_by_name_with_relative_links():
    body = memory_to_body("Text.", [("Ogham", "entities/ogham-e42"), ("A [b]", "entities/a-b-e7")])
    assert body == (
        "Text.\n\n"
        "<!-- x-ogham-mentions -->\n"
        "## Mentions\n"
        "\n"
        "- [Ogham](../entities/ogham-e42.md)\n"
        "- [A \\[b\\]](../entities/a-b-e7.md)\n"
        "<!-- /x-ogham-mentions -->"
    )


# ── round trip: byte-stable over the whole tree ───────────────────────────


def test_export_import_export_is_byte_stable(tmp_path):
    first = _export(tmp_path / "first")
    back = _reexport_inputs(first)
    assert back["stats"]["total"] == len(MEMORIES)
    second = tmp_path / "second"
    export_okf_bundle(
        back["memories"],
        second,
        MANIFEST,
        entities=back["entities"],
        edges=back["edges"],
        aliases=back["aliases"],
        memory_entities=back["memory_entities"],
    )
    left, right = _tree(first), _tree(second)
    assert sorted(left) == sorted(right)
    for name in left:
        assert left[name] == right[name], f"{name} changed across the round trip"
    # And the round trip covered what it claims to.
    assert any(b"<!-- x-ogham-mentions -->" in v for v in left.values())
    assert any(b"<!-- x-ogham-relations -->" in v for v in left.values())
    assert any(b"superseded_by:" in v for v in left.values())


def test_round_trip_preserves_content_exactly(tmp_path):
    imported, _ = import_okf_bundle(_export(tmp_path / "b"))
    by_id = {m["id"]: m for m in imported}
    for original in MEMORIES:
        assert by_id[original["id"]]["content"] == original["content"]
    assert by_id[TRICKY]["content"] == TRICKY_CONTENT
    assert by_id[OLD]["mentions"] == ["ogham-e42", "supabase-e99"]
    assert by_id[OLD]["superseded_by"] == NEW
    assert by_id[OLD]["importance"] == 0.7
    assert "status" not in by_id[OLD]["metadata"]


# ── legacy v0.1 bundle ────────────────────────────────────────────────────


def test_a_v01_bundle_still_imports_every_memory_field():
    memories, stats = import_okf_bundle(FIXTURE_V01)
    assert stats["total"] == 1
    # README.md has no frontmatter. That has always counted as skipped
    # (unparseable), not as a typeless not-concept; unchanged here.
    assert stats["skipped_count"] == 1
    assert stats["not_concept_count"] == 0
    (m,) = memories
    assert m["id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
    assert m["source"] == "claude-code"
    assert m["timestamp"] == "2026-06-17T08:51:03.613750+00:00"
    assert m["mentions"] == ["ogham-e42", "supabase-e99"]
    assert m["content"] == "Use UUID PKs"
    assert "type:decision" in m["tags"]
    # MENTIONS is read as links, no longer filed into metadata as producer data.
    assert m["metadata"] == {"language": "en", "title": "Use UUID PKs"}


def test_a_v01_bundle_still_imports_every_entity_field():
    concepts, stats = import_okf_graph(FIXTURE_V01)
    by_name = {c.canonical_name: c for c in concepts}
    assert stats["total"] == 2
    ogham = by_name["Ogham"]
    assert ogham.entity_type == "project"
    assert ogham.source_entity_id == 42
    assert ogham.aliases == ["OpenBrain"]
    assert ogham.edges == [("OWNS", "supabase-e99")]  # dangling list is not an edge


def test_a_v01_bundle_reexports_as_v03(tmp_path):
    back = _reexport_inputs(FIXTURE_V01)
    out = tmp_path / "v03"
    export_okf_bundle(
        back["memories"],
        out,
        MANIFEST,
        entities=back["entities"],
        edges=back["edges"],
        aliases=back["aliases"],
        memory_entities=back["memory_entities"],
    )
    fm, body = read_concept(next((out / "memories").glob("*.md")))
    assert fm["x-ogham-id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
    assert fm["generated"]["at"] == "2026-06-17T08:51:03.613750+00:00"
    for legacy in ("id", "source", "timestamp", "MENTIONS"):
        assert legacy not in fm
    assert body.startswith("Use UUID PKs\n\n<!-- x-ogham-mentions -->")


# ── bundles without index.md or okf_version (§11.2, §12) ──────────────────


def _import(bundle: Path) -> dict[str, Any]:
    from ogham.export_import import import_memories

    with (
        patch(
            "ogham.export_import.generate_embeddings_batch",
            side_effect=lambda texts, **kw: [[0.0]] * len(texts),
        ),
        patch("ogham.export_import._upsert_memory") as upsert,
        patch("ogham.export_import.store_memories_batch"),
        patch("ogham.export_import.get_profile_ttl", return_value=None),
    ):
        result = import_memories(str(bundle), profile="p")
    result["_upserts"] = [c.args[0] for c in upsert.call_args_list]
    return result


def test_a_bundle_without_index_md_imports_and_warns(tmp_path, caplog):
    out = _export(tmp_path / "b")
    (out / "index.md").unlink()
    with caplog.at_level("WARNING", logger="ogham.export_import"):
        result = _import(out)
    assert result["imported"] == len(MEMORIES)
    assert any("no index.md" in w for w in result["warnings"])
    assert any("no index.md" in r.getMessage() for r in caplog.records)


def test_a_bundle_without_okf_version_imports_and_warns(tmp_path, caplog):
    out = _export(tmp_path / "b")
    (out / "index.md").write_text("---\ntitle: no version\n---\n# x\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="ogham.export_import"):
        result = _import(out)
    assert result["imported"] == len(MEMORIES)
    assert any("okf_version" in r.getMessage() for r in caplog.records)


def test_a_v01_bundle_imports_without_a_version_warning(tmp_path, caplog):
    copy = tmp_path / "v01"
    shutil.copytree(FIXTURE_V01, copy)
    with caplog.at_level("WARNING", logger="ogham.export_import"):
        result = _import(copy)
    assert result["imported"] == 1
    assert result["warnings"] == []


def test_a_directory_with_no_concepts_is_still_refused(tmp_path):
    (tmp_path / "README.md").write_text("# hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no OKF concept files"):
        _import(tmp_path)


# ── edge visibility, counted the way an OKF v0.3 consumer counts ──────────

#: §6.1 inline links, the same shape the upstream validator matches. Wiki
#: links do not match, which is the point.
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_PATH_FIELDS = ("resource", "superseded_by", "computation")


def _is_concept(path: Path) -> bool:
    if path.suffix != ".md" or path.name in ("index.md", "log.md") or not path.is_file():
        return False
    try:
        fm, _ = read_concept(path)
    except ValueError:
        return False
    return isinstance(fm.get("type"), str) and bool(fm["type"].strip())


def _resolve(bundle: Path, source: Path, target: str) -> Path | None:
    """OKF §6: a scheme is external; `/` is the bundle root; else the file's dir."""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
        return None
    clean = target.split("#", 1)[0]
    if not clean:
        return None
    base = bundle if clean.startswith("/") else source.parent
    candidate = Path(os.path.normpath(base / clean.lstrip("/")))
    return candidate if bundle in candidate.parents else None


def okf_v03_edge_count(bundle: Path) -> int:
    """Edges of the §6.4 bundle graph: body links and path fields to concepts."""
    bundle = bundle.resolve()
    edges = 0
    for path in sorted(bundle.rglob("*.md")):
        if not _is_concept(path):
            continue
        fm, body = read_concept(path)
        targets = list(_LINK.findall(body))
        targets += [fm[k] for k in _PATH_FIELDS if isinstance(fm.get(k), str)]
        sources = fm.get("sources") or []
        targets += [s["resource"] for s in sources if isinstance(s, dict) and "resource" in s]
        for target in targets:
            resolved = _resolve(bundle, path, target)
            if resolved is not None and _is_concept(resolved):
                edges += 1
    return edges


GRAPH_ONLY = [m for m in MEMORIES if not m.get("superseded_by")]


def test_an_exported_bundle_has_edges_an_okf_v03_consumer_can_see(tmp_path):
    """GRAPH_ONLY has no superseded memory, so every edge counted here comes
    from the body sections -- path fields contribute nothing."""
    out = _export(tmp_path / "b", memories=GRAPH_ONLY)
    # 3 memories x their mentions (1 + 1 + 1) + 3 entity relations.
    assert okf_v03_edge_count(out) == 6


def test_bite_check_without_the_body_sections_a_v03_consumer_sees_no_edges(tmp_path):
    """The proof the previous test bites: the same export with the body
    sections switched off -- frontmatter keys still present -- counts zero, so
    the `== 6` above would fail. That is what every v0.1 bundle looked like."""
    with (
        patch.object(bundle_mod, "memory_to_body", side_effect=lambda content, *a, **k: content),
        patch.object(bundle_mod, "entity_to_body", return_value=""),
    ):
        out = _export(tmp_path / "b", memories=GRAPH_ONLY)
    fm, _ = read_concept(next((out / "memories").glob("ogham-runs*")))
    assert fm["x-ogham-mentions"], "structured key still written"
    assert okf_v03_edge_count(out) == 0


def test_the_v01_fixture_has_no_edges_for_a_v03_consumer():
    """Wiki links are not links in OKF v0.3 (§6.1) -- the TBU-297 defect."""
    assert okf_v03_edge_count(FIXTURE_V01) == 0


def test_superseded_by_is_a_path_field_edge(tmp_path):
    with (
        patch.object(bundle_mod, "memory_to_body", side_effect=lambda content, *a, **k: content),
        patch.object(bundle_mod, "entity_to_body", return_value=""),
    ):
        out = _export(tmp_path / "b")
    assert okf_v03_edge_count(out) == 1  # OLD -> NEW, the only in-bundle replacement


# ── TBU-253: importance on import ─────────────────────────────────────────


def test_build_row_preserves_a_carried_importance():
    from ogham.export_import import _build_row

    row = _build_row({"content": "x", "importance": 0.9}, [0.0], "p", None)
    assert row["importance"] == 0.9


@pytest.mark.parametrize("carried", [None, 1.5, -0.1, "high", True])
def test_build_row_computes_a_missing_or_malformed_importance(carried):
    from ogham.export_import import _build_row
    from ogham.extraction import compute_importance

    content = "We decided to use `ogham/okf/links.py` for the error handling architecture."
    mem = {"content": content, "tags": ["a", "b", "c"]}
    if carried is not None:
        mem["importance"] = carried
    row = _build_row(mem, [0.0], "p", None)
    expected = compute_importance(content, ["a", "b", "c"])
    assert row["importance"] == expected
    assert expected != 0.5, "fixture must distinguish computed from the column default"


def test_okf_import_preserves_importance_on_upsert_and_computes_it_on_insert(tmp_path):
    from ogham.extraction import compute_importance
    from ogham.okf.serialization import write_concept

    (tmp_path / "memories").mkdir()
    write_concept(
        tmp_path / "memories" / "a.md",
        {"type": "Memory", "x-ogham-id": OLD, "x-ogham-importance": 0.8},
        "carried",
    )
    write_concept(tmp_path / "memories" / "b.md", {"type": "Memory", "x-ogham-id": NEW}, "absent")
    stored: list[dict] = []
    from ogham.export_import import import_memories

    with (
        patch(
            "ogham.export_import.generate_embeddings_batch",
            side_effect=lambda texts, **kw: [[0.0]] * len(texts),
        ),
        patch("ogham.export_import._upsert_memory", side_effect=stored.append),
        patch("ogham.export_import.get_profile_ttl", return_value=None),
    ):
        import_memories(str(tmp_path), profile="p")
    by_id = {m["id"]: m for m in stored}
    # Carried: the backend writes it, overwriting a stored value.
    assert by_id[OLD]["importance"] == 0.8
    # Not carried: no opinion. The backend keeps a stored importance and uses
    # importance_if_new only when it creates the row.
    assert by_id[NEW]["importance"] is None
    assert by_id[NEW]["importance_if_new"] == compute_importance("absent", [])


def test_okf_export_emits_importance(tmp_path):
    out = _export(tmp_path / "b", memories=[MEMORIES[0]], memory_entities={})
    fm, _ = read_concept(next((out / "memories").glob("*.md")))
    assert fm["x-ogham-importance"] == 0.7


def test_the_postgres_export_query_selects_importance(monkeypatch):
    """export_memories reads get_all_memories_full; before TBU-253 it never
    selected importance, so neither JSON nor OKF exports could carry it."""
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    seen: list[str] = []

    def fake_execute(query, params=None, *, fetch="all"):
        seen.append(query)
        return []

    monkeypatch.setattr(backend, "_execute", fake_execute)
    backend.get_all_memories_full("p")
    assert "importance" in seen[0]


def _pg_upsert(monkeypatch, memory: dict) -> tuple[str, dict]:
    from ogham.backends.postgres import PostgresBackend

    backend = PostgresBackend()
    calls: list[tuple] = []

    def fake_execute(query, params=None, *, fetch="all"):
        calls.append((" ".join(query.split()), params))
        return {"id": OLD}

    monkeypatch.setattr(backend, "_execute", fake_execute)
    backend.upsert_memory({"id": OLD, "content": "x", "embedding": [0.0], **memory})
    return calls[0]


#: The ON CONFLICT assignment that keeps a stored importance when the record
#: carries none. Removing the COALESCE is the bite check for the tests below.
_PG_PRESERVE = "importance = COALESCE(%(importance)s::float, memories.importance)"
_PG_INSERT = "COALESCE(%(importance)s::float, %(importance_if_new)s::float)"


def test_postgres_upsert_a_carried_importance_overwrites(monkeypatch):
    query, params = _pg_upsert(monkeypatch, {"importance": 0.8, "importance_if_new": 0.8})
    assert params["importance"] == 0.8
    # Non-null param: COALESCE yields it on insert and on conflict alike.
    assert _PG_INSERT in query and _PG_PRESERVE in query


def test_postgres_upsert_an_absent_importance_preserves_the_stored_value(monkeypatch):
    query, params = _pg_upsert(monkeypatch, {"importance": None, "importance_if_new": 0.6})
    assert params["importance"] is None
    conflict = query.split("ON CONFLICT", 1)[1]
    assert _PG_PRESERVE in conflict, "conflict branch must fall back to the stored importance"
    assert "EXCLUDED.importance" not in conflict


def test_postgres_upsert_an_absent_importance_on_a_new_row_uses_the_computed_value(monkeypatch):
    query, params = _pg_upsert(monkeypatch, {"importance": None, "importance_if_new": 0.6})
    assert params["importance_if_new"] == 0.6
    assert _PG_INSERT in query.split("ON CONFLICT", 1)[0]


def _sb_upsert(memory: dict, *, exists: bool) -> dict:
    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend()
    client = MagicMock()
    lookup = client.from_.return_value.select.return_value.eq.return_value.limit.return_value
    lookup.execute.return_value.data = [{"id": OLD}] if exists else []
    client.from_.return_value.upsert.return_value.execute.return_value.data = [{"id": OLD}]
    with patch.object(backend, "_get_client", return_value=client):
        backend.upsert_memory({"id": OLD, "content": "x", "embedding": [0.0], **memory})
    return client.from_.return_value.upsert.call_args.args[0]


def test_supabase_upsert_a_carried_importance_overwrites():
    row = _sb_upsert({"importance": 0.8, "importance_if_new": 0.8}, exists=True)
    assert row["importance"] == 0.8


def test_supabase_upsert_an_absent_importance_preserves_the_stored_value():
    """merge-duplicates overwrites every column in the payload, so leaving the
    stored value alone means leaving importance OUT of the payload."""
    row = _sb_upsert({"importance": None, "importance_if_new": 0.6}, exists=True)
    assert "importance" not in row


def test_supabase_upsert_an_absent_importance_on_a_new_row_uses_the_computed_value():
    row = _sb_upsert({"importance": None, "importance_if_new": 0.6}, exists=False)
    assert row["importance"] == 0.6


def test_supabase_upsert_an_absent_surprise_preserves_the_stored_value():
    """It used to send surprise=0.5 unconditionally, resetting it on re-import."""
    row = _sb_upsert({"importance": 0.4}, exists=True)
    assert "surprise" not in row


def test_supabase_upsert_a_new_row_gets_the_default_surprise():
    row = _sb_upsert({"importance": 0.4}, exists=False)
    assert row["surprise"] == 0.5


def test_supabase_upsert_a_carried_surprise_is_written():
    row = _sb_upsert({"importance": 0.4, "surprise": 0.8}, exists=True)
    assert row["surprise"] == 0.8


@pytest.mark.postgres_integration
def test_postgres_upsert_importance_against_a_real_database(pg_test_profile, pg_client):
    """The three cases on real rows, so the COALESCE is checked by Postgres,
    not by string matching."""
    import uuid

    from ogham.database import upsert_memory

    memory_id = str(uuid.uuid4())
    base = {"id": memory_id, "content": "x", "profile": pg_test_profile, "embedding": [0.0] * 512}

    def stored() -> float:
        row = pg_client.fetchone(
            "SELECT importance FROM memories WHERE id = %(id)s", {"id": memory_id}
        )
        return round(float(row["importance"]), 4)

    upsert_memory({**base, "importance": None, "importance_if_new": 0.6})
    assert stored() == 0.6  # new row, nothing carried: computed value
    upsert_memory({**base, "importance": None, "importance_if_new": 0.1})
    assert stored() == 0.6  # existing row, nothing carried: preserved
    upsert_memory({**base, "importance": 0.9, "importance_if_new": 0.9})
    assert stored() == 0.9  # carried: overwrites


# ── supersession on export_memories ───────────────────────────────────────


def test_export_marks_memories_a_newer_memory_contradicts():
    from ogham.export_import import _mark_superseded

    rows = [
        {"id": OLD, "created_at": "2026-08-01T00:00:00+00:00"},
        {"id": NEW, "created_at": "2026-08-02T00:00:00+00:00"},
    ]
    pairs = [{"stale_id": OLD, "newer_id": NEW, "strength": 1.0}]
    with patch("ogham.database.in_result_contradictions", return_value=pairs):
        marked = _mark_superseded(rows, "p")
    assert marked[0]["superseded_by"] == NEW
    assert "superseded_by" not in marked[1]
    assert "superseded_by" not in rows[0], "caller's dicts are not mutated"


def test_export_without_the_supersession_lookup_still_exports():
    from ogham.export_import import _mark_superseded

    rows = [{"id": OLD, "created_at": "2026-08-01T00:00:00+00:00"}]
    with patch("ogham.database.in_result_contradictions", side_effect=RuntimeError("no 047")):
        assert _mark_superseded(rows, "p") == rows


# ── TBU-277: explicit evidence_class on graph import ──────────────────────


def test_graph_import_passes_evidence_class_explicitly():
    from ogham.okf.entities import ParsedEntityConcept
    from ogham.okf.graph_import import IMPORTED_EVIDENCE_CLASS, apply_okf_graph

    graph = MagicMock()
    graph.find_entity.return_value = None
    graph.upsert_entity.side_effect = [1, 2]
    graph.list_edges.return_value = []
    concepts = [
        ParsedEntityConcept("A", "project", 1, [], [("OWNS", "b-e2")], "a-e1"),
        ParsedEntityConcept("B", "component", 2, [], [], "b-e2"),
    ]
    apply_okf_graph(concepts, "p", graph)
    assert IMPORTED_EVIDENCE_CLASS == "inferred"
    for call in graph.upsert_entity.call_args_list:
        assert call.kwargs == {"evidence_class": "inferred"}


def test_upsert_entity_requires_an_evidence_class():
    from ogham.postgres.entity_graph import PostgresEntityGraph

    graph = PostgresEntityGraph(MagicMock(), ["OWNS"])
    with pytest.raises(TypeError):
        graph.upsert_entity("A", "project")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="evidence_class"):
        graph.upsert_entity("A", "project", evidence_class="0.7")


def test_postgres_upsert_entity_inserts_the_evidence_class():
    from ogham.postgres.entity_graph import PostgresEntityGraph

    pool = MagicMock()
    conn = pool.connection.return_value.__enter__.return_value
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {"id": 7}
    graph = PostgresEntityGraph(pool, ["OWNS"])
    assert graph.upsert_entity("A", "project", evidence_class="inferred") == 7
    sql, params = cur.execute.call_args.args
    assert "evidence_class" in sql.split("ON CONFLICT")[0]
    assert "evidence_class" not in sql.split("ON CONFLICT")[1], "an upsert must never demote"
    assert params == ("A", "project", "inferred")


def test_supabase_upsert_entity_inserts_the_evidence_class():
    from ogham.supabase.entity_graph import SupabaseEntityGraph

    client = MagicMock()
    table = client.table.return_value
    lookup = table.select.return_value.eq.return_value.eq.return_value.limit.return_value
    lookup.execute.return_value.data = []
    table.insert.return_value.execute.return_value.data = [{"id": 9}]
    graph = SupabaseEntityGraph(client, allowed_predicates=["OWNS"])
    assert graph.upsert_entity("A", "project", evidence_class="inferred") == 9
    payload = table.insert.call_args.args[0]
    assert payload["evidence_class"] == "inferred"


def test_json_export_rounds_importance_and_leaves_the_rows_alone():
    import json

    from ogham.export_import import _export_json

    rows = [{"id": OLD, "content": "x", "importance": 0.30000000000000004}]
    out = json.loads(_export_json("p", rows))
    assert out["memories"][0]["importance"] == 0.3
    assert rows[0]["importance"] == 0.30000000000000004


def test_entity_aliases_are_written_unprefixed_for_obsidian():
    """Kevin, 2026-11-19: Obsidian reads `aliases` natively and is a supported
    source, so this key keeps its bare name. The validator's I203 is accepted."""
    from ogham.okf.entities import entity_to_frontmatter, frontmatter_to_entity

    fm = entity_to_frontmatter(ENTITIES[0], [], ["OpenBrain"], {})
    assert fm["aliases"] == ["OpenBrain"]
    assert "x-ogham-aliases" not in fm
    # Bundles from the first v0.20 development commit spelled it x-ogham-aliases.
    parsed = frontmatter_to_entity(
        {
            "type": "Entity",
            "x-ogham-canonical-name": "X",
            "x-ogham-entity-type": "thing",
            "x-ogham-aliases": ["Y"],
        },
        "x-e1",
    )
    assert parsed is not None and parsed.aliases == ["Y"]
