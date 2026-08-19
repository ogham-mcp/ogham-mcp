from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ogham.okf.entities import ENTITIES_DIR
from ogham.okf.serialization import read_concept

_RESERVED_FILENAMES = {"index.md", "log.md"}
_LINK_RE = re.compile(r"\]\(([^)\s]+\.md)(?:#[A-Za-z0-9_\-]*)?\)")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://", "//")


@dataclass
class ViewerConcept:
    id: str
    type: str
    title: str
    tags: list[str]
    body: str
    links_to: list[str] = field(default_factory=list)


def extract_links(body: str, doc_path: Path, bundle_dir: Path) -> list[str]:
    out: list[str] = []
    bundle_root = bundle_dir.resolve()
    doc_dir = doc_path.parent.resolve()
    for raw in _LINK_RE.findall(body):
        if raw.startswith(_EXTERNAL_PREFIXES):
            continue
        target_fs = (doc_dir / raw).resolve()
        try:
            rel = target_fs.relative_to(bundle_root)
        except ValueError:
            continue
        out.append(str(rel).replace("\\", "/").removesuffix(".md"))
    return out


def parse_bundle(bundle_dir: Path) -> list[ViewerConcept]:
    bundle_dir = Path(bundle_dir).resolve()
    entity_dir = bundle_dir / ENTITIES_DIR
    concepts: list[ViewerConcept] = []
    for md_path in sorted(bundle_dir.rglob("*.md")):
        if md_path.name in _RESERVED_FILENAMES:
            continue
        # The graph layer is not viewer content. This viewer derives edges from
        # markdown links in the BODY (extract_links), while entity concepts hold
        # their triples in frontmatter wiki links and have empty bodies by
        # design -- so ingesting them yields one disconnected, uncoloured node
        # per entity rather than the graph they describe.
        #
        # Directory containment, not a name check, so nested layouts are covered
        # too. Mirrors the skip in bundle.py::import_okf_bundle; both exist
        # because a bundle-wide rglob("*.md") sweeps up the graph layer.
        #
        # Rendering the typed graph properly means teaching this module the
        # frontmatter wiki-link grammar (okf.entities.resolve_wiki_link) and
        # giving "Entity" a palette colour. That is a feature, and it belongs
        # with the release that makes the graph round-trip.
        if entity_dir in md_path.parents:
            continue
        fm, body = read_concept(md_path)
        rel = md_path.relative_to(bundle_dir)
        concept_id = str(rel).replace("\\", "/").removesuffix(".md")
        concepts.append(
            ViewerConcept(
                id=concept_id,
                type=str(fm.get("type", "Memory")),
                title=str(fm.get("title") or concept_id.rsplit("/", 1)[-1]),
                tags=list(fm.get("tags") or []),
                body=body,
                links_to=extract_links(body, md_path, bundle_dir),
            )
        )
    return concepts
