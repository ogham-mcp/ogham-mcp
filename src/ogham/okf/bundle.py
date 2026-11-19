"""OKF bundle (directory) read/write orchestration. OKF v0.3."""

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ogham.entity_graph import Entity, EntityEdge
from ogham.okf.concept import (
    MEMORIES_DIR,
    frontmatter_to_memory,
    memory_to_body,
    memory_to_frontmatter,
)
from ogham.okf.context import CONTEXT_FILENAME, build_context
from ogham.okf.entities import (
    ENTITIES_DIR,
    ParsedEntityConcept,
    entity_to_body,
    entity_to_frontmatter,
    frontmatter_to_entity,
    make_entity_filename,
)
from ogham.okf.identity import make_filename
from ogham.okf.links import x_key
from ogham.okf.serialization import read_concept, write_concept

logger = logging.getLogger(__name__)

#: Declared in the bundle-root index.md. A quoted string per OKF §8.1 -- yaml
#: dumps a Python str that looks like a number with quotes.
_OKF_VERSION = "0.3"

#: Version of Ogham's own graph layer -- the `entities/` directory and the
#: frontmatter-triple shape inside it. `okf_version` covers the OKF container
#: and says nothing about this (D11). 1 = v0.18 shape (bare keys, wiki-link
#: values). 2 = v0.20 shape (`x-ogham-*` keys, relative-path values, a body
#: Relations section). The importer reads both.
_GRAPH_VERSION = 2

#: Index key for the graph version, and the v0.18 spelling still read.
GRAPH_VERSION_KEYS = (x_key("graph-version"), "ogham_graph_version")

_DEFAULT_TITLE = "Ogham memory export"
_DEFAULT_DESCRIPTION = (
    "Memories exported from an Ogham profile, one concept per memory. "
    "Use it to read or move that profile's knowledge."
)

#: Index keys Ogham owns outright and always rewrites.
_INDEX_SPEC_KEYS = ("okf_version", "title", "description", "license")


#: Governing @base for the bundle's JSON-LD context when the caller names none.
#: Bundles are portable, so the IRIs they mint must not depend on where the
#: directory happens to sit on disk.
_DEFAULT_BASE = "https://ogham-mcp.dev/bundle/"


def build_index_frontmatter(
    manifest: dict,
    *,
    title: str | None = None,
    description: str | None = None,
    license: str | None = None,
    existing: dict | None = None,
) -> dict:
    """Compose the bundle-root index.md frontmatter (OKF §8.1).

    Precedence for `title`, `description` and `license`: an explicit argument,
    then the value already in the bundle's index.md, then (title and
    description only) a default. `license` is never invented -- absent unless a
    caller or the existing file supplies one.

    §8.1: a producer that regenerates index.md MUST keep the existing
    frontmatter, so every other key from ``existing`` survives. ``manifest``
    keys are Ogham's own bookkeeping and overwrite stale values of the same
    name; `okf_version` is always the version this writer produces.
    """
    existing = dict(existing or {})
    fm: dict = {
        "okf_version": _OKF_VERSION,
        "title": title or existing.get("title") or _DEFAULT_TITLE,
        "description": description or existing.get("description") or _DEFAULT_DESCRIPTION,
    }
    chosen_license = license or existing.get("license")
    if chosen_license:
        fm["license"] = chosen_license
    for key, value in existing.items():
        if key not in _INDEX_SPEC_KEYS and key not in manifest:
            fm[key] = value
    fm.update(manifest)
    return fm


def write_index(
    bundle_dir: Path,
    manifest: dict,
    *,
    title: str | None = None,
    description: str | None = None,
    license: str | None = None,
    existing: dict | None = None,
) -> None:
    """Write the bundle-root index.md with the okf_version declaration.

    The bundle-root index.md is the only index.md that may carry frontmatter
    (OKF §8), and is where the OKF version is declared (§8.1). See
    ``build_index_frontmatter`` for how existing frontmatter is kept.
    """
    frontmatter = build_index_frontmatter(
        manifest, title=title, description=description, license=license, existing=existing
    )
    body = (
        "# Memories\n\n"
        "This bundle was produced by Ogham. "
        "See individual concept files in `memories/`.\n"
    )
    write_concept(bundle_dir / "index.md", frontmatter, body)


def _read_existing_index(bundle_dir: Path) -> dict | None:
    """Frontmatter of the index.md a re-export is about to replace, if any."""
    index_path = bundle_dir / "index.md"
    if not index_path.is_file():
        return None
    try:
        fm, _ = read_concept(index_path)
    except (ValueError, OSError) as exc:
        logger.warning("OKF export: existing %s unreadable, not preserved: %s", index_path, exc)
        return None
    return fm


def filter_expired(memories: list[dict]) -> list[dict]:
    """Drop memories whose expires_at is in the past.

    Memories with no expires_at or None are kept (no expiration). Default
    behaviour on export per the v0.15 design decision.
    """
    now = datetime.now(timezone.utc)
    kept = []
    for m in memories:
        expires = m.get("expires_at")
        if not expires:
            kept.append(m)
            continue
        try:
            # Postgres backend (psycopg) returns expires_at as a real datetime
            # object; the Supabase REST backend returns an ISO string (#TBU-162
            # audit -- same root cause as the hybrid_search datetime crash).
            if isinstance(expires, datetime):
                ts = expires
            else:
                ts = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        except (ValueError, AttributeError, TypeError):
            kept.append(m)  # unparseable = keep (safe default)
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts > now:
            kept.append(m)
    return kept


def export_okf_bundle(
    memories: list[dict[str, Any]],
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    include_viewer: bool = False,
    entities: list[Entity] | None = None,
    edges: list[EntityEdge] | None = None,
    aliases: dict[int, list[str]] | None = None,
    memory_entities: dict[str, list[int]] | None = None,
    base: str | None = None,
    title: str | None = None,
    description: str | None = None,
    license: str | None = None,
) -> None:
    """Export a list of memories to an OKF v0.3 bundle directory.

    Atomicity guarantee:
    - Fresh targets (bundle_dir absent): os.rename(staging, bundle_dir) is a
      single syscall -- either the target exists with full contents or it does not.
    - Existing targets: shutil.rmtree(bundle_dir) then shutil.move(staging, bundle_dir).
      Between those two syscalls a SIGKILL leaves the old bundle gone and the new
      bundle still in the temp staging directory. The TemporaryDirectory context manager
      cleans up staging on normal exit but NOT after SIGKILL, so the new bundle may
      be orphaned in a `.okf-tmp-*` sibling. True crash-safety for in-place updates
      would require a backup-rename pattern; this is not needed for v1 self-hosted usage.

    Pre-existing target is replaced, not merged.
    Filters expired memories by default per v0.15 design decision.

    If include_viewer is True, also writes a self-contained viewer.html at the
    bundle root after the atomic rename completes. The viewer is regenerated
    fresh on every export -- it never partially updates.

    The graph layer (``entities``/``edges``/``aliases``/``memory_entities``) is
    additive and every parameter is keyword-only with a None default: called
    with memories alone, this writes no `entities/`, no `context.jsonld` and no
    graph version. tests/test_okf_bundle_entities.py asserts that on file
    contents.

    ``base`` is the governing @base for context.jsonld; the default is used when
    the caller has no better IRI, which is the normal case for a portable
    bundle.

    ``title``/``description``/``license`` go into the bundle-root index.md
    (OKF §8.1). An existing index.md at ``bundle_dir`` has its frontmatter
    kept, per §8.1; see ``build_index_frontmatter``.

    A memory dict with a truthy ``superseded_by`` (a memory id) is exported as
    ``status: deprecated``, with a ``superseded_by`` path when that memory is
    in the bundle too (OKF §5.4).
    """
    bundle_dir = Path(bundle_dir)
    fresh = filter_expired(memories)
    existing_index = _read_existing_index(bundle_dir)

    entity_list = list(entities or [])
    # One table, built before any file is written, so memory concepts and entity
    # concepts resolve their links against identical paths. Values are
    # bundle-relative and carry no `.md`.
    path_by_id = {e.id: f"{ENTITIES_DIR}/{make_entity_filename(e)[:-3]}" for e in entity_list}
    name_by_id = {e.id: e.canonical_name for e in entity_list}
    # Same idea for memories: a superseded memory links to its replacement's
    # file, and only a replacement that is actually being written may be named.
    memory_path_by_id = {str(m["id"]): f"{MEMORIES_DIR}/{make_filename(m)}" for m in fresh}
    # Edges are emitted on the subject only (D4), so group them once here rather
    # than rescanning the full edge list per entity.
    edges_by_subject: dict[int, list[EntityEdge]] = {}
    for edge in edges or []:
        edges_by_subject.setdefault(edge.subject_id, []).append(edge)

    if entity_list:
        # Copied, never mutated: the manifest belongs to the caller, and
        # export_memories builds one and reuses it. Absent on a memories-only
        # bundle, which has no graph layer to version.
        manifest = {**manifest, GRAPH_VERSION_KEYS[0]: _GRAPH_VERSION}

    # Use a sibling temp dir so the final rename is on the same filesystem.
    parent = bundle_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".okf-tmp-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        if existing_index is not None:
            # A re-export replaces a bundle that may carry an older Ogham's
            # graph-version key. Keeping both would declare two versions.
            existing_index = {
                k: v for k, v in existing_index.items() if k not in GRAPH_VERSION_KEYS
            }
        write_index(
            staging,
            manifest,
            title=title,
            description=description,
            license=license,
            existing=existing_index,
        )
        memories_dir = staging / MEMORIES_DIR
        memories_dir.mkdir()
        for memory in fresh:
            # An entity id with no exported concept would produce a link to a
            # file that is not there, so filter against the same path table the
            # entity concepts use rather than assuming the two agree.
            linked = [
                eid
                for eid in (memory_entities or {}).get(str(memory["id"]), [])
                if eid in path_by_id
            ]
            superseded_id = memory.get("superseded_by")
            fm = memory_to_frontmatter(
                memory,
                entity_paths=[path_by_id[eid] for eid in linked],
                superseded_by_path=(
                    memory_path_by_id.get(str(superseded_id)) if superseded_id else None
                ),
            )
            body = memory_to_body(
                memory.get("content") or "",
                [(name_by_id[eid], path_by_id[eid]) for eid in linked],
            )
            write_concept(memories_dir / make_filename(memory), fm, body)
        if entity_list:
            entities_dir = staging / ENTITIES_DIR
            entities_dir.mkdir()
            for entity in entity_list:
                subject_edges = edges_by_subject.get(entity.id, [])
                fm = entity_to_frontmatter(
                    entity,
                    subject_edges,
                    (aliases or {}).get(entity.id, []),
                    path_by_id,
                )
                # An entity has no prose, so its body is only the marked
                # Relations section -- the edges as OKF v0.3 links -- or empty.
                body = entity_to_body(entity, subject_edges, path_by_id, name_by_id)
                write_concept(entities_dir / make_entity_filename(entity), fm, body)
            # Written only alongside entities. A memories-only bundle has no
            # predicates to give terms to, and stays byte-identical to what
            # every prior release produced.
            (staging / CONTEXT_FILENAME).write_text(
                json.dumps(build_context(base or _DEFAULT_BASE), indent=2) + "\n",
                encoding="utf-8",
            )
        # Atomic replace
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        shutil.move(str(staging), str(bundle_dir))

    if include_viewer:
        from ogham.okf.viewer import build_viewer

        build_viewer(bundle_dir)


_RESERVED_FILENAMES = {"index.md", "log.md"}


def import_okf_bundle(bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read an OKF bundle directory into a list of memory dicts + stats.

    Stats includes `total` (concepts found) and `missing_id_count` (concepts
    that arrived without our id extension -- they will become NEW memories,
    not upserts, when the caller writes them).

    Reserved filenames (index.md, log.md per spec §3.1) are skipped at every
    directory level, and so are dot-paths (§3.2 rule 4: not part of the bundle,
    which matters for a bundle distributed as a git repository).

    A `.md` file whose frontmatter has no non-empty `type` is not a concept
    (OKF §3.2 rule 2) and is skipped, counted in `not_concept_count`. This is
    what keeps a README.md, or a directory of some other tool's notes, from
    becoming memories now that a missing index.md no longer stops the import
    (§11.2).

    The `entities/` graph layer is skipped here; ``import_okf_graph`` reads it.
    A skipped entity is not a dropped memory, so it is not counted in
    `skipped_count`.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"{bundle_dir} is not a directory")

    entity_dir = bundle_dir / ENTITIES_DIR
    memories: list[dict] = []
    missing_id_count = 0
    skipped_count = 0
    not_concept_count = 0
    for md_path in sorted(bundle_dir.rglob("*.md")):
        if md_path.name in _RESERVED_FILENAMES:
            continue
        if any(part.startswith(".") for part in md_path.relative_to(bundle_dir).parts):
            continue
        # Containment, not a filename or single-parent check: rglob descends, so
        # a nested layout under entities/ has to be excluded too. Placed before
        # read_concept so the skip can never be mistaken for a parse failure.
        if entity_dir in md_path.parents:
            continue
        try:
            fm, body = read_concept(md_path)
        except ValueError:
            # Malformed concept -- skip but surface the drop in stats so the
            # import tool can warn the operator.
            skipped_count += 1
            continue
        concept_type = fm.get("type")
        if not isinstance(concept_type, str) or not concept_type.strip():
            not_concept_count += 1
            continue
        memory = frontmatter_to_memory(fm, body)
        if memory["id"] is None:
            missing_id_count += 1
        memories.append(memory)

    stats = {
        "total": len(memories),
        "missing_id_count": missing_id_count,
        "skipped_count": skipped_count,
        "not_concept_count": not_concept_count,
    }
    return memories, stats


#: Bounds for graph import. Kevin's ruling 2026-08-20: importing a bundle from
#: someone else's install is NOT supported, so these are corruption guards, not
#: an adversarial threat model. They exist so a malformed bundle fails with a
#: number instead of hanging.
MAX_ENTITY_CONCEPTS = 50_000
MAX_ENTITY_FILE_BYTES = 1_000_000


def import_okf_graph(bundle_dir: Path) -> tuple[list[ParsedEntityConcept], dict[str, Any]]:
    """Read a bundle's ``entities/`` layer into parsed concepts + stats.

    Reads both graph-layer shapes: version 1 (``ogham_graph_version: 1``, bare
    keys, wiki-link values) and version 2 (``x-ogham-graph-version: 2``,
    ``x-ogham-*`` keys, relative paths). IO and parsing only -- resolving
    concepts to local ids and writing them is the caller's job, exactly as
    ``import_okf_bundle`` returns memory dicts rather than storing them.

    Returns concepts in filename order so a run is reproducible.

    **Scope, decided 2026-08-20:** this imports YOUR OWN bundles. Profiles are a
    convenience namespace here rather than a trust boundary, so writing shared
    ``entities`` rows is accepted; and bundles are not treated as untrusted
    input. Importing a third-party bundle is not supported -- see the caps above.

    Stats: ``total`` concepts parsed, ``skipped_count`` files that were not
    valid entity concepts, ``edge_count`` edges found, ``oversize_count`` files
    rejected by the size cap.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"{bundle_dir} is not a directory")

    entity_dir = bundle_dir / ENTITIES_DIR
    concepts: list[ParsedEntityConcept] = []
    skipped_count = 0
    oversize_count = 0
    if not entity_dir.is_dir():
        return concepts, {
            "total": 0,
            "skipped_count": 0,
            "edge_count": 0,
            "oversize_count": 0,
            "graph_present": False,
        }

    for md_path in sorted(entity_dir.rglob("*.md")):
        if md_path.name in _RESERVED_FILENAMES:
            continue
        if len(concepts) >= MAX_ENTITY_CONCEPTS:
            break
        try:
            if md_path.stat().st_size > MAX_ENTITY_FILE_BYTES:
                oversize_count += 1
                continue
            fm, _body = read_concept(md_path)
        except (ValueError, OSError):
            skipped_count += 1
            continue
        concept = frontmatter_to_entity(fm, md_path.stem)
        if concept is None:
            # A concept in entities/ that is not an Entity is a malformed
            # bundle, not a memory -- surface it rather than silently widening
            # what import_okf_bundle returns.
            skipped_count += 1
            continue
        concepts.append(concept)

    return concepts, {
        "total": len(concepts),
        "skipped_count": skipped_count,
        "edge_count": sum(len(c.edges) for c in concepts),
        "oversize_count": oversize_count,
        "graph_present": True,
    }
