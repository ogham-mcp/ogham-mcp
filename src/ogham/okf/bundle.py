"""OKF bundle (directory) read/write orchestration."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ogham.entity_graph import Entity, EntityEdge
from ogham.okf.concept import frontmatter_to_memory, memory_to_frontmatter
from ogham.okf.context import CONTEXT_FILENAME, build_context
from ogham.okf.entities import ENTITIES_DIR, entity_to_frontmatter, make_entity_filename
from ogham.okf.identity import make_filename
from ogham.okf.serialization import read_concept, write_concept

_OKF_VERSION = "0.1"

#: Version of Ogham's own graph layer -- the `entities/` directory and the
#: frontmatter-triple shape inside it. `okf_version` covers the OKF container
#: and says nothing about this (D11). v0.18 mints the format, so declaring it
#: costs nothing here and gives a later importer something to branch on.
_GRAPH_VERSION = 1


#: Governing @base for the bundle's JSON-LD context when the caller names none.
#: Bundles are portable, so the IRIs they mint must not depend on where the
#: directory happens to sit on disk.
_DEFAULT_BASE = "https://ogham-mcp.dev/bundle/"


def write_index(bundle_dir: Path, manifest: dict) -> None:
    """Write the bundle-root index.md with okf_version declaration.

    Per OKF spec §11 + §6: the bundle-root index.md is the ONLY index.md where
    frontmatter is permitted, and is where the supported OKF version is declared.
    """
    frontmatter = {"okf_version": _OKF_VERSION, **manifest}
    body = (
        "# Memories\n\n"
        "This bundle was produced by Ogham. "
        "See individual concept files in `memories/`.\n"
    )
    write_concept(bundle_dir / "index.md", frontmatter, body)


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
) -> None:
    """Export a list of memories to an OKF v0.1 bundle directory.

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
    with memories alone, this writes exactly the bundle it wrote before the
    graph layer existed, byte for byte. That is the compatibility guarantee, and
    tests/test_okf_bundle_entities.py asserts it on file contents.

    ``base`` is the governing @base for context.jsonld; the default is used when
    the caller has no better IRI, which is the normal case for a portable
    bundle.

    Export only -- nothing here reads a bundle back into the graph. Import of
    the entities layer is a later release (D10); v0.18's only import-side
    obligation is not to mistake entity concepts for memories.
    """
    bundle_dir = Path(bundle_dir)
    fresh = filter_expired(memories)

    entity_list = list(entities or [])
    # One table, built before any file is written, so memory concepts and entity
    # concepts resolve their links against identical paths. Values are
    # bundle-relative and carry no `.md`, matching the wiki-link grammar
    # (§4.4.1) where the extension is never part of the target.
    path_by_id = {e.id: f"{ENTITIES_DIR}/{make_entity_filename(e)[:-3]}" for e in entity_list}
    # Edges are emitted on the subject only (D4), so group them once here rather
    # than rescanning the full edge list per entity.
    edges_by_subject: dict[int, list[EntityEdge]] = {}
    for edge in edges or []:
        edges_by_subject.setdefault(edge.subject_id, []).append(edge)

    if entity_list:
        # Copied, never mutated: the manifest belongs to the caller, and
        # export_memories builds one and reuses it. Absent on a memories-only
        # bundle, which has no graph layer to version.
        manifest = {**manifest, "ogham_graph_version": _GRAPH_VERSION}

    # Use a sibling temp dir so the final rename is on the same filesystem.
    parent = bundle_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".okf-tmp-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        write_index(staging, manifest)
        memories_dir = staging / "memories"
        memories_dir.mkdir()
        for memory in fresh:
            # An entity id with no exported concept would produce a link to a
            # file that is not there, so filter against the same path table the
            # entity concepts use rather than assuming the two agree.
            linked = (memory_entities or {}).get(str(memory["id"]), [])
            fm = memory_to_frontmatter(
                memory,
                entity_paths=[path_by_id[eid] for eid in linked if eid in path_by_id],
            )
            body = memory.get("content") or ""
            filename = make_filename(memory)
            write_concept(memories_dir / filename, fm, body)
        if entity_list:
            entities_dir = staging / ENTITIES_DIR
            entities_dir.mkdir()
            for entity in entity_list:
                fm = entity_to_frontmatter(
                    entity,
                    edges_by_subject.get(entity.id, []),
                    (aliases or {}).get(entity.id, []),
                    path_by_id,
                )
                # Entity bodies are empty: SPEC §5.3 reserves the body for prose
                # and entities have none.
                write_concept(entities_dir / make_entity_filename(entity), fm, "")
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
    directory level.

    The `entities/` graph layer is deliberately skipped, not consumed. Entity
    concepts are valid OKF concepts, so nothing here would refuse them -- they
    would simply arrive as memories with no id and an empty body. Reading them
    back into the entity graph is a later release's job (D10): it means mutating
    the globally-scoped `entities` table from a bundle that may not be ours, and
    that trust boundary is not settled. Skipping is not an oversight, and a
    skipped entity is not a dropped memory, so it is not counted in
    `skipped_count`.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"{bundle_dir} is not a directory")

    entity_dir = bundle_dir / ENTITIES_DIR
    memories: list[dict] = []
    missing_id_count = 0
    skipped_count = 0
    for md_path in sorted(bundle_dir.rglob("*.md")):
        if md_path.name in _RESERVED_FILENAMES:
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
        memory = frontmatter_to_memory(fm, body)
        if memory["id"] is None:
            missing_id_count += 1
        memories.append(memory)

    stats = {
        "total": len(memories),
        "missing_id_count": missing_id_count,
        "skipped_count": skipped_count,
    }
    return memories, stats
