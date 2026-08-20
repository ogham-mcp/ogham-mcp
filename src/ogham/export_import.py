"""Export and import memory data."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ogham.database import (
    batch_check_duplicates,
    get_all_memories_full,
    get_profile_ttl,
    store_memories_batch,
)
from ogham.embeddings import generate_embeddings_batch

if TYPE_CHECKING:
    from ogham.entity_graph import Entity, EntityEdge

logger = logging.getLogger(__name__)


class _SkipAudit(Exception):
    """Internal: a dry run mutated nothing, so there is nothing to audit."""


def _list_all_memories(profile: str) -> list[dict[str, Any]]:
    """Fetch all memories for a profile. Extracted so tests can patch it."""
    return get_all_memories_full(profile)


def _get_producer_version() -> str:
    """Return the producer string for OKF bundle manifests."""
    try:
        import importlib.metadata

        version = importlib.metadata.version("ogham-mcp")
        return f"ogham-mcp/{version}"
    except Exception:
        # TODO: add __version__ to ogham/__init__.py in a future cleanup
        return "ogham-mcp/dev"


def _fetch_graph(
    profile: str,
) -> tuple[list["Entity"], list["EntityEdge"], dict[int, list[str]], dict[str, list[int]]]:
    """Read the whole entity graph for ``profile``, for the OKF bundle writer.

    Fails OPEN, matching ``_demote_superseded`` in service.py: an install that
    never ran the entity migrations (pre-036), or a backend with no entity-graph
    implementation at all (gateway), must still get a valid memories-only
    bundle. The graph layer is additive to the bundle -- refusing to export
    anything because a join table is missing would be the wrong trade.

    The whole fetch is one try block on purpose. These four reads are one
    consistent view of the graph, and a partial one is worse than none: edges
    without their entities are the dangling links the bundle format is built to
    make impossible.
    """
    # Imported lazily and by module, not by name: the test suite patches
    # ``ogham.database.get_entity_graph_and_vocab``, and a module-level
    # ``from ... import`` would bind the original past the patch.
    from ogham.database import get_entity_graph_and_vocab, get_memory_entities

    # Accumulate into locals and publish only on full success. Assigning the
    # four results directly would leave the earlier ones populated when a later
    # read raises, returning exactly the partial view this docstring rules out.
    #
    # That is not a rare race. `derived_from` is added to entity_edges by
    # ALTER TABLE in migration 046, and `list_edges` names it while
    # `list_entities` does not -- so on an install sitting at 041-045 the second
    # read fails *deterministically* while the first succeeds. An install at
    # 041/042 loses aliases the same way (entity_aliases arrives in 043).
    try:
        graph, _vocab = get_entity_graph_and_vocab()
        fetched_entities = graph.list_entities(profile)
        fetched_edges = graph.list_edges(profile)
        fetched_aliases = graph.list_aliases(profile)
        fetched_mem_entities = get_memory_entities(profile)
    except Exception as exc:
        logger.warning("OKF export: graph layer unavailable, exporting memories only: %s", exc)
        return [], [], {}, {}
    return fetched_entities, fetched_edges, fetched_aliases, fetched_mem_entities


def export_memories(profile: str, format: str = "json", *, include_viewer: bool = True) -> str:
    """Export all memories in a profile to a string or bundle path.

    For format='okf', writes an OKF v0.1 bundle directory to cwd and returns
    the directory path as a string. The bundle gets a self-contained viewer.html
    by default; pass include_viewer=False to skip it.
    For 'json'/'markdown', returns the data inline as a string and ignores
    include_viewer.
    """
    memories = _list_all_memories(profile)

    if format == "okf":
        # Imported from the defining module, not the ``ogham.okf`` package
        # re-export: patching the package attribute would leave this binding
        # untouched, so tests could not observe what the bundle writer is handed.
        from ogham.okf.bundle import export_okf_bundle

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bundle_dir = Path.cwd() / f"ogham-okf-{profile}-{stamp}"
        manifest = {
            "producer": _get_producer_version(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
        }
        entities, edges, aliases, mem_entities = _fetch_graph(profile)
        export_okf_bundle(
            memories,
            bundle_dir,
            manifest,
            include_viewer=include_viewer,
            entities=entities,
            edges=edges,
            aliases=aliases,
            memory_entities=mem_entities,
        )
        return str(bundle_dir)

    if format == "markdown":
        return _export_markdown(profile, memories)
    return _export_json(profile, memories)


def _export_json(profile: str, memories: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "profile": profile,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(memories),
            "memories": memories,
        },
        indent=2,
        default=str,
    )


def _export_markdown(profile: str, memories: list[dict[str, Any]]) -> str:
    lines = [
        "# Ogham Memory Export",
        "",
        f"**Profile:** {profile}",
        f"**Count:** {len(memories)}",
        f"**Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
    ]
    for mem in memories:
        lines.append(f"## {mem.get('created_at', 'unknown')[:10]}")
        tags = mem.get("tags", [])
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        source = mem.get("source")
        if source:
            lines.append(f"**Source:** {source}")
        lines.append("")
        lines.append(mem["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _build_row(
    mem: dict[str, Any],
    embedding: list[float],
    profile: str,
    expires_at: str | None,
) -> dict[str, Any]:
    """Build a row dict ready for database insertion.

    NOTE: deliberately does NOT carry ``mem["created_at"]`` into the row.
    The ``memories`` table has ``DEFAULT now()`` on ``created_at``, which
    means every imported memory is timestamped at INGEST time, not at the
    historical date the source records (e.g. Claude.ai conversation date).
    Compaction logic in ``ogham.compression.get_compression_target`` keys
    on ``created_at``; passing through the original date here would cause
    backdated imports to compact immediately on insert (cdeust/Cortex hit
    this bug 2026-05; their fix was a ``created_at -> ingested_at`` rename
    that recovered MRR_with_consolidation 0.222 -> 0.8264). Importers
    should put historical dates in ``metadata`` instead (see
    ``claude_ai_import.py`` -> ``metadata.claude_created_at``).
    """
    row = {
        "content": mem["content"],
        "embedding": str(embedding),
        "profile": profile,
        "metadata": mem.get("metadata") or {},
        "source": mem.get("source"),
        "tags": mem.get("tags") or [],
    }
    if expires_at is not None:
        row["expires_at"] = expires_at
    return row


def _upsert_memory(memory: dict[str, Any]) -> None:
    """Insert or update a memory by id. Used by OKF imports for round-trip.

    Calls backend.upsert_memory which performs ON CONFLICT (id) DO UPDATE on
    Postgres/Supabase, or a GET-then-PUT on the gateway backend.
    NOTE: the caller is responsible for generating and embedding the content
    BEFORE calling this; the memory dict must contain an ``embedding`` key.
    """
    from ogham.database import upsert_memory as _db_upsert

    _db_upsert(memory)


def _looks_like_okf_bundle_dir(data: str) -> bool:
    # Path.is_dir() raises OSError(36, "File name too long") on Linux when any
    # path component exceeds NAME_MAX (255 bytes) -- which happens whenever
    # `data` is a JSON payload mistakenly passed where a path is expected.
    # macOS silently returns False, so this bug only surfaces on Linux CI / prod.
    try:
        return Path(data).is_dir()
    except OSError:
        return False


def import_memories(
    data: str,
    profile: str,
    dedup_threshold: float = 0.0,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_embed_progress: Callable[[int, int], None] | None = None,
    import_graph: bool = False,
    graph_dry_run: bool = False,
) -> dict[str, Any]:
    """Import memories from a JSON string or an OKF bundle directory path.

    Shape detection: if ``data`` is a string path to an existing directory,
    it is treated as an OKF v0.1 bundle. Memories with an ``id`` in frontmatter
    are upserted (ON CONFLICT (id) DO UPDATE). Memories without an ``id`` are
    inserted as new (mint a new UUID via the standard insert path).

    The existing JSON path (``data`` is a JSON string) keeps working exactly
    as v0.9.1 ships -- issue #20 fix stays valid for all existing users.

    The bundle's ``entities/`` layer is read only when ``import_graph=True``.
    It defaults OFF because ``entities`` has no profile column -- it is global,
    scoped only through ``memory_entities`` and ``entity_edges`` -- so importing
    a graph mutates rows every profile reads. Profiles are a convenience
    namespace rather than a trust boundary here (decided 2026-08-20), which
    makes that acceptable when asked for and surprising when not.

    Scope: this imports YOUR OWN bundles. Importing a third-party bundle is not
    supported -- see the caps in ``okf/bundle.py``.

    Args:
        on_progress: Optional callback(imported, skipped, total) called after each memory.
        on_embed_progress: Optional callback(embedded, total) called after each batch.
        import_graph: Also read ``entities/`` back into the entity graph.
        graph_dry_run: With ``import_graph``, report what the graph import
            would do and write nothing. Memories are still imported.
    """
    # ── OKF bundle path ────────────────────────────────────────────────
    if isinstance(data, str) and _looks_like_okf_bundle_dir(data):
        # Pre-flight: confirm this is actually an OKF bundle (has index.md declaring
        # okf_version), not just any directory the user pointed at by accident.
        bundle_dir = Path(data)
        index_path = bundle_dir / "index.md"
        if not index_path.exists():
            raise ValueError(
                f"{data} is a directory but doesn't look like an OKF bundle "
                f"(missing index.md with okf_version declaration)"
            )
        # Quick frontmatter check -- read just enough to confirm okf_version is declared.
        from ogham.okf.serialization import read_concept as _read_concept

        try:
            fm, _ = _read_concept(index_path)
        except ValueError as e:
            raise ValueError(f"{data}/index.md is not a valid OKF bundle root: {e}") from e
        if "okf_version" not in fm:
            raise ValueError(
                f"{data}/index.md exists but does not declare okf_version -- "
                f"not a recognizable OKF bundle"
            )

        from ogham.okf import import_okf_bundle

        okf_memories, stats = import_okf_bundle(bundle_dir)

        # Split: memories with id → upsert; memories without id → regular insert.
        with_id: list[dict[str, Any]] = []
        without_id: list[dict[str, Any]] = []
        for mem in okf_memories:
            (with_id if mem.get("id") is not None else without_id).append(mem)

        # Upsert memories that carry their UUID.
        upserted = 0
        if with_id:
            all_texts_upsert = [m["content"] for m in with_id]
            embeddings_upsert = generate_embeddings_batch(
                all_texts_upsert, on_progress=on_embed_progress
            )
            for mem, embedding in zip(with_id, embeddings_upsert):
                mem_with_embedding = {**mem, "embedding": embedding, "profile": profile}
                _upsert_memory(mem_with_embedding)
                upserted += 1

        # Insert memories that have no id (treat as new).
        inserted = 0
        if without_id:
            import uuid

            ttl_days = get_profile_ttl(profile)
            expires_at = None
            if ttl_days is not None:
                expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
            all_texts_insert = [m["content"] for m in without_id]
            embeddings_insert = generate_embeddings_batch(all_texts_insert)
            rows_to_insert = [
                _build_row(
                    {**m, "id": str(uuid.uuid4())},
                    emb,
                    profile,
                    expires_at,
                )
                for m, emb in zip(without_id, embeddings_insert)
            ]
            store_memories_batch(rows_to_insert)
            inserted = len(rows_to_insert)

        result: dict[str, Any] = {
            "status": "complete",
            "profile": profile,
            "imported": upserted + inserted,
            "skipped": 0,
            "total": stats["total"],
            "missing_id_count": stats["missing_id_count"],
            "skipped_count": stats["skipped_count"],
        }

        if import_graph:
            from ogham.database import emit_audit_event, get_entity_graph_and_vocab
            from ogham.okf.bundle import import_okf_graph
            from ogham.okf.graph_import import apply_okf_graph

            concepts, graph_stats = import_okf_graph(bundle_dir)
            if concepts:
                graph, _vocab = get_entity_graph_and_vocab()
                applied = apply_okf_graph(concepts, profile, graph, dry_run=graph_dry_run)
                graph_stats.update(applied)
                # Audit: without this, "the import did something to my graph and
                # I cannot tell what" is not diagnosable even in principle.
                try:
                    if graph_dry_run:
                        raise _SkipAudit  # a dry run changed nothing to audit
                    emit_audit_event(
                        profile=profile,
                        operation="import_okf_graph",
                        outcome="success",
                        metadata=graph_stats,
                    )
                except _SkipAudit:
                    pass
                except Exception as exc:
                    logger.debug("audit event for graph import skipped: %s", exc)
            result["graph"] = graph_stats

        return result

    # ── JSON string path (v0.9.1 behaviour, unchanged) ─────────────────
    parsed = json.loads(data)
    memories = parsed.get("memories", [])
    total = len(memories)

    ttl_days = get_profile_ttl(profile)
    expires_at = None
    if ttl_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    # Phase 1: Batch embed all memories upfront
    all_texts = [mem["content"] for mem in memories]
    embeddings = generate_embeddings_batch(all_texts, on_progress=on_embed_progress)

    # Phase 2: Parallel batch dedup (concurrent RPC batches to use multiple DB cores)
    skipped = 0
    to_insert: list[dict[str, Any]] = []

    if dedup_threshold > 0:
        dedup_batch_size = 50
        is_dup = [False] * total

        # Build batch ranges
        batch_ranges = [
            (start, min(start + dedup_batch_size, total))
            for start in range(0, total, dedup_batch_size)
        ]

        def _check_batch(batch_range: tuple[int, int]) -> tuple[int, int, list[bool]]:
            start, end = batch_range
            batch_embeddings = embeddings[start:end]
            results = batch_check_duplicates(
                query_embeddings=batch_embeddings,
                profile=profile,
                threshold=dedup_threshold,
            )
            return start, end, results

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_check_batch, br) for br in batch_ranges]
            completed = 0
            for future in futures:
                start, end, batch_results = future.result()
                for i, dup in enumerate(batch_results):
                    is_dup[start + i] = dup
                    if dup:
                        skipped += 1
                completed += end - start
                if on_progress:
                    on_progress(completed - skipped, skipped, total)

        for i, (mem, embedding) in enumerate(zip(memories, embeddings)):
            if not is_dup[i]:
                to_insert.append(_build_row(mem, embedding, profile, expires_at))
    else:
        for mem, embedding in zip(memories, embeddings):
            to_insert.append(_build_row(mem, embedding, profile, expires_at))
        if on_progress:
            on_progress(len(to_insert), 0, total)

    # Phase 3: Batch insert non-duplicates
    batch_size = 100
    for start in range(0, len(to_insert), batch_size):
        batch = to_insert[start : start + batch_size]
        store_memories_batch(batch)

    imported = len(to_insert)

    return {
        "status": "complete",
        "profile": profile,
        "imported": imported,
        "skipped": skipped,
        "total": total,
    }
