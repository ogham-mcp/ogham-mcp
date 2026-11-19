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


def _mark_superseded(memories: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    """Copy ``memories``, setting ``superseded_by`` on each one a newer memory contradicts.

    There is no ``superseded_by`` column: supersession is derived from
    ``contradicts`` edges, oriented stale -> newer by created_at (migration 047,
    TBU-207), exactly as search does in ``service._demote_superseded``. The same
    ``find_superseded`` rule picks the newest corrector when there are several.

    Fails OPEN, like ``_fetch_graph``: an install without 047, or a backend that
    cannot express the lookup (Supabase returns [] here), still exports, just
    without lifecycle marks. Never mutates the caller's dicts.
    """
    from ogham.database import in_result_contradictions
    from ogham.supersession import find_superseded

    if not memories:
        return memories
    try:
        dates = {str(m["id"]): m.get("created_at") for m in memories}
        pairs = [
            {
                "in_result_id": edge["stale_id"],
                "other_id": edge["newer_id"],
                "other_created_at": dates.get(str(edge["newer_id"])),
                "strength": edge.get("strength"),
            }
            for edge in in_result_contradictions(profile, list(dates))
        ]
        superseded = find_superseded(memories, pairs)
    except Exception as exc:  # noqa: BLE001 -- lifecycle marks are additive
        logger.warning("OKF export: supersession lookup unavailable, no status marks: %s", exc)
        return memories
    return [
        {**m, "superseded_by": superseded[str(m["id"])]["superseded_by"]}
        if str(m["id"]) in superseded
        else m
        for m in memories
    ]


def export_memories(profile: str, format: str = "json", *, include_viewer: bool = True) -> str:
    """Export all memories in a profile to a string or bundle path.

    For format='okf', writes an OKF v0.3 bundle directory to cwd and returns
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
            _mark_superseded(memories, profile),
            bundle_dir,
            manifest,
            include_viewer=include_viewer,
            entities=entities,
            edges=edges,
            aliases=aliases,
            memory_entities=mem_entities,
            title=f"Ogham memories: {profile}",
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
            "memories": [_round_importance(m) for m in memories],
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
        "importance": _resolve_importance(mem),
    }
    if expires_at is not None:
        row["expires_at"] = expires_at
    return row


def _carried_importance(mem: dict[str, Any]) -> float | None:
    """The importance the record carries, if it is usable: a number in [0, 1].

    The record is untrusted input, so anything else counts as not carried.
    """
    value = mem.get("importance")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0:
        return float(value)
    return None


def _resolve_importance(mem: dict[str, Any]) -> float:
    """Importance for a NEW memory created by import (TBU-253).

    A carried value is preserved. A missing or malformed one is computed
    exactly as the interactive write path does -- ``compute_importance(content,
    tags)``, see ``service.store_memory_enriched`` -- instead of falling to the
    column default of 0.5.

    Only for inserts. An upsert onto an existing row must not replace a stored
    importance with a computed guess; see the upsert call site.
    """
    from ogham.extraction import compute_importance

    carried = _carried_importance(mem)
    if carried is not None:
        return carried
    return compute_importance(mem.get("content") or "", list(mem.get("tags") or []))


def _round_importance(memory: dict[str, Any]) -> dict[str, Any]:
    """Export-side: importance to 4 decimal places (0.30000000000000004 -> 0.3).

    A copy, so the caller's row is untouched. Import accepts whatever it is
    given; rounding is a presentation choice made once, on the way out.
    """
    value = memory.get("importance")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {**memory, "importance": round(float(value), 4)}
    return memory


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


def _check_okf_index(bundle_dir: Path) -> list[str]:
    """Read the bundle-root index.md for diagnostics only. Never refuses.

    OKF v0.3 §11.2: consumers MUST NOT reject a bundle for a missing index.md,
    and §12 says a consumer SHOULD NOT refuse an unknown or missing version.
    Up to v0.19 this rejected both. Each problem is logged as a warning and
    returned so the caller can surface it.
    """
    from ogham.okf import SUPPORTED_OKF_VERSION
    from ogham.okf.bundle import _GRAPH_VERSION, GRAPH_VERSION_KEYS
    from ogham.okf.serialization import read_concept

    warnings: list[str] = []
    index_path = bundle_dir / "index.md"
    fm: dict[str, Any] = {}
    if not index_path.is_file():
        warnings.append(f"{bundle_dir} has no index.md; importing it as an OKF bundle anyway")
    else:
        try:
            fm, _ = read_concept(index_path)
        except ValueError as exc:
            warnings.append(f"{index_path} has no readable frontmatter ({exc}); continuing")
        if fm and "okf_version" not in fm:
            warnings.append(f"{index_path} declares no okf_version; reading it best-effort")
        elif "okf_version" in fm and str(fm["okf_version"]) not in ("0.1", SUPPORTED_OKF_VERSION):
            warnings.append(
                f"{index_path} declares okf_version {fm['okf_version']!r}; this Ogham "
                f"writes {SUPPORTED_OKF_VERSION} and reads it best-effort"
            )
    graph_version = next((fm[k] for k in GRAPH_VERSION_KEYS if k in fm), None)
    if isinstance(graph_version, int) and graph_version > _GRAPH_VERSION:
        warnings.append(
            f"{index_path} declares graph version {graph_version}; this Ogham reads up "
            f"to {_GRAPH_VERSION}, so some of the graph layer may be skipped"
        )
    for message in warnings:
        logger.warning("OKF import: %s", message)
    return warnings


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
    it is treated as an OKF bundle (v0.3, and the v0.1 shape earlier Ogham
    releases wrote). Memories with an ``id`` in frontmatter
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
        bundle_dir = Path(data)
        warnings = _check_okf_index(bundle_dir)

        from ogham.okf import import_okf_bundle
        from ogham.okf.bundle import import_okf_graph as _read_graph

        okf_memories, stats = import_okf_bundle(bundle_dir)
        if stats["total"] == 0 and _read_graph(bundle_dir)[1]["total"] == 0:
            # The one refusal left (OKF §11.2 forbids refusing over index.md or
            # version). A directory with no concept anywhere -- no .md file
            # whose frontmatter has a `type` (§3.2) -- is not a bundle, and is
            # most likely the wrong path.
            raise ValueError(
                f"{data} contains no OKF concept files (a .md file whose YAML "
                f"frontmatter has a non-empty `type`) -- not an OKF bundle"
            )

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
                mem_with_embedding = {
                    **mem,
                    "embedding": embedding,
                    "profile": profile,
                    # Carried value or None. None means "no opinion": the
                    # backend keeps a stored importance on conflict and uses
                    # importance_if_new only when the row is created.
                    "importance": _carried_importance(mem),
                    "importance_if_new": _resolve_importance(mem),
                }
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
            "not_concept_count": stats["not_concept_count"],
            "warnings": warnings,
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
