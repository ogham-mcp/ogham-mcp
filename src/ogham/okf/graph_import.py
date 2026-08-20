"""Apply parsed OKF entity concepts to the graph (TBU-219, the write half).

Separate from ``bundle.py`` on purpose: that module owns IO and marshalling and
touches no database, exactly as ``import_okf_bundle`` returns memory dicts
rather than storing them.

Two findings from the 2026-08-04 design council shape this:

* **Merge, not restore.** ``store_triple`` supersedes a matching current edge,
  so importing into a populated profile moves edges forward in time rather than
  restoring bundle state. Fidelity is only defined against a fresh profile and
  nothing enforces freshness, so ``apply_okf_graph`` reports what it found
  rather than pretending it restored anything.
* **Retry was not idempotent.** Per-edge writes meant a failure partway left a
  half-merged graph, and re-running re-superseded everything already written and
  re-stamped ``valid_from`` -- the natural operator reflex degraded data on each
  attempt. Fixed here by skipping an edge that already exists as a current row,
  so a retry converges instead of churning.
"""

from __future__ import annotations

import logging
from typing import Any

from ogham.entity_graph import V1_PREDICATES, Predicate
from ogham.okf.entities import ParsedEntityConcept

logger = logging.getLogger(__name__)


def apply_okf_graph(
    concepts: list[ParsedEntityConcept],
    profile: str,
    graph: Any,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write parsed concepts into the entity graph. Returns stats.

    Resolution is **by note name**, never by the source entity id: the id
    identifies a concept inside the bundle it came from and means nothing here.
    Local ids come from the natural key (canonical_name, entity_type).

    An edge whose object is not present in the bundle is counted as
    ``unresolved_edges`` rather than dropped silently -- the same reason the
    exporter writes ``ogham_dangling`` instead of omitting the edge.

    ``dry_run`` writes nothing and reports what would happen. It exists because
    the honest safety net for this operation is weak: importing into a populated
    profile MERGES rather than restores, and the obvious undo -- snapshot the
    profile first -- is built from the same export path, so restoring from it
    would flatten ``strength`` and re-stamp ``valid_from``. That is not an undo.
    When you cannot offer a reliable way back, the next best thing is a reliable
    way to look first.

    The dry run is accurate rather than indicative: it resolves existing
    entities read-only through ``resolve_alias`` on the QUALIFIED reference, so
    ``entities_existing`` and ``edges_already_present`` are real counts, not
    estimates.
    """
    note_to_local_id: dict[str, int] = {}
    entities_seen = 0
    entities_existing = 0
    #: Notes whose entity does not exist yet. In a dry run there is no id to
    #: key on, so edges touching them are counted as "would write" without a
    #: lookup -- a not-yet-created entity has no edges by definition.
    pending_notes: set[str] = set()

    # Pass 1: every concept becomes a local entity BEFORE any edge is written,
    # so an edge can never fail merely because its object came later in the
    # directory listing.
    for concept in concepts:
        entities_seen += 1
        existing_id = _existing_entity_id(graph, concept, profile)
        if existing_id is not None:
            entities_existing += 1

        if dry_run:
            if existing_id is not None and concept.note_name:
                note_to_local_id[concept.note_name] = existing_id
            elif concept.note_name:
                pending_notes.add(concept.note_name)
            continue

        local_id = graph.upsert_entity(concept.canonical_name, concept.entity_type)
        if concept.note_name:
            note_to_local_id[concept.note_name] = local_id
        for alias in concept.aliases:
            try:
                graph.add_alias(local_id, alias, profile)
            except Exception as exc:
                logger.debug("alias %r skipped for entity %s: %s", alias, local_id, exc)

    # Existing current edges, so a retry converges rather than re-superseding.
    existing: set[tuple[int, str, int]] = set()
    try:
        for edge in graph.list_edges(profile, current_only=True):
            existing.add((edge.subject_id, str(edge.predicate), edge.object_id))
    except Exception as exc:  # pragma: no cover - backend without list_edges
        logger.warning("could not read existing edges, retry may re-supersede: %s", exc)

    edges_written = 0
    edges_already_present = 0
    unresolved_edges = 0

    for concept in concepts:
        subject_id = note_to_local_id.get(concept.note_name)
        if subject_id is None:
            if dry_run and concept.note_name in pending_notes:
                # Subject would be created, so its edges would be attempted.
                # Each still needs its own object check below, so fall through
                # with a sentinel rather than counting them all as writes.
                subject_id = -1
            else:
                unresolved_edges += len(concept.edges)
                continue
        for predicate, target_note in concept.edges:
            if predicate not in V1_PREDICATES:
                # Belt and braces: the parser allowlists too, but a caller can
                # hand us concepts it built itself.
                unresolved_edges += 1
                continue
            object_id = note_to_local_id.get(target_note)
            if object_id is None:
                if dry_run and target_note in pending_notes:
                    # Object exists in the bundle but not yet locally, so the
                    # edge would be written once both ends are created.
                    edges_written += 1
                    continue
                unresolved_edges += 1
                continue
            if subject_id == object_id:
                # store_triple rejects these; counting is more useful than raising
                # halfway through an import.
                unresolved_edges += 1
                continue
            key = (subject_id, predicate, object_id)
            if key in existing:
                edges_already_present += 1
                continue
            if not dry_run:
                graph.store_triple(
                    subject_id,
                    Predicate(predicate),
                    object_id,
                    None,
                    profile,
                )
            existing.add(key)
            edges_written += 1

    return {
        "dry_run": dry_run,
        "entities": entities_seen,
        "entities_existing": entities_existing,
        "entities_new": entities_seen - entities_existing,
        "edges_written": edges_written,
        "edges_already_present": edges_already_present,
        "unresolved_edges": unresolved_edges,
        "profile": profile,
    }


def _existing_entity_id(graph: Any, concept: ParsedEntityConcept, profile: str) -> int | None:
    """Look up an entity read-only, on the exact natural key.

    Both parts, passed separately. Resolving a qualified ``type:name`` STRING
    would only work for the 11 types ``extract_entities`` produces, and a bundle
    may carry any type -- which is exactly the case import introduces.
    """
    try:
        return graph.find_entity(concept.canonical_name, concept.entity_type)
    except Exception as exc:
        logger.debug("read-only lookup failed for %s: %s", concept.canonical_name, exc)
        return None
