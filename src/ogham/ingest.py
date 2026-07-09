"""The shared "-> OM" ingest contract.

An adapter is: an iterable of raw items + a ``to_record`` mapper + a ``source``
+ a ``dedup_key_field``. ``run_ingest`` owns the loop, dedup, dry-run,
disabled-handling, and per-item resilience once, for every adapter (Obsidian,
Telegram, ...). Extracted at the second adapter (rule-of-three).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from ogham import database
from ogham.service import store_memory_enriched

logger = logging.getLogger(__name__)


@dataclass
class IngestRecord:
    """One item mapped to the OM store shape. ``metadata`` MUST contain the
    dedup-key field named by ``run_ingest``'s ``dedup_key_field``."""

    content: str
    tags: list[str]
    metadata: dict


class IngestService(Protocol):
    def fetch_existing_keys(self, profile: str, source: str, key_field: str) -> set[str]: ...
    def store(self, record: IngestRecord, profile: str, source: str) -> dict: ...


class DefaultIngestService:
    """Adapts ``ogham.database`` + ``ogham.service`` to ``IngestService``."""

    def fetch_existing_keys(self, profile: str, source: str, key_field: str) -> set[str]:
        """All stored ``metadata[key_field]`` values for ``source`` in ``profile``.

        One backend round-trip (mirrors the Linear importer's
        ``fetch_all_tracker_ids``) instead of an N-item per-record lookup.
        """
        keys: set[str] = set()
        for memory in database.get_backend().get_all_memories_full(profile):
            if memory.get("source") != source:
                continue
            value = (memory.get("metadata") or {}).get(key_field)
            if value is not None:
                keys.add(str(value))
        return keys

    def store(self, record: IngestRecord, profile: str, source: str) -> dict:
        return store_memory_enriched(
            content=record.content,
            profile=profile,
            source=source,
            tags=record.tags,
            metadata=record.metadata,
        )


def run_ingest(
    *,
    items: Iterable[Any],
    to_record: Callable[[Any], IngestRecord | None],
    service: IngestService,
    profile: str,
    source: str,
    dedup_key_field: str,
    existing: set[str] | None = None,
    dry_run: bool = False,
    stop_on_store_error: bool = False,
) -> dict:
    """Map + store each item, deduped by ``str(metadata[dedup_key_field])``.

    Map errors (``to_record`` raising) and a record missing the dedup-key field
    are ALWAYS skipped + counted as an error, run continues -- they are
    permanent/item-specific, not something a retry fixes. A store ``ValueError``
    is likewise a PERMANENT rejection (e.g. content too short/empty) -- it is
    counted under ``skipped_ignored`` and the run continues, never stalling.
    Any other STORE error is counted as an error; when ``stop_on_store_error``
    is True the loop instead BREAKS (``stopped=True`` in the result) so the
    caller can avoid acking (e.g. advancing a cursor past) an item that never
    durably landed -- stall-over-lose. A changed item yields a new memory; no
    supersession. Correctness (dedup) lives in the DB via ``existing``.

    Returns ``{"scanned","stored","skipped_duplicate","skipped_ignored",
    "disabled","errors","stopped"}``.
    """
    if existing is None:
        existing = service.fetch_existing_keys(profile, source, dedup_key_field)
    scanned = stored = skipped_duplicate = skipped_ignored = disabled = errors = 0
    stopped = False

    for item in items:
        scanned += 1
        try:
            record = to_record(item)
        except Exception as exc:  # noqa: BLE001 -- mapping is item-specific/permanent; skip + continue
            logger.warning("ingest[%s]: mapping failed: %s", source, exc)
            errors += 1
            continue
        if record is None or not record.content.strip():
            skipped_ignored += 1
            continue
        key_value = record.metadata.get(dedup_key_field)
        if key_value is None:
            logger.warning("ingest[%s]: record missing dedup field %r", source, dedup_key_field)
            errors += 1
            continue
        key = str(key_value)
        if key in existing:
            skipped_duplicate += 1
            continue
        if dry_run:
            existing.add(key)
            stored += 1
            continue
        try:
            result = service.store(record, profile, source)
        except ValueError as exc:
            # Permanent: the store validated and rejected this content (e.g. too
            # short / empty). It will never store, so skip + continue -- never
            # stall the drain on it (that would block everything behind it).
            logger.info("ingest[%s]: skipped (rejected by store): %s", source, exc)
            skipped_ignored += 1
            continue
        except Exception as exc:  # noqa: BLE001 -- transient (DB/network); may stop
            logger.warning("ingest[%s]: store failed: %s", source, exc)
            errors += 1
            if stop_on_store_error:
                stopped = True
                break
            continue
        if isinstance(result, dict) and result.get("status") == "disabled":
            disabled += 1
            continue
        existing.add(key)  # within-run dedup for identical items
        stored += 1

    return {
        "scanned": scanned,
        "stored": stored,
        "skipped_duplicate": skipped_duplicate,
        "skipped_ignored": skipped_ignored,
        "disabled": disabled,
        "errors": errors,
        "stopped": stopped,
    }
