"""Telegram ingest MCP tool + CLI backend -- pull messages via getUpdates.

``ingest_telegram_impl`` is pure over an injected client + ``IngestService``
(unit-testable with no network, no DB). The ``@mcp.tool`` wrapper composes the
concrete ``TelegramClient`` + ``DefaultIngestService`` and self-registers on
import. Mirrors ``ogham.tools.import_linear`` / ``import_obsidian``.
"""

from __future__ import annotations

import os
from typing import Protocol

from ogham.app import mcp
from ogham.importers.telegram import TelegramClient, map_update_to_record, update_chat_id
from ogham.ingest import DefaultIngestService, IngestService, run_ingest
from ogham.tools.memory import get_active_profile

_SOURCE = "telegram"
_DEDUP_KEY = "telegram_update_id"


class _GetUpdatesClient(Protocol):
    """Structural type for ``ingest_telegram_impl`` -- lets tests inject a
    fake without subclassing ``TelegramClient``."""

    def get_updates(self, offset: int | None = None, timeout: int = 0) -> list[dict]: ...


def _start_offset(existing: set[str]) -> int | None:
    """Next getUpdates offset = max stored update_id + 1 (state-free cursor).

    Guards against a non-numeric stored key so one bad row can't brick every
    future run.
    """
    numeric = [int(k) for k in existing if k.lstrip("-").isdigit()]
    return max(numeric) + 1 if numeric else None


def ingest_telegram_impl(
    *,
    client: _GetUpdatesClient,
    service: IngestService,
    profile: str,
    source: str = _SOURCE,
    allowed_chat_ids: set[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """Pull Telegram messages and store new ones. Idempotent; safe to re-run.

    Interleaves fetch and store: each page is stored (in update_id order,
    stopping at the first store error) BEFORE the next page is requested. The
    getUpdates offset -- a server-side ack at request time -- therefore only ever
    advances past updates that were durably stored (or intentionally skipped:
    no-text / dedup / allowlist-filtered). A transient store failure stalls the
    run without advancing, so the update is re-fetched next run rather than lost.
    Trade-off: a PERSISTENTLY failing update stalls newer capture until it stores
    or ages out of Telegram's <=24h window (surfaced in ``errors``).
    """
    existing = service.fetch_existing_keys(profile, source, _DEDUP_KEY)
    offset = _start_offset(existing)
    totals = {
        "scanned": 0,
        "stored": 0,
        "skipped_duplicate": 0,
        "skipped_ignored": 0,
        "disabled": 0,
        "errors": 0,
    }
    while True:
        batch = client.get_updates(offset=offset)
        if not batch:
            break
        if allowed_chat_ids is not None:
            items = [u for u in batch if update_chat_id(u) in allowed_chat_ids]
        else:
            items = batch
        summary = run_ingest(
            items=items,
            to_record=map_update_to_record,
            service=service,
            profile=profile,
            source=source,
            dedup_key_field=_DEDUP_KEY,
            existing=existing,
            dry_run=dry_run,
            stop_on_store_error=True,
        )
        for key in totals:
            totals[key] += summary[key]
        if summary["stopped"]:
            break  # store error halted the page; do NOT advance offset -> re-fetch next run
        offset = max(u["update_id"] for u in batch) + 1
    return totals


def _allowed_chat_ids_from_env() -> set[int] | None:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS")
    if not raw:
        return None
    try:
        return {int(x) for x in raw.split(",") if x.strip()}
    except ValueError as exc:
        raise ValueError(
            f"TELEGRAM_ALLOWED_CHAT_IDS must be comma-separated integers: {exc}"
        ) from None


@mcp.tool
def ingest_telegram(
    profile: str | None = None,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Ingest Telegram messages into Ogham, deduped by telegram_update_id.

    Outbound getUpdates poll -- no webhook, no exposed endpoint. Requires the
    ``TELEGRAM_BOT_TOKEN`` env var; honors optional ``TELEGRAM_ALLOWED_CHAT_IDS``.

    Returns:
        ``{"scanned","stored","skipped_duplicate","skipped_ignored","disabled","errors"}``.

    Raises:
        ValueError: if ``TELEGRAM_BOT_TOKEN`` is not set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")
    target = profile or get_active_profile()
    return ingest_telegram_impl(
        client=TelegramClient(token=token),
        service=DefaultIngestService(),
        profile=target,
        source=source,
        allowed_chat_ids=_allowed_chat_ids_from_env(),
        dry_run=dry_run,
    )
