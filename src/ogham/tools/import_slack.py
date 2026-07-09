"""Slack ingest MCP tool + CLI backend -- poll conversations.history.

``ingest_slack_impl`` is pure over an injected client + ``IngestService``
(unit-testable with no network, no DB). The ``@mcp.tool`` wrapper composes the
concrete ``SlackClient`` + ``DefaultIngestService`` and self-registers on import.
Mirrors ``ogham.tools.import_telegram``.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from ogham.app import mcp
from ogham.importers.slack import SlackClient, _parse_ts, map_message_to_record
from ogham.ingest import DefaultIngestService, IngestService, run_ingest
from ogham.tools.memory import get_active_profile

logger = logging.getLogger(__name__)

_SOURCE = "slack"
_DEDUP_KEY = "slack_key"


class _ConversationsHistoryClient(Protocol):
    """Structural type for ``ingest_slack_impl`` -- lets tests inject a fake."""

    def conversations_history(
        self, channel: str, oldest: str | None = None, cursor: str | None = None, limit: int = 200
    ) -> tuple[list[dict], str | None]: ...


def _channel_oldest(existing: set[str], channel: str) -> str | None:
    """Max stored ts for ``channel`` (state-free cursor), parsed from the
    ``channel:ts`` dedup keys. Malformed ts values are ignored so one bad row
    cannot brick the cursor."""
    prefix = f"{channel}:"
    candidates = [k[len(prefix) :] for k in existing if k.startswith(prefix)]
    valid = [t for t in candidates if _parse_ts(t) is not None]
    return max(valid, key=lambda t: _parse_ts(t) or (0, 0)) if valid else None


def ingest_slack_impl(
    *,
    client: _ConversationsHistoryClient,
    service: IngestService,
    profile: str,
    channels: list[str],
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Poll each channel's conversations.history and store new messages.

    Per channel: state-free cursor ``oldest`` = max stored ts; collect all pages
    (conversations.history is re-fetchable -- no ack -- so fetching ahead is
    safe), sort oldest-first by ``ts`` (a plain list reverse only orders
    correctly within a single page; multi-page runs need a real sort across the
    whole collected set; malformed ts sorts first and is skipped by the mapper,
    never crashes), then run_ingest with stop_on_store_error so the
    max-stored-ts cursor stays contiguous (a store failure re-fetches next
    run). Dedup by ``channel_id:ts``.
    """
    existing = service.fetch_existing_keys(profile, source, _DEDUP_KEY)
    totals = {
        "scanned": 0,
        "stored": 0,
        "skipped_duplicate": 0,
        "skipped_ignored": 0,
        "disabled": 0,
        "errors": 0,
    }
    for channel in channels:
        oldest = _channel_oldest(existing, channel)
        messages: list[dict] = []
        cursor: str | None = None
        while True:
            page, cursor = client.conversations_history(channel, oldest=oldest, cursor=cursor)
            messages.extend(page)
            if not cursor:
                break
        messages.sort(key=lambda m: _parse_ts(m.get("ts")) or (0, 0))  # store in ts order
        summary = run_ingest(
            items=messages,
            to_record=lambda m, ch=channel: map_message_to_record(ch, m),
            service=service,
            profile=profile,
            source=source,
            dedup_key_field=_DEDUP_KEY,
            existing=existing,
            dry_run=dry_run,
            stop_on_store_error=True,
        )
        if summary["errors"] or summary.get("stopped"):
            logger.warning(
                "ingest[slack]: channel %s -- errors=%d stopped=%s",
                channel,
                summary["errors"],
                summary.get("stopped"),
            )
        for key in totals:
            totals[key] += summary[key]
    return totals


def _channels_from_env() -> list[str]:
    raw = os.environ.get("SLACK_CHANNELS")
    channels = [c.strip() for c in (raw or "").split(",") if c.strip()]
    if not channels:
        raise ValueError("SLACK_CHANNELS not set (comma-separated channel IDs to poll)")
    return channels


@mcp.tool
def ingest_slack(
    profile: str | None = None,
    source: str = _SOURCE,
    dry_run: bool = False,
) -> dict:
    """Ingest Slack channel messages into Ogham, deduped by channel_id:ts.

    Outbound conversations.history poll -- no webhook, no Socket Mode. Requires
    ``SLACK_BOT_TOKEN`` and ``SLACK_CHANNELS`` env vars.

    Returns:
        ``{"scanned","stored","skipped_duplicate","skipped_ignored","disabled","errors"}``.

    Raises:
        ValueError: if ``SLACK_BOT_TOKEN`` or ``SLACK_CHANNELS`` is not set.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise ValueError("SLACK_BOT_TOKEN not set")
    target = profile or get_active_profile()
    return ingest_slack_impl(
        client=SlackClient(token=token),
        service=DefaultIngestService(),
        profile=target,
        channels=_channels_from_env(),
        source=source,
        dry_run=dry_run,
    )
