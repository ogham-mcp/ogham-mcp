"""Telegram importer -- pull messages via getUpdates and map to IngestRecord.

Outbound-only: the adapter calls out to api.telegram.org. No webhook, no public
endpoint. The mapper is pure; the client wraps httpx (injectable for tests),
same shape as ``ogham.importers.linear.LinearClient``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from ogham.ingest import IngestRecord

TELEGRAM_API = "https://api.telegram.org"

# The four update kinds that carry a user/channel message we capture.
# `business_message` / `guest_message` also exist in the Bot API but are
# intentionally out of scope for a personal capture bot.
_MESSAGE_KEYS = ("message", "edited_message", "channel_post", "edited_channel_post")


class TelegramClient:
    """Thin getUpdates client scoped to what the importer needs."""

    def __init__(self, token: str, http_client: httpx.Client | None = None):
        self._token = token
        self._http = http_client or httpx.Client(timeout=30.0)

    def get_updates(self, offset: int | None = None, timeout: int = 0) -> list[dict]:
        """Return the pending updates. ``offset`` = last update_id + 1 confirms
        prior updates server-side. Raises RuntimeError if Telegram reports not-ok.

        The bot token lives in the request URL (the Bot API has no header
        auth), so request/HTTP errors are sanitized here -- never let a
        token-bearing URL reach an exception message or a log line.
        """
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = self._http.get(f"{TELEGRAM_API}/bot{self._token}/getUpdates", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"telegram getUpdates HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"telegram getUpdates request failed: {type(exc).__name__}"
            ) from None
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"telegram getUpdates failed: {payload.get('description')}")
        return payload.get("result", [])


def _message_of(update: dict) -> dict | None:
    for key in _MESSAGE_KEYS:
        msg = update.get(key)
        if isinstance(msg, dict):
            return msg
    return None


def update_chat_id(update: dict) -> int | None:
    msg = _message_of(update)
    if msg is None:
        return None
    chat = msg.get("chat") or {}
    cid = chat.get("id")
    return int(cid) if cid is not None else None


def map_update_to_record(update: dict) -> IngestRecord | None:
    """Map a getUpdates entry to an IngestRecord, or None if it has no text.

    Captures ``text`` or media ``caption`` from message / edited_message /
    channel_post. ``telegram_update_id`` is the dedup key.
    """
    msg = _message_of(update)
    if msg is None:
        return None
    content = msg.get("text") or msg.get("caption")
    if not content:
        return None
    update_id = update.get("update_id")
    if update_id is None:
        # Belt-and-suspenders: the API guarantees this, but a missing
        # update_id would otherwise poison the offset/dedup math downstream.
        return None

    chat = msg.get("chat") or {}
    sender = msg.get("from") or {}
    date_val = msg.get("date")
    date_iso = (
        datetime.fromtimestamp(int(date_val), tz=timezone.utc).isoformat()
        if date_val is not None
        else None
    )
    metadata = {
        "telegram_update_id": update_id,
        "message_id": msg.get("message_id"),
        "chat_id": chat.get("id"),
        "chat_type": chat.get("type"),
        "chat_title": chat.get("title"),
        "chat_username": chat.get("username"),
        "from_id": sender.get("id"),
        "from_username": sender.get("username"),
        "date": date_iso,
    }
    return IngestRecord(content=content, tags=[], metadata=metadata)
