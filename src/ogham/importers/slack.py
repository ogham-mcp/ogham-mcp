"""Slack importer -- poll conversations.history and map to IngestRecord.

Outbound-only: the adapter calls out to slack.com/api over HTTPS with a Bearer
token header (no token in the URL, unlike Telegram). The mapper is pure; the
client wraps httpx (injectable for tests), same shape as ``TelegramClient``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ogham.ingest import IngestRecord

SLACK_API = "https://slack.com/api"


def _parse_ts(ts: object) -> tuple[int, int] | None:
    """Parse a Slack ts ('seconds.microseconds') into an exact (sec, usec) tuple
    for precise ordering -- avoids float64 precision loss at 16-digit timestamps.
    Returns None for a missing or malformed ts."""
    if not isinstance(ts, str) or not ts:
        return None
    parts = ts.split(".")
    if len(parts) == 1 and parts[0].isdigit():
        return (int(parts[0]), 0)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return None


class SlackClient:
    """Thin conversations.history client scoped to what the importer needs."""

    def __init__(self, token: str, http_client: httpx.Client | None = None):
        self._token = token
        self._http = http_client or httpx.Client(timeout=30.0)

    def conversations_history(
        self,
        channel: str,
        oldest: str | None = None,
        cursor: str | None = None,
        limit: int = 200,
    ) -> tuple[list[dict], str | None]:
        """Return ``(messages, next_cursor)`` for one page. ``oldest`` is a Slack
        ts string; ``cursor`` paginates. Raises RuntimeError if Slack reports
        not-ok. The token rides an Authorization header, so it never appears in
        the URL or an error message."""
        params: dict[str, Any] = {"channel": channel, "limit": limit}
        if oldest is not None:
            params["oldest"] = oldest
        if cursor is not None:
            params["cursor"] = cursor
        try:
            resp = self._http.get(
                f"{SLACK_API}/conversations.history",
                params=params,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"slack conversations.history HTTP {exc.response.status_code}"
            ) from None
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"slack conversations.history request failed: {type(exc).__name__}"
            ) from None
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"slack conversations.history failed: {payload.get('error')}")
        next_cursor = (payload.get("response_metadata") or {}).get("next_cursor") or None
        return payload.get("messages", []), next_cursor


def map_message_to_record(channel_id: str, message: dict) -> IngestRecord | None:
    """Map a conversations.history message to an IngestRecord, or None to skip.

    Skips messages carrying a ``subtype`` (system events, bot messages), empty
    text, or a missing/malformed ``ts``. Dedup key = ``channel_id:ts``.
    """
    if message.get("subtype"):
        return None
    text = message.get("text")
    if not text:
        return None
    ts = message.get("ts")
    if _parse_ts(ts) is None:
        return None
    metadata = {
        "slack_key": f"{channel_id}:{ts}",
        "channel_id": channel_id,
        "message_ts": ts,
        "user": message.get("user"),
        "thread_ts": message.get("thread_ts"),
    }
    return IngestRecord(content=text, tags=[], metadata=metadata)
