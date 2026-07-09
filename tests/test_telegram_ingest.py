import httpx
from typer.testing import CliRunner

from ogham.cli import app
from ogham.importers.telegram import TelegramClient, map_update_to_record, update_chat_id
from ogham.tools.import_telegram import (
    _allowed_chat_ids_from_env,
    _start_offset,
    ingest_telegram_impl,
)


def _client(handler):
    return TelegramClient("TOKEN", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_get_updates_builds_request_and_parses_result():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True, "result": [{"update_id": 5}]})

    updates = _client(handler).get_updates(offset=5, timeout=0)
    assert updates == [{"update_id": 5}]
    assert "/botTOKEN/getUpdates" in seen["url"]
    assert "offset=5" in seen["url"]


def test_get_updates_raises_on_not_ok():
    def handler(request):
        return httpx.Response(200, json={"ok": False, "description": "bad token"})

    try:
        _client(handler).get_updates()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "bad token" in str(e)
        assert "TOKEN" not in str(e)


def test_get_updates_sanitizes_http_error_no_token_in_message():
    def handler(request):
        return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

    try:
        _client(handler).get_updates()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "TOKEN" not in str(e)
        assert "401" in str(e)


def _msg_update(uid=10, text: str | None = "hello", chat_id=42, mid=7, date=1751800000):
    return {
        "update_id": uid,
        "message": {
            "message_id": mid,
            "date": date,
            "chat": {"id": chat_id, "type": "private", "title": None, "username": "kev"},
            "from": {"id": 99, "username": "kev"},
            "text": text,
        },
    }


def test_map_text_message():
    rec = map_update_to_record(_msg_update(text="hi there"))
    assert rec is not None
    assert rec.content == "hi there"
    assert rec.metadata["telegram_update_id"] == 10
    assert rec.metadata["chat_id"] == 42
    assert rec.metadata["message_id"] == 7
    assert isinstance(rec.metadata["date"], str)  # unix -> ISO string


def test_map_caption_when_no_text():
    u = _msg_update(text=None)
    u["message"]["caption"] = "a photo caption"
    del u["message"]["text"]
    u["message"]["photo"] = [{"file_id": "x"}]
    rec = map_update_to_record(u)
    assert rec is not None and rec.content == "a photo caption"


def test_map_no_text_no_caption_returns_none():
    u = _msg_update(text=None)
    del u["message"]["text"]
    assert map_update_to_record(u) is None


def test_map_edited_message():
    u = {"update_id": 11, "edited_message": _msg_update(uid=11)["message"]}
    rec = map_update_to_record(u)
    assert rec is not None and rec.metadata["telegram_update_id"] == 11


def test_map_channel_post():
    u = {"update_id": 12, "channel_post": _msg_update(uid=12)["message"]}
    rec = map_update_to_record(u)
    assert rec is not None and rec.content == "hello"


def test_map_edited_channel_post():
    u = {"update_id": 13, "edited_channel_post": _msg_update(uid=13)["message"]}
    rec = map_update_to_record(u)
    assert rec is not None and rec.metadata["telegram_update_id"] == 13


def test_map_missing_update_id_returns_none():
    u = _msg_update()
    del u["update_id"]
    assert map_update_to_record(u) is None


def test_update_chat_id():
    assert update_chat_id(_msg_update(chat_id=77)) == 77
    assert update_chat_id({"update_id": 1}) is None


def test_start_offset_ignores_non_numeric_existing_key():
    assert _start_offset({"5", "bad"}) == 6
    assert _start_offset(set()) is None


def test_allowed_chat_ids_from_env_raises_clean_value_error():
    import pytest

    with pytest.raises(ValueError, match="TELEGRAM_ALLOWED_CHAT_IDS"):
        import os

        os.environ["TELEGRAM_ALLOWED_CHAT_IDS"] = "abc"
        try:
            _allowed_chat_ids_from_env()
        finally:
            del os.environ["TELEGRAM_ALLOWED_CHAT_IDS"]


class _FakeTgClient:
    """Returns scripted batches on successive get_updates calls, then []."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.offsets = []

    def get_updates(self, offset=None, timeout=0):
        self.offsets.append(offset)
        return self._batches.pop(0) if self._batches else []


class _FakeSvc:
    def __init__(self, existing=None, raise_on_update_id=None):
        self.existing = set(existing or [])
        self.stored = []
        self.raise_on_update_id = raise_on_update_id or set()

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        if record.metadata["telegram_update_id"] in self.raise_on_update_id:
            raise RuntimeError("boom store")
        self.stored.append(record)
        return {"status": "stored", "id": "x"}


def _u(uid, text="hi", chat_id=42):
    return {
        "update_id": uid,
        "message": {
            "message_id": uid,
            "date": 1751800000,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 1},
            "text": text,
        },
    }


def test_impl_stores_and_drains_across_pages():
    client = _FakeTgClient([[_u(1), _u(2)], [_u(3)]])  # then [] terminates
    svc = _FakeSvc()
    r = ingest_telegram_impl(client=client, service=svc, profile="work")
    assert r["scanned"] == 3 and r["stored"] == 3
    assert {rec.metadata["telegram_update_id"] for rec in svc.stored} == {1, 2, 3}
    # offset advanced across pages: first call unset, second call past page 1, third past page 2
    assert client.offsets == [None, 3, 4]


def test_impl_store_error_stops_drain_and_does_not_advance():
    client = _FakeTgClient([[_u(1), _u(2), _u(3)], [_u(4)]])
    svc = _FakeSvc(raise_on_update_id={2})
    r = ingest_telegram_impl(client=client, service=svc, profile="work")
    assert len(client.offsets) == 1  # second page never fetched
    assert r["stored"] == 1 and r["errors"] == 1
    stored_ids = {rec.metadata["telegram_update_id"] for rec in svc.stored}
    assert stored_ids == {1}
    assert 2 not in stored_ids and 3 not in stored_ids


def test_impl_offset_derived_from_existing():
    client = _FakeTgClient([[_u(6)]])
    svc = _FakeSvc(existing={"5"})  # max stored update_id = 5 -> first offset 6
    ingest_telegram_impl(client=client, service=svc, profile="work")
    assert client.offsets[0] == 6


def test_impl_allowlist_filters_other_chats():
    client = _FakeTgClient([[_u(1, chat_id=42), _u(2, chat_id=999)]])
    svc = _FakeSvc()
    r = ingest_telegram_impl(client=client, service=svc, profile="work", allowed_chat_ids={42})
    assert r["stored"] == 1
    assert svc.stored[0].metadata["chat_id"] == 42


def test_impl_fully_filtered_page_still_advances_offset():
    # Every update in the page is outside the allowlist -> items is empty, but
    # the page is still considered "handled" (not stopped), so the offset
    # advances past it and the next (empty) page terminates the loop.
    client = _FakeTgClient([[_u(1, chat_id=999), _u(2, chat_id=999)]])
    svc = _FakeSvc()
    r = ingest_telegram_impl(client=client, service=svc, profile="work", allowed_chat_ids={42})
    assert r["stored"] == 0
    assert client.offsets == [None, 3]  # advanced past the filtered page's max update_id (2) + 1


def test_impl_dedups_existing_update_id():
    client = _FakeTgClient([[_u(1)]])
    svc = _FakeSvc(existing={"1"})
    r = ingest_telegram_impl(client=client, service=svc, profile="work")
    assert r["stored"] == 0 and r["skipped_duplicate"] == 1


def test_cli_missing_token_errors(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["ingest-telegram", "--profile", "work"])
    assert result.exit_code == 1
    assert "TELEGRAM_BOT_TOKEN" in result.output


def test_cli_dry_run_reports_counts(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    # No network: swap the client for one that returns no updates, and the service for a fake.
    monkeypatch.setattr("ogham.cli.TelegramClient", lambda token: _FakeTgClient([]))
    monkeypatch.setattr("ogham.cli.DefaultIngestService", _FakeSvc)
    result = CliRunner().invoke(app, ["ingest-telegram", "--profile", "work", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "scanned=0" in result.output
