import httpx
from typer.testing import CliRunner

from ogham.cli import app
from ogham.importers.slack import SlackClient, map_message_to_record
from ogham.tools.import_slack import _channel_oldest, ingest_slack_impl


def _client(handler):
    return SlackClient(
        "xoxb-TOKEN", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_conversations_history_parses_messages_and_cursor():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": [{"ts": "1.1", "text": "hi", "user": "U1"}],
                "response_metadata": {"next_cursor": "CUR2"},
            },
        )

    messages, cursor = _client(handler).conversations_history("C1", oldest="1.0", cursor="CUR1")
    assert messages == [{"ts": "1.1", "text": "hi", "user": "U1"}]
    assert cursor == "CUR2"
    assert "conversations.history" in seen["url"]
    assert (
        "channel=C1" in seen["url"] and "oldest=1.0" in seen["url"] and "cursor=CUR1" in seen["url"]
    )
    assert seen["auth"] == "Bearer xoxb-TOKEN"


def test_conversations_history_empty_cursor_is_none():
    def handler(request):
        return httpx.Response(
            200, json={"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}}
        )

    messages, cursor = _client(handler).conversations_history("C1")
    assert messages == [] and cursor is None


def test_conversations_history_not_ok_raises_clean_error():
    def handler(request):
        return httpx.Response(200, json={"ok": False, "error": "not_in_channel"})

    try:
        _client(handler).conversations_history("C1")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "not_in_channel" in str(e)
        assert "xoxb-TOKEN" not in str(e)


def test_map_text_message():
    rec = map_message_to_record("C1", {"ts": "1234.5", "text": "hello team", "user": "U9"})
    assert rec is not None
    assert rec.content == "hello team"
    assert rec.metadata["slack_key"] == "C1:1234.5"
    assert rec.metadata["channel_id"] == "C1"
    assert rec.metadata["message_ts"] == "1234.5"
    assert rec.metadata["user"] == "U9"


def test_map_skips_subtype_and_empty_and_missing_ts():
    assert (
        map_message_to_record("C1", {"ts": "1.1", "text": "joined", "subtype": "channel_join"})
        is None
    )
    assert map_message_to_record("C1", {"ts": "1.1", "text": "", "user": "U1"}) is None
    assert map_message_to_record("C1", {"text": "no ts", "user": "U1"}) is None


def test_map_skips_none_and_malformed_ts():
    assert map_message_to_record("C1", {"ts": None, "text": "hi", "user": "U1"}) is None
    assert map_message_to_record("C1", {"ts": "weird", "text": "hi", "user": "U1"}) is None


def test_channel_oldest_picks_max_ts_per_channel():
    existing = {"C1:100.1", "C1:200.2", "C2:50.5", "C1:bad"}
    assert _channel_oldest(existing, "C1") == "200.2"  # ignores the non-numeric key
    assert _channel_oldest(existing, "C2") == "50.5"
    assert _channel_oldest(existing, "C3") is None


class _FakeSlackClient:
    """Serves scripted pages per channel: {channel: [(messages, next_cursor), ...]}."""

    def __init__(self, pages):
        self._pages = {c: list(p) for c, p in pages.items()}
        self.calls = []

    def conversations_history(self, channel, oldest=None, cursor=None, limit=200):
        self.calls.append((channel, oldest, cursor))
        queue = self._pages.get(channel, [])
        return queue.pop(0) if queue else ([], None)


class _FakeSvc:
    def __init__(self, existing=None, reject_ts=None, raise_ts=None):
        self.existing = set(existing or [])
        self.reject_ts = reject_ts or set()  # -> ValueError (permanent skip)
        self.raise_ts = raise_ts or set()  # -> RuntimeError (transient stop)
        self.stored = []

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        ts = record.metadata["message_ts"]
        if ts in self.reject_ts:
            raise ValueError("content too short")
        if ts in self.raise_ts:
            raise RuntimeError("boom")
        self.stored.append(record)
        return {"status": "stored", "id": "x"}


def _msg(ts, text="a real message", user="U1"):
    return {"ts": ts, "text": text, "user": user}


def test_impl_single_channel_paginates_and_stores(tmp_path):
    client = _FakeSlackClient({"C1": [([_msg("2.0"), _msg("3.0")], "CUR"), ([_msg("1.0")], None)]})
    svc = _FakeSvc()
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1"])
    assert r["scanned"] == 3 and r["stored"] == 3
    # stored oldest-first (sorted across both pages): 1.0, 2.0, 3.0
    assert [rec.metadata["message_ts"] for rec in svc.stored] == ["1.0", "2.0", "3.0"]


def test_impl_multi_channel_totals(tmp_path):
    client = _FakeSlackClient({"C1": [([_msg("1.0")], None)], "C2": [([_msg("2.0")], None)]})
    svc = _FakeSvc()
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1", "C2"])
    assert r["stored"] == 2


def test_impl_derives_oldest_per_channel(tmp_path):
    client = _FakeSlackClient({"C1": [([_msg("9.0")], None)]})
    svc = _FakeSvc(existing={"C1:5.5"})
    ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1"])
    assert client.calls[0] == ("C1", "5.5", None)  # oldest = max stored ts for C1


def test_impl_store_error_stops_that_channel(tmp_path):
    # page [3,2] -> sorted to [2,3]; store 2 ok, 3 raises -> stop
    client = _FakeSlackClient({"C1": [([_msg("3.0"), _msg("2.0")], None)]})
    svc = _FakeSvc(raise_ts={"3.0"})
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1"])
    assert r["errors"] == 1
    assert [rec.metadata["message_ts"] for rec in svc.stored] == [
        "2.0"
    ]  # 3.0 not stored, not skipped-past


def test_impl_dedups_existing(tmp_path):
    client = _FakeSlackClient({"C1": [([_msg("1.0")], None)]})
    svc = _FakeSvc(existing={"C1:1.0"})
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1"])
    assert r["stored"] == 0 and r["skipped_duplicate"] == 1


def test_impl_multi_channel_one_channel_fails_other_clean(tmp_path):
    # C1: page [3,2] -> sorted [2,3]; store 2 ok, 3 raises -> stop C1.
    # C2: clean, stores independently. Totals sum both channels.
    client = _FakeSlackClient(
        {
            "C1": [([_msg("3.0"), _msg("2.0")], None)],
            "C2": [([_msg("5.0")], None)],
        }
    )
    svc = _FakeSvc(raise_ts={"3.0"})
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1", "C2"])
    assert r["errors"] == 1
    assert r["stored"] == 2
    assert {rec.metadata["message_ts"] for rec in svc.stored} == {"2.0", "5.0"}


def test_impl_malformed_ts_skipped_run_does_not_crash(tmp_path):
    # A None ts and a non-numeric ts must not crash the sort; both are skipped
    # (skipped_ignored) and valid messages in the same channel still store.
    client = _FakeSlackClient(
        {"C1": [([_msg("1.0"), {"ts": None, "text": "x", "user": "U1"}, _msg("weird")], None)]}
    )
    svc = _FakeSvc()
    r = ingest_slack_impl(client=client, service=svc, profile="work", channels=["C1"])
    assert r["scanned"] == 3
    assert r["stored"] == 1
    assert r["skipped_ignored"] == 2
    assert [rec.metadata["message_ts"] for rec in svc.stored] == ["1.0"]


def test_cli_missing_token_errors(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["ingest-slack", "--profile", "work"])
    assert result.exit_code == 1
    assert "SLACK_BOT_TOKEN" in result.output


def test_cli_dry_run_reports_counts(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-TOKEN")
    monkeypatch.setenv("SLACK_CHANNELS", "C1")
    monkeypatch.setattr(
        "ogham.cli.SlackClient", lambda token: _FakeSlackClient({"C1": [([], None)]})
    )
    monkeypatch.setattr("ogham.cli.DefaultIngestService", _FakeSvc)
    result = CliRunner().invoke(app, ["ingest-slack", "--profile", "work", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "scanned=0" in result.output
