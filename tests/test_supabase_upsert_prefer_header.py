"""Regression test for v0.15.1: Supabase upsert Prefer header overwrite.

In v0.15.0 the SupabaseBackend client was initialised with a global
`Prefer: return=representation` header. postgrest-py's per-call upsert
builds its own Prefer header that includes `resolution=merge-duplicates`,
but the global header silently overwrites the per-call one during
`SyncRequestBuilder.upsert`'s `headers.update(self.headers)` step.

Result: every Supabase upsert was sent as a plain INSERT, which failed
with `duplicate key value violates unique constraint memories_pkey` on
the second call. The bug shipped to PyPI as 0.15.0 and was caught
during the live OKF round-trip demo. Fix: drop the global Prefer
header; let each operation builder set its own.

This test pins the fix by asserting the global header is absent. A
future change that re-adds it MUST also handle the merge-vs-overwrite
semantics.
"""

import pytest


@pytest.fixture
def supabase_settings(monkeypatch):
    """Patch settings so SupabaseBackend can initialise without env."""
    from ogham.config import settings as cfg

    monkeypatch.setattr(cfg, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(cfg, "supabase_key", "sb_secret_fakekey_for_testing")
    monkeypatch.setattr(cfg, "bare_postgrest", False)
    yield


def test_supabase_client_does_not_set_global_prefer_header(supabase_settings):
    """The client's global headers MUST NOT contain a Prefer key.

    If a future change re-adds it, every upsert silently degrades to
    plain INSERT and the second OKF import fails with duplicate-key.
    """
    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend()
    client = backend._get_client()

    # The SyncPostgrestClient stores its global headers; check directly.
    assert "Prefer" not in client.headers, (
        "Global Prefer header is set on the Supabase client. This will "
        "silently overwrite the per-call `resolution=merge-duplicates` "
        "header that postgrest-py builds for upserts, causing all upserts "
        "to fall through to plain INSERTs. See "
        "tests/test_supabase_upsert_prefer_header.py docstring for the "
        "v0.15.0 -> v0.15.1 history."
    )
    # Be explicit about what HAS to be there.
    assert "apikey" in client.headers
    assert "Authorization" in client.headers


def test_supabase_upsert_sends_resolution_merge_duplicates(supabase_settings):
    """The actual upsert request must include resolution=merge-duplicates.

    Captures the request via an httpx event hook and asserts the Prefer
    header contains both `return=representation` (so we get the row back)
    AND `resolution=merge-duplicates` (so ON CONFLICT DO UPDATE fires).
    """
    from postgrest import APIError

    from ogham.backends.supabase import SupabaseBackend

    backend = SupabaseBackend()
    client = backend._get_client()

    captured = {}

    def capture_request(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)

    client.session.event_hooks["request"].append(capture_request)

    # The request will fail with a network error since example.supabase.co
    # isn't a real endpoint, but the hook fires BEFORE the send, so we
    # still capture the request we wanted to send.
    memory = {
        "id": "00000000-0000-0000-0000-000000000001",
        "content": "regression test",
        "profile": "test",
        "metadata": {},
        "source": "test",
        "tags": [],
        "embedding": [0.1] * 512,
        # Both carried, so upsert_memory skips the existence lookup (TBU-253)
        # and the first request the hook captures is the upsert itself.
        "importance": 0.5,
        "surprise": 0.5,
    }
    with pytest.raises((Exception, APIError)):
        backend.upsert_memory(memory)

    # Both behaviours we depend on:
    assert "return=representation" in captured["headers"].get("prefer", ""), (
        "upsert dropped `return=representation` -- backend can't read the returned row"
    )
    assert "resolution=merge-duplicates" in captured["headers"].get("prefer", ""), (
        "upsert dropped `resolution=merge-duplicates` -- PostgREST will "
        "treat this as a plain INSERT and duplicate-key-fail on the "
        "second call. This is the v0.15.0 OKF re-import bug."
    )
    assert "on_conflict=id" in captured["url"], (
        "upsert dropped the on_conflict=id query parameter -- PostgREST "
        "won't know which column to resolve the conflict against."
    )
