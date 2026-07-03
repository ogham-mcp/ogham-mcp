"""TBU-162: temporal-boost path must tolerate datetime-typed row dates.

On the Postgres backend, `created_at` comes back as a real `datetime` object
(not an ISO string, as the Supabase REST backend returns). `_extract_memory_date`
already coerces its `created_at` fallback to a 10-char date string via
`str(...)[:10]`, so today `_temporal_rerank` never actually receives a raw
`datetime` through that path -- but the block is made defensively
datetime-aware anyway so a future change to `_extract_memory_date` (or any
other row source) can't silently regress into the
`except (ValueError, TypeError): continue` swallow.
"""

from datetime import datetime, timezone

import ogham.service as svc


def test_extract_memory_date_coerces_datetime_created_at_to_date_string():
    """Confirms today's actual contract: _extract_memory_date always returns
    a plain str (or None), even when created_at is a real datetime object.
    Other callers (e.g. the ordering-query sort key at service.py ~864, which
    mixes this return value with a "9999" string sentinel) rely on this --
    widening the return type to leak a raw datetime would break those sites.
    """
    row = {
        "created_at": datetime(2026, 5, 20, 12, 30, tzinfo=timezone.utc),
        "metadata": {},
        "content": "no embedded date here",
    }
    result = svc._extract_memory_date(row)
    assert result == "2026-05-20"
    assert isinstance(result, str)


def test_temporal_rerank_boosts_anchor_date_when_created_at_is_datetime(monkeypatch):
    """Reproduces the Postgres-backend row shape end-to-end through
    _temporal_rerank and asserts the boost is actually applied -- not
    silently skipped by the try/except safety net."""
    monkeypatch.setattr(
        svc,
        "resolve_temporal_query",
        lambda query: ("2026-05-20T00:00:00", "2026-05-20T00:00:00"),
    )
    results = [
        {
            "id": "a",
            "content": "on the anchor date",
            "metadata": {},
            "created_at": datetime(2026, 5, 20, 9, 0, tzinfo=timezone.utc),
            "relevance": 0.5,
        },
        {
            "id": "b",
            "content": "far from the anchor date",
            "metadata": {},
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "relevance": 0.5,
        },
    ]

    out = svc._temporal_rerank(results, "memories from that day")

    by_id = {r["id"]: r["relevance"] for r in out}
    assert by_id["a"] > 0.5  # boosted
    assert by_id["a"] > by_id["b"]


def test_temporal_rerank_defensive_against_raw_datetime_from_extract_memory_date(monkeypatch):
    """Defensive guard: if _extract_memory_date's contract ever changes to
    surface a raw datetime, _temporal_rerank must use it directly rather than
    crash and have the boost silently swallowed."""
    monkeypatch.setattr(
        svc,
        "resolve_temporal_query",
        lambda query: ("2026-05-20T00:00:00", "2026-05-20T00:00:00"),
    )
    monkeypatch.setattr(
        svc, "_extract_memory_date", lambda r: datetime(2026, 5, 20, tzinfo=timezone.utc)
    )
    results = [{"id": "a", "relevance": 0.5}]

    out = svc._temporal_rerank(results, "memories from that day")

    assert out[0]["relevance"] > 0.5  # same-day grace boost applied, not skipped
