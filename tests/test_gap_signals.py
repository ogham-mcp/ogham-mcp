from datetime import datetime, timezone

import ogham.gap_signals as gs
from ogham.gap_signals import GapReport, compute_in_result_signals, is_material, render_template


def _row(id_, *, created_at, confidence):
    return {
        "id": id_,
        "content": "x",
        "created_at": created_at,
        "confidence": confidence,
        "tags": [],
    }


def test_in_result_signals_staleness_and_confidence():
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [
        _row("a", created_at="2026-05-20T00:00:00+00:00", confidence=0.4),
        _row("b", created_at="2026-01-01T00:00:00+00:00", confidence=0.45),
    ]
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)

    assert isinstance(report, GapReport)
    assert report.staleness["newest_write_days"] == 7  # most recent = 2026-05-20
    assert report.staleness["older_than_stale_days"] == 1  # only "b" is >90d
    assert report.low_confidence["below_floor"] == 2
    assert report.low_confidence["max_confidence"] == 0.45
    assert report.scope_size == 2


def test_in_result_signals_empty_scope():
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    report = compute_in_result_signals([], now=now, stale_days=90, confidence_floor=0.5)
    assert report.scope_size == 0
    assert report.staleness["newest_write_days"] is None
    assert report.low_confidence["below_floor"] == 0


def test_in_result_signals_clamps_future_dates():
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [_row("a", created_at="2026-06-30T00:00:00+00:00", confidence=0.9)]  # future
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)
    assert report.staleness["newest_write_days"] == 0  # clamped, never negative


def test_in_result_signals_max_confidence_is_result_set_max():
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [
        _row("a", created_at="2026-05-26T00:00:00+00:00", confidence=0.3),
        _row("b", created_at="2026-05-26T00:00:00+00:00", confidence=0.8),  # above floor
    ]
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)
    assert report.low_confidence["below_floor"] == 1  # only "a"
    assert report.low_confidence["max_confidence"] == 0.8  # whole-set max, not below-floor max


def test_in_result_signals_skips_non_numeric_confidence_and_null_dates():
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [
        {"id": "a", "content": "x", "created_at": None, "confidence": "high", "tags": []},
        {
            "id": "b",
            "content": "x",
            "created_at": "2026-05-20T00:00:00+00:00",
            "confidence": 0.4,
            "tags": [],
        },
    ]
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)
    assert report.scope_size == 2
    assert report.staleness["newest_write_days"] == 7  # only "b" parsed
    assert report.low_confidence["below_floor"] == 1  # "high" skipped, 0.4 counts


def test_is_material_true_on_low_confidence():
    r = GapReport(scope_size=2, low_confidence={"below_floor": 1, "confidence_floor": 0.5})
    assert is_material(r) is True


def test_is_material_true_on_stale():
    r = GapReport(scope_size=2, staleness={"newest_write_days": 120, "stale_days_threshold": 90})
    assert is_material(r) is True


def test_is_material_true_on_contradiction():
    r = GapReport(scope_size=2, contradictions={"count": 1, "pairs": [["a", "b"]]})
    assert is_material(r) is True


def test_is_material_false_when_healthy():
    r = GapReport(
        scope_size=2,
        staleness={"newest_write_days": 3, "stale_days_threshold": 90},
        low_confidence={"below_floor": 0, "confidence_floor": 0.5},
        contradictions={"count": 0, "pairs": []},
    )
    assert is_material(r) is False


def test_render_template_mentions_only_present_facts():
    r = GapReport(
        scope_size=3,
        staleness={"newest_write_days": 55, "older_than_stale_days": 3, "stale_days_threshold": 30},
        low_confidence={"below_floor": 3, "confidence_floor": 0.5, "max_confidence": 0.45},
        contradictions={"count": 1, "pairs": [["a", "b"]]},
    )
    text = render_template(r)
    assert "55" in text
    assert "1" in text
    assert "confiden" in text.lower()
    assert "you don't know" not in text.lower()


def test_render_template_healthy_is_empty():
    r = GapReport(scope_size=2, contradictions={"count": 0, "pairs": []})
    assert render_template(r) == ""


def test_narrate_uses_llm_and_falls_back(monkeypatch):
    r = GapReport(
        scope_size=3,
        staleness={"newest_write_days": 55, "older_than_stale_days": 3, "stale_days_threshold": 30},
        contradictions={"count": 1, "pairs": [["a", "b"]]},
        low_confidence={"below_floor": 3, "confidence_floor": 0.5, "max_confidence": 0.45},
    )
    calls = {}

    def fake_synthesize(prompt, *, provider, model, **kw):
        calls["prompt"] = prompt
        calls["max_tokens"] = kw.get("max_tokens")
        return "Heads up: 1 contradiction; newest write 55 days old."

    monkeypatch.setattr(gs, "synthesize", fake_synthesize)
    out = gs.narrate(r, provider="gemini", model="gemini-x")
    assert "55" in out and "1" in out
    assert "55" in calls["prompt"]
    # Budget must leave room for "thinking" models or Gemini 2.5 Flash truncates
    # the answer to a fragment (verified live, #262). Guard against a regression.
    assert calls["max_tokens"] >= 1024

    def boom(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr(gs, "synthesize", boom)
    out2 = gs.narrate(r, provider="gemini", model="gemini-x")
    assert out2 == render_template(r)


def test_narrate_healthy_returns_empty(monkeypatch):
    r = GapReport(scope_size=2, contradictions={"count": 0, "pairs": []})
    monkeypatch.setattr(gs, "synthesize", lambda *a, **k: "should not be called")
    assert gs.narrate(r, provider="gemini", model="gemini-x") == ""


def test_gap_settings_defaults():
    from ogham.config import settings

    assert settings.gap_stale_days == 90
    assert settings.gap_confidence_floor == 0.5
    assert settings.gap_synthesis_provider == ""


def test_compute_deep_signals_adds_contradictions(monkeypatch):
    r = GapReport(scope_size=2)
    monkeypatch.setattr(
        gs,
        "gap_out_of_result_contradictions",
        lambda profile, ids, sample_size=10: {
            "count": 1,
            "pairs": [{"in_result_id": "a", "other_id": "c", "strength": 0.9}],
        },
    )
    out = gs.compute_deep_signals(r, profile="work", result_ids=["a", "b"])
    assert out.contradictions["count"] == 1


def test_compute_deep_signals_coverage_degrades(monkeypatch):
    r = GapReport(scope_size=2)
    monkeypatch.setattr(
        gs, "gap_out_of_result_contradictions", lambda *a, **k: {"count": 0, "pairs": []}
    )

    def boom(*a, **k):
        raise RuntimeError('relation "topic_summaries" does not exist')

    monkeypatch.setattr(gs, "_coverage_lookup", boom)
    out = gs.compute_deep_signals(r, profile="work", result_ids=["a"])
    assert out.coverage == {"note": "wiki not enabled"}


def test_parse_dt_accepts_datetime_object():
    """TBU-162: Postgres backend (psycopg) returns created_at as a datetime,
    not a str. _parse_dt must tolerate it instead of raising TypeError."""
    dt = datetime(2026, 5, 20, 12, 30, tzinfo=timezone.utc)
    out = gs._parse_dt(dt)
    assert out == dt


def test_parse_dt_datetime_object_naive_becomes_utc_aware():
    dt = datetime(2026, 5, 20, 12, 30)  # naive
    out = gs._parse_dt(dt)
    assert out is not None
    assert out.tzinfo == timezone.utc


def test_parse_dt_accepts_iso_string_with_offset():
    out = gs._parse_dt("2026-05-20T00:00:00+00:00")
    assert out == datetime(2026, 5, 20, tzinfo=timezone.utc)


def test_parse_dt_accepts_iso_string_with_trailing_z():
    out = gs._parse_dt("2026-05-20T00:00:00Z")
    assert out == datetime(2026, 5, 20, tzinfo=timezone.utc)


def test_parse_dt_returns_none_for_none_and_garbage():
    assert gs._parse_dt(None) is None
    assert gs._parse_dt("not-a-date") is None
    assert gs._parse_dt(12345) is None  # type: ignore[arg-type]


def test_compute_in_result_signals_tolerates_datetime_created_at_rows():
    """Reproduces the real TBU-162 crash: rows shaped like the Postgres
    backend returns them (created_at is a datetime object, not a string).
    Must not raise, and must still produce staleness signals."""
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [
        _row("a", created_at=datetime(2026, 5, 20, tzinfo=timezone.utc), confidence=0.4),
        _row("b", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), confidence=0.45),
    ]
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)
    assert report.staleness["newest_write_days"] == 7
    assert report.staleness["older_than_stale_days"] == 1
    assert report.scope_size == 2


def test_compute_in_result_signals_tolerates_mixed_str_and_datetime_rows():
    """A hybrid_search result set could plausibly mix backends in theory;
    at minimum the function must not care which shape each row is in."""
    now = datetime(2026, 5, 27, tzinfo=timezone.utc)
    rows = [
        _row("a", created_at=datetime(2026, 5, 20, tzinfo=timezone.utc), confidence=0.4),
        _row("b", created_at="2026-01-01T00:00:00+00:00", confidence=0.45),
    ]
    report = compute_in_result_signals(rows, now=now, stale_days=90, confidence_floor=0.5)
    assert report.staleness["newest_write_days"] == 7
    assert report.scope_size == 2


def test_compute_deep_signals_no_result_ids_skips_lookup(monkeypatch):
    r = GapReport(scope_size=0)
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return {"count": 0, "pairs": []}

    monkeypatch.setattr(gs, "gap_out_of_result_contradictions", spy)
    monkeypatch.setattr(gs, "_coverage_lookup", lambda *a, **k: {})
    gs.compute_deep_signals(r, profile="work", result_ids=[])
    assert called["n"] == 0  # no contradiction lookup when there are no result ids
