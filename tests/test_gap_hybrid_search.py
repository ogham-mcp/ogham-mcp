from datetime import UTC, datetime, timedelta

import ogham.tools.memory as memtool


def _patch_search(monkeypatch, rows):
    monkeypatch.setattr(memtool, "get_active_profile", lambda: "work")
    from ogham import service

    monkeypatch.setattr(service, "search_memories_enriched", lambda **k: rows)
    monkeypatch.setattr(memtool.settings, "wiki_injection_enabled", False, raising=False)
    from ogham import flow_control

    monkeypatch.setattr(flow_control, "recall_enabled", lambda: True)


def test_gap_auto_silent_when_healthy(monkeypatch):
    # Relative, not a fixed date: a hardcoded "healthy" timestamp ages past
    # gap_stale_days (90) and turns this test red with no code change. It did,
    # on 2026-08-24, and public CI stayed red for six weeks.
    fresh = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    rows = [{"id": "a", "created_at": fresh, "confidence": 0.9, "tags": []}]
    _patch_search(monkeypatch, rows)
    out = memtool.hybrid_search("q", gap="auto")
    assert out["gap_note"] is None


def test_gap_auto_attaches_when_material(monkeypatch):
    rows = [{"id": "a", "created_at": "2026-01-01T00:00:00+00:00", "confidence": 0.3, "tags": []}]
    _patch_search(monkeypatch, rows)
    out = memtool.hybrid_search("q", gap="auto")
    assert out["gap_note"] is not None
    assert out["gap_note"]["low_confidence"]["below_floor"] == 1


def test_gap_off_returns_none(monkeypatch):
    rows = [{"id": "a", "created_at": "2026-01-01T00:00:00+00:00", "confidence": 0.3, "tags": []}]
    _patch_search(monkeypatch, rows)
    out = memtool.hybrid_search("q", gap="off")
    assert out["gap_note"] is None


def test_gap_suppressed_in_bench_mode(monkeypatch):
    rows = [{"id": "a", "created_at": "2026-01-01T00:00:00+00:00", "confidence": 0.3, "tags": []}]
    _patch_search(monkeypatch, rows)
    monkeypatch.setenv("OGHAM_BENCH_MODE", "true")
    out = memtool.hybrid_search("q", gap="auto")
    assert out["gap_note"] is None


def test_gap_prose_default_reuses_llm_model(monkeypatch):
    """With no gap_synthesis_* override, the prose path must reuse the wiki/
    recompute synthesis LLM (settings.llm_provider/llm_model), NOT send an
    empty model id -- empty 400s and silently falls back to the template (#262).
    """
    rows = [{"id": "a", "created_at": "2026-01-01T00:00:00+00:00", "confidence": 0.3, "tags": []}]
    _patch_search(monkeypatch, rows)
    monkeypatch.setattr(memtool.settings, "gap_synthesis_provider", "", raising=False)
    monkeypatch.setattr(memtool.settings, "gap_synthesis_model", "", raising=False)
    monkeypatch.setattr(memtool.settings, "llm_provider", "gemini", raising=False)
    monkeypatch.setattr(memtool.settings, "llm_model", "gemini-2.5-flash", raising=False)

    from ogham import gap_signals

    # deep lookup hits the DB; bypass it -- we're testing model resolution only.
    monkeypatch.setattr(gap_signals, "compute_deep_signals", lambda report, **k: report)
    captured = {}

    def fake_narrate(report, *, provider, model):
        captured["provider"] = provider
        captured["model"] = model
        return "heads up"

    monkeypatch.setattr(gap_signals, "narrate", fake_narrate)
    out = memtool.hybrid_search("q", gap="prose")

    assert captured["model"] == "gemini-2.5-flash"  # not "" (the bug)
    assert captured["provider"] == "gemini"
    assert out["gap_note"] == "heads up"
