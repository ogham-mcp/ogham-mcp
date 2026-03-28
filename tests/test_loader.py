"""Tests for ogham.data.loader -- language data loading with caching."""

from ogham.data.loader import (
    _load_language_file,
    get_all_day_names,
    get_day_names,
    get_query_hints,
    invalidate_cache,
)


def test_load_english():
    """en.yaml loads and day_names has 7 entries (Sun-Sat)."""
    invalidate_cache()
    days = get_day_names("en")
    assert len(days) == 7
    assert days["monday"] == 1
    assert days["sunday"] == 0
    assert days["saturday"] == 6


def test_fallback_to_english():
    """Requesting a nonexistent language falls back to English data."""
    invalidate_cache()
    days = get_day_names("xx")
    assert len(days) == 7
    assert days["monday"] == 1


def test_cache_hit():
    """Second call uses cache -- verify via lru_cache stats."""
    invalidate_cache()
    _load_language_file("en")
    _load_language_file("en")
    info = _load_language_file.cache_info()
    assert info.hits >= 1


def test_invalidate_cache():
    """Cache clears correctly -- misses reset after invalidation."""
    _load_language_file("en")
    invalidate_cache()
    info = _load_language_file.cache_info()
    assert info.currsize == 0


def test_get_all_day_names():
    """get_all_day_names returns merged dict from all available languages."""
    invalidate_cache()
    all_days = get_all_day_names()
    # Must include English entries at minimum
    assert "monday" in all_days
    assert all_days["monday"] == 1
    # Should have at least 7 entries (English alone)
    assert len(all_days) >= 7


def test_get_query_hints():
    """get_query_hints returns list of hint strings."""
    invalidate_cache()
    multi_hop = get_query_hints("en", "multi_hop")
    assert isinstance(multi_hop, list)
    assert len(multi_hop) > 0
    assert "how many" in multi_hop

    # All hints (no type filter)
    all_hints = get_query_hints("en")
    assert isinstance(all_hints, list)
    assert len(all_hints) >= len(multi_hop)
    assert "chronological" in all_hints
