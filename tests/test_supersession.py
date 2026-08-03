"""Corrections must outrank what they supersede (TBU-207).

Measured on the live store before this landed: 7 of 8 queries returned the
correction missing from the top-10 or below the memories it explicitly
supersedes. The store had the `contradicts` edges the whole time, at full
strength, and reported them in `gap_note` -- ranking simply ignored them.

The rule under test is recency, not edge direction: a correction is written
after the thing it corrects, and all 148 `contradicts` edges in the live store
have the source strictly newer than the target.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ogham.supersession import apply_supersession, find_superseded

NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=30)


def _row(rid, created, relevance=0.5):
    return {"id": rid, "created_at": created, "relevance": relevance, "content": rid}


def _pair(in_id, other_id, other_created, strength=1.0):
    return {
        "in_result_id": in_id,
        "other_id": other_id,
        "other_created_at": other_created,
        "strength": strength,
    }


def test_older_contradicted_memory_is_marked():
    rows = [_row("stale", OLD)]
    found = find_superseded(rows, [_pair("stale", "correction", NOW)])
    assert found["stale"]["superseded_by"] == "correction"


def test_newer_memory_is_not_demoted_by_an_older_contradiction():
    """An older counterpart is a disagreement, not a correction."""
    rows = [_row("current", NOW)]
    assert find_superseded(rows, [_pair("current", "ancient", OLD)]) == {}


def test_equal_timestamps_are_left_alone():
    """Two peers written together -- guessing between them beats nothing badly."""
    rows = [_row("a", NOW)]
    assert find_superseded(rows, [_pair("a", "b", NOW)]) == {}


def test_superseded_row_sinks_below_untouched_rows():
    """The measured failure: the stale memory outranking its own correction."""
    rows = [_row("stale", OLD, relevance=0.9), _row("fine", NOW, relevance=0.1)]
    out = apply_supersession(rows, [_pair("stale", "correction", NOW)])

    assert [r["id"] for r in out] == ["fine", "stale"]
    assert out[-1]["superseded_by"] == "correction"


def test_superseded_rows_are_kept_not_dropped():
    """Silently removing a memory somebody stored is a bigger surprise."""
    rows = [_row("stale", OLD)]
    out = apply_supersession(rows, [_pair("stale", "correction", NOW)])
    assert len(out) == 1
    assert out[0]["id"] == "stale"


def test_untouched_rows_keep_their_relative_order():
    rows = [_row("a", NOW, 0.9), _row("b", NOW, 0.8), _row("c", NOW, 0.7)]
    out = apply_supersession(rows, [])
    assert [r["id"] for r in out] == ["a", "b", "c"]


def test_no_contradictions_returns_the_same_list_object():
    """The common path must cost nothing."""
    rows = [_row("a", NOW)]
    assert apply_supersession(rows, []) is rows


def test_newest_corrector_wins_when_several_contradict():
    later = NOW + timedelta(days=1)
    rows = [_row("stale", OLD)]
    found = find_superseded(rows, [_pair("stale", "first", NOW), _pair("stale", "second", later)])
    assert found["stale"]["superseded_by"] == "second"


def test_pairs_for_rows_not_in_the_result_are_ignored():
    rows = [_row("a", NOW)]
    assert find_superseded(rows, [_pair("somebody-else", "x", NOW)]) == {}


@pytest.mark.parametrize("bad", [None, "not-a-date", 12345, ""])
def test_unparseable_timestamps_never_demote(bad):
    """A malformed date must not silently reorder a result set."""
    rows = [_row("a", OLD)]
    assert find_superseded(rows, [_pair("a", "x", bad)]) == {}
    assert find_superseded([_row("a", bad)], [_pair("a", "x", NOW)]) == {}


def test_iso_strings_are_accepted():
    """Postgres returns datetimes, Supabase returns ISO strings. Both must work."""
    rows = [_row("stale", "2026-06-01T00:00:00Z")]
    found = find_superseded(rows, [_pair("stale", "fix", "2026-07-01T00:00:00+00:00")])
    assert found["stale"]["superseded_by"] == "fix"


def test_the_gateway_case_end_to_end():
    """The concrete failure from the audit, in the shape the store produced it.

    7341d555 ("the managed gateway is DEAD, prior memories are STALE") was
    written 2026-07-02. f42c6516, which it contradicts, is older and was
    ranking above it.
    """
    stale = _row("f42c6516", datetime(2026, 5, 1, tzinfo=timezone.utc), relevance=0.8)
    other = _row("unrelated", NOW, relevance=0.2)
    out = apply_supersession(
        [stale, other],
        [_pair("f42c6516", "7341d555", datetime(2026, 7, 2, tzinfo=timezone.utc))],
    )
    assert out[0]["id"] == "unrelated"
    assert out[-1]["id"] == "f42c6516"
    assert out[-1]["superseded_by"] == "7341d555"
