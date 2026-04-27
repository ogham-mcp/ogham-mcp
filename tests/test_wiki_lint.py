"""Unit tests for src/ogham/wiki_lint.py + the lint_wiki MCP tool.

Backend calls are mocked so SQL shape and orchestration are validated
without a live DB. Postgres integration coverage will land in the
existing live-DB suite once we have a fixture profile primed with a
known mix of contradictions, orphans, and stale rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------- #
# find_contradictions
# --------------------------------------------------------------------- #


def test_find_contradictions_returns_count_and_sample():
    """SQL shape lives in migration 031's wiki_lint_contradictions function;
    end-to-end coverage in tests/test_wiki_integration.py. Here we just
    verify the dispatch passes through to the backend RPC method.
    """
    from ogham import wiki_lint

    sample_rows = [
        {
            "source_id": "mem-1",
            "target_id": "mem-2",
            "strength": 0.9,
            "created_at": datetime(2026, 4, 25, tzinfo=timezone.utc),
        }
    ]
    fake = MagicMock()
    fake.wiki_lint_contradictions.return_value = {"count": 1, "sample": sample_rows}
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_contradictions("work")

    assert out["count"] == 1
    assert out["sample"][0]["source_id"] == "mem-1"
    fake.wiki_lint_contradictions.assert_called_once_with("work", sample_size=10)


def test_find_contradictions_zero_when_no_rows():
    from ogham import wiki_lint

    fake = MagicMock()
    fake.wiki_lint_contradictions.return_value = {"count": 0, "sample": []}
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_contradictions("work")
    assert out["count"] == 0
    assert out["sample"] == []


# --------------------------------------------------------------------- #
# find_orphans
# --------------------------------------------------------------------- #


def test_find_orphans_passes_grace_minutes_to_backend():
    """5-minute grace window prevents just-stored memories from being
    flagged as orphans before auto_link finishes. The window itself is
    enforced by migration 031's wiki_lint_orphans function -- here we
    verify the dispatch passes the right value.
    """
    from ogham import wiki_lint

    fake = MagicMock()
    fake.wiki_lint_orphans.return_value = {"count": 0, "sample": []}
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        wiki_lint.find_orphans("work")

    fake.wiki_lint_orphans.assert_called_once_with("work", sample_size=10, grace_minutes=5)


def test_find_orphans_returns_count_and_sample():
    from ogham import wiki_lint

    sample_rows = [
        {
            "id": "lonely-1",
            "content": "no neighbours",
            "tags": ["solo"],
            "created_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
        }
    ]
    fake = MagicMock()
    fake.wiki_lint_orphans.return_value = {"count": 1, "sample": sample_rows}
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_orphans("work")
    assert out["count"] == 1
    assert out["sample"][0]["id"] == "lonely-1"


# --------------------------------------------------------------------- #
# find_stale_lifecycle
# --------------------------------------------------------------------- #


def test_find_stale_lifecycle_uses_stable_days_threshold():
    """The day threshold is enforced by migration 031's
    wiki_lint_stale_lifecycle. Here we verify the value reaches the
    backend RPC and round-trips into the response.
    """
    from ogham import wiki_lint

    fake = MagicMock()
    fake.wiki_lint_stale_lifecycle.return_value = {
        "count": 0,
        "sample": [],
        "older_than_days": 120,
    }
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_stale_lifecycle("work", older_than_days=120)

    fake.wiki_lint_stale_lifecycle.assert_called_once_with(
        "work", older_than_days=120, sample_size=10
    )
    assert out["older_than_days"] == 120


# --------------------------------------------------------------------- #
# find_stale_summaries
# --------------------------------------------------------------------- #


def test_find_stale_summaries_caps_sample_at_size():
    from ogham import wiki_lint

    rows = [
        {
            "id": f"sum-{i}",
            "topic_key": f"t-{i}",
            "version": 1,
            "stale_reason": "test",
            "updated_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
        }
        for i in range(20)
    ]
    with patch("ogham.wiki_lint.list_stale", return_value=rows):
        out = wiki_lint.find_stale_summaries("work", sample_size=5)
    assert out["count"] == 20
    assert len(out["sample"]) == 5
    assert out["sample"][0]["id"] == "sum-0"


# --------------------------------------------------------------------- #
# find_summary_drift
# --------------------------------------------------------------------- #


def test_find_summary_drift_no_fresh_summaries_returns_zero():
    from ogham import wiki_lint

    fake = MagicMock()
    fake.wiki_topic_list_fresh_for_drift.return_value = []
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_summary_drift("work")
    assert out["count"] == 0
    assert out["sample"] == []


def test_find_summary_drift_detects_hash_mismatch():
    """Stored hash differs from re-hash of current sources -> drift."""
    from ogham import wiki_lint
    from ogham.topic_summaries import compute_source_hash

    # Stored summary references {a, b}. Current state has {a, b, c} --
    # one new memory got tagged but the hook didn't fire.
    stored_hash = compute_source_hash(["a", "b"])
    fresh_summary_row = {
        "id": "sum-1",
        "topic_key": "topic-x",
        "source_hash": stored_hash,
    }

    fake = MagicMock()
    fake.wiki_topic_list_fresh_for_drift.return_value = [fresh_summary_row]
    fake.wiki_recompute_get_source_ids.return_value = ["a", "b", "c"]
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_summary_drift("work")

    assert out["count"] == 1
    assert out["sample"][0]["topic_key"] == "topic-x"
    assert out["sample"][0]["current_source_count"] == 3


def test_find_summary_drift_no_drift_when_hash_matches():
    from ogham import wiki_lint
    from ogham.topic_summaries import compute_source_hash

    stored_hash = compute_source_hash(["a", "b"])
    fresh_summary_row = {
        "id": "sum-1",
        "topic_key": "topic-x",
        "source_hash": stored_hash,
    }

    fake = MagicMock()
    fake.wiki_topic_list_fresh_for_drift.return_value = [fresh_summary_row]
    fake.wiki_recompute_get_source_ids.return_value = ["a", "b"]
    with patch("ogham.wiki_lint.get_backend", return_value=fake):
        out = wiki_lint.find_summary_drift("work")

    assert out["count"] == 0


# --------------------------------------------------------------------- #
# lint_report aggregator
# --------------------------------------------------------------------- #


def test_lint_report_aggregates_all_categories():
    from ogham import wiki_lint

    with (
        patch.object(wiki_lint, "find_contradictions", return_value={"count": 2, "sample": []}),
        patch.object(wiki_lint, "find_orphans", return_value={"count": 5, "sample": []}),
        patch.object(
            wiki_lint,
            "find_stale_lifecycle",
            return_value={"count": 1, "older_than_days": 90, "sample": []},
        ),
        patch.object(wiki_lint, "find_stale_summaries", return_value={"count": 0, "sample": []}),
        patch.object(wiki_lint, "find_summary_drift", return_value={"count": 3, "sample": []}),
    ):
        out = wiki_lint.lint_report("work")

    assert out["profile"] == "work"
    assert out["healthy"] is False
    assert out["issue_count"] == 2 + 5 + 1 + 0 + 3
    assert out["contradictions"]["count"] == 2
    assert out["orphans"]["count"] == 5


def test_lint_report_healthy_when_all_zero():
    from ogham import wiki_lint

    zero = {"count": 0, "sample": []}
    with (
        patch.object(wiki_lint, "find_contradictions", return_value=zero),
        patch.object(wiki_lint, "find_orphans", return_value=zero),
        patch.object(
            wiki_lint, "find_stale_lifecycle", return_value={**zero, "older_than_days": 90}
        ),
        patch.object(wiki_lint, "find_stale_summaries", return_value=zero),
        patch.object(wiki_lint, "find_summary_drift", return_value=zero),
    ):
        out = wiki_lint.lint_report("work")

    assert out["healthy"] is True
    assert out["issue_count"] == 0


def test_lint_report_skips_drift_when_disabled():
    from ogham import wiki_lint

    zero = {"count": 0, "sample": []}
    with (
        patch.object(wiki_lint, "find_contradictions", return_value=zero),
        patch.object(wiki_lint, "find_orphans", return_value=zero),
        patch.object(
            wiki_lint, "find_stale_lifecycle", return_value={**zero, "older_than_days": 90}
        ),
        patch.object(wiki_lint, "find_stale_summaries", return_value=zero),
        patch.object(wiki_lint, "find_summary_drift") as mock_drift,
    ):
        out = wiki_lint.lint_report("work", include_drift=False)

    mock_drift.assert_not_called()
    assert out["summary_drift"]["skipped"] is True


# --------------------------------------------------------------------- #
# lint_wiki MCP tool
# --------------------------------------------------------------------- #


def test_lint_wiki_uses_active_profile():
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki, "lint_report", return_value={"profile": "work", "healthy": True, "issue_count": 0}
        ) as mock_lint,
    ):
        out = wiki.lint_wiki()

    mock_lint.assert_called_once_with(
        profile="work",
        stable_days=90,
        sample_size=10,
        include_drift=True,
    )
    assert out["profile"] == "work"


def test_lint_wiki_passes_args_through():
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="personal"),
        patch.object(wiki, "lint_report", return_value={"profile": "personal"}) as mock_lint,
    ):
        wiki.lint_wiki(stable_days=180, sample_size=25, include_drift=False)

    mock_lint.assert_called_once_with(
        profile="personal",
        stable_days=180,
        sample_size=25,
        include_drift=False,
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
