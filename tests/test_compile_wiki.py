"""Unit tests for src/ogham/tools/wiki.py — the T1.1 compile_wiki MCP tool.

Wiki tool logic is thin orchestration over recompute_topic_summary +
get_summary_by_topic + the active-profile resolver. These tests mock
the underlying calls so we can exercise the wiring (response shape,
profile pass-through, error paths) without a live Postgres or LLM.

Postgres integration coverage already lives in test_recompute.py; the
wiki tool inherits that via the recompute_topic_summary call it wraps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest


def _make_summary_row(
    *,
    topic_key: str = "quantum",
    profile: str = "work",
    content: str = "## Overview\n\nQuantum stuff.\n",
    status: str = "fresh",
    version: int = 2,
) -> dict:
    return {
        "id": "11111111-2222-3333-4444-555555555555",
        "topic_key": topic_key,
        "profile_id": profile,
        "content": content,
        "embedding": [0.0] * 512,
        "source_count": 3,
        "source_cursor": "aaaa",
        "source_hash": bytes.fromhex("ab" * 32),
        "token_count": 200,
        "importance": 0.5,
        "model_used": "openai/gpt-4o-mini",
        "version": version,
        "status": status,
        "created_at": datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 4, 27, 10, 5, 0, tzinfo=timezone.utc),
        "stale_reason": None,
    }


def test_compile_wiki_no_sources_short_circuits():
    """Topic with no tagged memories returns no_sources without writing."""
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={"action": "no_sources", "profile": "work", "topic_key": "ghost"},
        ),
        patch.object(wiki, "get_summary_by_topic") as fetched,
    ):
        out = wiki.compile_wiki(topic="ghost")

    assert out["status"] == "no_sources"
    assert out["topic_key"] == "ghost"
    assert out["profile"] == "work"
    assert "ghost" in out["message"]
    fetched.assert_not_called()


def test_compile_wiki_recomputed_returns_stamped_markdown():
    """Happy path: recompute returns 'recomputed', tool fetches the row + stamps frontmatter."""
    from ogham.tools import wiki

    summary = _make_summary_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={"action": "recomputed", "summary_id": summary["id"], "source_count": 3},
        ),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.compile_wiki(topic="quantum")

    assert out["action"] == "recomputed"
    assert out["topic_key"] == "quantum"
    assert out["status"] == "fresh"
    assert out["version"] == 2
    assert out["source_hash"] == "ab" * 32
    assert out["model_used"] == "openai/gpt-4o-mini"

    # Stamped markdown carries YAML frontmatter + the body content.
    md = out["markdown"]
    assert md.startswith("---\n")
    assert f"ogham_id: {summary['id']}" in md
    assert "topic_key: quantum" in md
    assert "profile: work" in md
    assert "version: 2" in md
    assert "source_hash: " + ("ab" * 32) in md
    assert "## Overview" in md  # body preserved


def test_compile_wiki_skipped_action_returns_existing_row():
    """When recompute hash-matches and skips, we still return the existing row."""
    from ogham.tools import wiki

    summary = _make_summary_row(version=1)
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={
                "action": "skipped",
                "reason": "source_hash_match",
                "summary_id": summary["id"],
            },
        ),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.compile_wiki(topic="quantum")

    assert out["action"] == "skipped"
    assert out["version"] == 1
    assert "## Overview" in out["markdown"]


def test_compile_wiki_passes_provider_and_model_through():
    """Optional provider/model overrides reach the recompute call unchanged."""
    from ogham.tools import wiki

    summary = _make_summary_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={"action": "recomputed", "summary_id": summary["id"], "source_count": 3},
        ) as mock_recompute,
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        wiki.compile_wiki(
            topic="quantum",
            provider="gemini",
            model="gemini-2.5-flash",
        )

    mock_recompute.assert_called_once_with(
        profile="work",
        topic_key="quantum",
        provider="gemini",
        model="gemini-2.5-flash",
        force_oversize=False,
    )


def test_compile_wiki_recompute_succeeded_but_row_missing_returns_error():
    """Race condition guard: if the row vanishes between recompute + fetch, surface clearly."""
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={"action": "recomputed", "summary_id": "x", "source_count": 1},
        ),
        patch.object(wiki, "get_summary_by_topic", return_value=None),
    ):
        out = wiki.compile_wiki(topic="quantum")

    assert out["status"] == "error"
    assert "summary row was not found" in out["message"]


def test_query_topic_summary_not_cached_returns_explicit_status():
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=None),
    ):
        out = wiki.query_topic_summary(topic="ghost")

    assert out["status"] == "not_cached"
    assert out["topic_key"] == "ghost"
    assert "compile_wiki" in out["message"]  # nudge the user toward the right tool


def test_query_topic_summary_cached_returns_stamped_markdown():
    from ogham.tools import wiki

    summary = _make_summary_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum")

    assert out["topic_key"] == "quantum"
    assert out["status"] == "fresh"
    assert out["markdown"].startswith("---\n")
    assert "## Overview" in out["markdown"]


def test_query_topic_summary_does_not_trigger_recompute():
    """Read-only path must never call recompute_topic_summary."""
    from ogham.tools import wiki

    summary = _make_summary_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
        patch.object(wiki, "recompute_topic_summary") as recompute,
    ):
        wiki.query_topic_summary(topic="quantum")

    recompute.assert_not_called()


def test_format_summary_response_handles_missing_source_hash():
    """Defensive: a row with NULL source_hash (very old data) shouldn't crash."""
    from ogham.tools import wiki

    summary = _make_summary_row()
    summary["source_hash"] = None
    out = wiki._format_summary_response(summary)

    assert out["source_hash"] is None


# --------------------------------------------------------------------- #
# v0.13: progressive recall -- query_topic_summary level= parameter
# --------------------------------------------------------------------- #


def _make_three_form_row(**overrides):
    """Summary row with all three forms populated (post-033 schema)."""
    base = _make_summary_row()
    base["tldr_short"] = "One paragraph summary about quantum stuff."
    base["tldr_one_line"] = "Quantum stuff in one sentence."
    base.update(overrides)
    return base


def test_query_topic_summary_default_level_is_body():
    """Default level=body preserves v0.12 behaviour: full content returned."""
    from ogham.tools import wiki

    summary = _make_three_form_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum")

    assert out["level"] == "body"
    assert "## Overview" in out["markdown"]
    assert out["content"] == summary["content"]
    # No fallback fields when the requested level resolves cleanly.
    assert "fallback_reason" not in out


def test_query_topic_summary_level_short_returns_tldr_short():
    """level='short' returns the paragraph form."""
    from ogham.tools import wiki

    summary = _make_three_form_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum", level="short")

    assert out["level"] == "short"
    assert out["content"] == "One paragraph summary about quantum stuff."
    assert "level: short" in out["markdown"]


def test_query_topic_summary_level_one_line_returns_tldr_one_line():
    """level='one_line' returns the single sentence form."""
    from ogham.tools import wiki

    summary = _make_three_form_row()
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum", level="one_line")

    assert out["level"] == "one_line"
    assert out["content"] == "Quantum stuff in one sentence."


def test_query_topic_summary_short_falls_back_to_body_when_null():
    """Pre-033 rows have NULL TLDR fields. Falling back to body is back-compat."""
    from ogham.tools import wiki

    # Legacy row: only `content` populated; tldr_short / tldr_one_line absent.
    summary = _make_summary_row()
    # Don't add tldr_* keys at all -- simulates pre-033 row shape.
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum", level="short")

    # Falls back: served level is body, requested level reported separately.
    assert out["level"] == "body"
    assert out["requested_level"] == "short"
    assert "fallback_reason" in out
    assert out["content"] == summary["content"]


def test_query_topic_summary_one_line_falls_back_when_explicitly_null():
    """A row that *has* the column but with NULL value still falls back."""
    from ogham.tools import wiki

    summary = _make_three_form_row(tldr_one_line=None)
    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=summary),
    ):
        out = wiki.query_topic_summary(topic="quantum", level="one_line")

    assert out["level"] == "body"
    assert out["requested_level"] == "one_line"
    assert out["content"] == summary["content"]


def test_query_topic_summary_unknown_level_raises():
    """Defensive: typo in level= surfaces clearly rather than silently returning body."""
    from ogham.tools import wiki

    with patch.object(wiki, "get_active_profile", return_value="work"):
        with pytest.raises(ValueError, match="unknown level"):
            wiki.query_topic_summary(topic="x", level="bogus")  # type: ignore[arg-type]


def test_query_topic_summary_not_cached_does_not_consult_level():
    """When the row is missing entirely, level= is irrelevant."""
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=None),
    ):
        out = wiki.query_topic_summary(topic="ghost", level="short")

    assert out["status"] == "not_cached"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
