"""Recall envelope (TBU-295) and file-edit capture policy (TBU-288).

The fixture is shared with the Go client; both must render identical bytes.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ogham.hooks import format_recall_envelope, post_compact, post_tool, session_start

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "recall_envelope_cases.json").read_text(encoding="utf-8")
)["cases"]
_LIMITS = {"session-start": 8, "post-compact": 10}


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_envelope_matches_fixture(case):
    out = format_recall_envelope(
        case["rows"], case["event"], case["project"], case["profile"], _LIMITS[case["event"]]
    )
    assert out == case["expected"]
    assert len(out) <= 2400


def test_fixture_covers_required_spec_cases():
    names = {c["name"] for c in _CASES}
    assert {
        "hook_rows_filtered",
        "superseded_rows_filtered",
        "all_filtered_empty",
        "multibyte_at_truncation_200",
        "newlines_and_whitespace_collapsed",
        "missing_created_at",
        "cap_2400_drops_whole_lines",
    } <= names


def test_created_at_datetime_object_is_accepted():
    from datetime import UTC, datetime

    row = {"content": "x", "created_at": datetime(2026, 11, 10, 1, 0, tzinfo=UTC), "source": "s"}
    out = format_recall_envelope([row], "session-start", "p", "work", 8)
    assert "- [2026-11-10] x · s\n" in out


def _hook_rows():
    return [
        {"content": "Edit: /p/a.py", "source": "hook:post-tool", "tags": []},
        {"content": "kept", "source": "claude-code", "tags": [], "created_at": "2026-11-10"},
    ]


def test_session_start_overfetches_and_renders_envelope():
    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=_hook_rows()) as search,
        patch("ogham.hooks.lifecycle_submit"),
    ):
        out = session_start(cwd="/tmp/proj", profile="work")
    assert search.call_args.kwargs["limit"] == 24
    assert out.startswith("⟦ogham⟧") and out.endswith("⟦/ogham⟧\n")
    assert "kept" in out and "Edit:" not in out
    assert "trigger: session-start · project=proj" in out


def test_post_compact_overfetches_and_renders_envelope():
    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=_hook_rows()) as search,
    ):
        out = post_compact(cwd="/tmp/proj", profile="work")
    assert search.call_args.kwargs["limit"] == 30
    assert "trigger: post-compact · project=proj" in out


def test_session_start_search_failure_returns_empty():
    with (
        patch("ogham.embeddings.generate_embedding", side_effect=RuntimeError("boom")),
        patch("ogham.hooks.lifecycle_submit"),
    ):
        assert session_start(cwd="/tmp/proj") == ""


def test_session_start_all_filtered_returns_empty():
    rows = [{"content": "x", "source": "hook:post-tool"}]
    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=rows),
        patch("ogham.hooks.lifecycle_submit"),
    ):
        assert session_start(cwd="/tmp/proj") == ""


# --- Capture policy (TBU-288) ---

_EDIT_INPUTS = {
    "Edit": {
        "file_path": "/p/src/config.py",
        "old_string": "MAX_RETRIES = 3",
        "new_string": "MAX_RETRIES = 5",
    },
    "Write": {
        "file_path": "/p/src/new_module.py",
        "content": '"""Brand new module docstring."""\n',
    },
    "MultiEdit": {
        "file_path": "/p/src/config.py",
        "edits": [{"old_string": "MAX_RETRIES = 3", "new_string": "MAX_RETRIES = 5"}],
    },
}


def _run_post_tool(tool, enabled):
    hook_input = {
        "tool_name": tool,
        "tool_input": _EDIT_INPUTS[tool],
        "tool_response": "File created successfully",
        "cwd": "/p",
        "session_id": "s1",
    }
    with (
        patch("ogham.hooks._capture_file_edits", return_value=enabled),
        patch("ogham.hooks._get_routine_tools", return_value=frozenset({"MultiEdit"})),
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
    ):
        return post_tool(hook_input, dry_run=True)


@pytest.mark.parametrize("tool", ["Edit", "Write", "MultiEdit"])
def test_file_edits_not_captured_by_default(tool):
    assert _run_post_tool(tool, enabled=False) is None


@pytest.mark.parametrize("tool", ["Edit", "Write"])
def test_file_edits_captured_when_enabled(tool):
    assert _run_post_tool(tool, enabled=True) is not None


def test_capture_switch_default_is_off():
    from ogham.hooks import _capture_file_edits

    assert _capture_file_edits() is False


def test_other_capture_paths_unaffected_by_switch():
    from ogham.hooks import _extract_memory_content

    with patch("ogham.hooks._capture_file_edits", return_value=False):
        got = _extract_memory_content(
            "UserPromptSubmit", "", {}, "", "/p", prompt="I always prefer tabs over spaces"
        )
    assert got is not None


# --- Supersession reaches the hooks (they call database.* directly) ---


def _stale_and_newer_rows():
    return [
        {
            "id": "old-1",
            "content": "STALE gateway runs in three regions",
            "source": "claude-code",
            "tags": [],
            "created_at": "2026-04-22T10:00:00Z",
        },
        {
            "id": "new-1",
            "content": "FRESH gateway was shut down",
            "source": "claude-code",
            "tags": [],
            "created_at": "2026-08-01T10:00:00Z",
        },
    ]


def _supersession_patches():
    return (
        patch(
            "ogham.database.in_result_contradictions",
            return_value=[{"stale_id": "old-1", "newer_id": "new-1", "strength": 1.0}],
        ),
        patch("ogham.database.gap_out_of_result_contradictions", return_value={"pairs": []}),
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=_stale_and_newer_rows()),
    )


def test_session_start_drops_superseded_rows():
    p1, p2, p3, p4 = _supersession_patches()
    with p1, p2, p3, p4, patch("ogham.hooks.lifecycle_submit"):
        out = session_start(cwd="/tmp/proj", profile="work")
    assert "FRESH gateway" in out
    assert "STALE gateway" not in out


def test_post_compact_drops_superseded_rows():
    p1, p2, p3, p4 = _supersession_patches()
    with p1, p2, p3, p4:
        out = post_compact(cwd="/tmp/proj", profile="work")
    assert "FRESH gateway" in out
    assert "STALE gateway" not in out
