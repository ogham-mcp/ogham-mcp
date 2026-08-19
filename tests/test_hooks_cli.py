"""Tests for hook CLI routing."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def clear_flow_overrides():
    from ogham.flow_control import clear_flow_overrides

    clear_flow_overrides()
    yield
    clear_flow_overrides()


def test_hooks_inscribe_dry_run_previews_tool_memory():
    from ogham.hooks_cli import hooks_app

    data = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "/src/ogham/dashboard.py",
            "old_string": "PANELS = []",
            "new_string": "PANELS = ['overview']",
        },
        "cwd": "/Users/dev/ogham-mcp",
        "session_id": "s1",
    }

    with patch("ogham.hooks_cli._read_stdin", return_value=data):
        result = runner.invoke(hooks_app, ["inscribe", "--dry-run"])

    assert result.exit_code == 0
    assert "dashboard.py" in result.output
    assert "[ogham-mcp]" in result.output


def test_hooks_inscribe_dry_run_routes_user_prompt_submit():
    from ogham.hooks_cli import hooks_app

    data = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "I prefer PostgreSQL over MySQL for this project",
        "cwd": "/Users/dev/ogham-mcp",
        "session_id": "s1",
    }

    with patch("ogham.hooks_cli._read_stdin", return_value=data):
        result = runner.invoke(hooks_app, ["inscribe", "--dry-run"])

    assert result.exit_code == 0
    assert "I prefer PostgreSQL over MySQL for this project [ogham-mcp]" in result.output


def test_hooks_inscribe_dry_run_reports_skipped_memory():
    from ogham.hooks_cli import hooks_app

    data = {
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "cwd": "/Users/dev/ogham-mcp",
        "session_id": "s1",
    }

    with patch("ogham.hooks_cli._read_stdin", return_value=data):
        result = runner.invoke(hooks_app, ["inscribe", "--dry-run"])

    assert result.exit_code == 0
    assert "No memory would be stored." in result.output


def test_hooks_recall_no_recall_skips_hooks():
    from ogham.hooks_cli import hooks_app

    with (
        patch("ogham.hooks.session_start") as session_start,
        patch("ogham.hooks.post_compact") as post_compact,
        patch("ogham.hooks_cli._should_recall", return_value=True),
    ):
        result = runner.invoke(hooks_app, ["recall", "--no-recall"])

    assert result.exit_code == 0
    session_start.assert_not_called()
    post_compact.assert_not_called()


def test_hooks_inscribe_no_inscribe_skips_hooks():
    from ogham.hooks_cli import hooks_app

    with (
        patch("ogham.hooks.post_tool") as post_tool,
        patch("ogham.hooks.pre_compact") as pre_compact,
    ):
        result = runner.invoke(hooks_app, ["inscribe", "--no-inscribe"])

    assert result.exit_code == 0
    post_tool.assert_not_called()
    pre_compact.assert_not_called()


def test_install_scopes_post_tool_use_to_edit_and_write(tmp_path, monkeypatch):
    """TBU-231. A match-all PostToolUse matcher fired on every Bash call and made
    65% of the store command noise. The installer must write an explicit tool
    list, not "".

    Claude Code evaluates a matcher containing only letters, digits, `_`, `-`,
    spaces, `,` and `|` as an exact string or |-separated list of exact tool
    names, so "Edit|Write" matches those two tools and nothing else.
    """
    import json

    from ogham import hooks_install

    monkeypatch.setattr(hooks_install.Path, "home", staticmethod(lambda: tmp_path))
    hooks_install._install_claude_code()

    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    post_tool_use = settings["hooks"]["PostToolUse"]
    matchers = [entry["matcher"] for entry in post_tool_use]
    assert matchers == ["Edit|Write"], f"expected a scoped matcher, got {matchers!r}"

    # The recall-side events are not tool-scoped and must stay match-all.
    for event in ("SessionStart", "PreCompact", "PostCompact"):
        assert [e["matcher"] for e in settings["hooks"][event]] == [""], (
            f"{event} is not a tool event -- it should stay match-all"
        )
