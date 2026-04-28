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
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -m 'feat: add dashboard'"},
        "cwd": "/Users/dev/ogham-mcp",
        "session_id": "s1",
    }

    with patch("ogham.hooks_cli._read_stdin", return_value=data):
        result = runner.invoke(hooks_app, ["inscribe", "--dry-run"])

    assert result.exit_code == 0
    assert "git commit: feat: add dashboard [ogham-mcp]" in result.output


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
