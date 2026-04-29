"""Tests for hook CLI routing."""

from unittest.mock import patch

from typer.testing import CliRunner

runner = CliRunner()


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
