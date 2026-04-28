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
