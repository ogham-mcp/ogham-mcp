"""Tests for Prefab dashboard MCP tools and standalone server.

Skipped when the optional `prefab_ui` extra isn't installed -- those tests
exercise dashboard rendering that requires the FastMCP Prefab UI components.
The CLI still runs without it; the dashboard is opt-in.
"""

from unittest.mock import patch

import pytest

# pytest.importorskip raises Skip if the module isn't importable. Module-level
# skip means the whole file is skipped; we don't pretend the tests pass.
pytest.importorskip("prefab_ui", reason="dashboard tests require the [dashboard] extra")

MOCK_STATS = {
    "total": 100,
    "profile": "work",
    "sources": {"claude-code": 80, "cli": 20},
    "top_tags": [{"tag": "type:decision", "count": 15}],
    "relationships": {"orphan_count": 30},
    "tagging": {"untagged_count": 5, "distinct_tag_count": 42},
    "decay": {"eligible_count": 60, "floor_count": 3},
}

MOCK_AUDIT = [
    {
        "created_at": "2026-04-16T10:00:00",
        "operation": "store",
        "profile": "work",
        "resource_id": "abc12345-6789",
        "outcome": "success",
    }
]


@pytest.fixture
def mock_backend():
    with patch("ogham.database.get_backend") as mock:
        backend = mock.return_value
        backend.get_memory_stats.return_value = MOCK_STATS
        yield backend


@pytest.fixture
def mock_audit():
    with patch("ogham.database.query_audit_log", return_value=MOCK_AUDIT):
        yield


def test_build_profile_health(mock_backend):
    from ogham.tools.dashboard import _build_profile_health

    result = _build_profile_health("work")
    assert result is not None
    mock_backend.get_memory_stats.assert_called_once_with("work")


def test_build_audit_log(mock_audit):
    from ogham.tools.dashboard import _build_audit_log

    result = _build_audit_log("work", limit=10)
    assert result is not None


def test_dashboard_server_stats_api(mock_backend):
    from starlette.testclient import TestClient

    from ogham.dashboard_server import create_app

    client = TestClient(create_app(profile="work"))
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 100


def test_dashboard_server_root_html(mock_backend):
    from starlette.testclient import TestClient

    from ogham.dashboard_server import create_app

    client = TestClient(create_app(profile="work"))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Ogham" in resp.text


def test_dashboard_cli_help():
    from typer.testing import CliRunner

    from ogham.cli import app

    result = CliRunner().invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "port" in result.stdout
    assert "profile" in result.stdout
