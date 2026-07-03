"""import_linear tool -- unit tests with mocked client + service."""

from unittest.mock import MagicMock

import pytest


def test_import_linear_stores_issues_and_dedupes():
    from ogham.tools.import_linear import import_linear_impl

    client = MagicMock()
    client.fetch_issues.return_value = [
        {
            "id": "u1",
            "identifier": "TBU-1",
            "title": "First",
            "description": "",
            "state": {"name": "In Progress"},
            "priority": 2,
            "labels": {"nodes": [{"name": "Bug"}]},
            "comments": {"nodes": []},
        },
        {
            "id": "u2",
            "identifier": "TBU-2",
            "title": "Second",
            "description": "",
            "state": {"name": "Done"},
            "priority": 3,
            "labels": {"nodes": []},
            "comments": {"nodes": []},
        },
    ]
    service = MagicMock()
    # u2 already exists (fetched once via fetch_all_tracker_ids); u1 is new.
    service.fetch_all_tracker_ids.return_value = {"u2"}
    service.store_memory.return_value = {"id": "new-uuid"}

    result = import_linear_impl(
        client=client,
        service=service,
        team_key="TBU",
        since_days=30,
        profile="work",
    )
    assert result == {"imported": 1, "skipped": 1, "disabled": 0}
    assert service.store_memory.call_count == 1
    service.fetch_all_tracker_ids.assert_called_once_with("work")


def test_import_linear_skips_when_issue_already_exists():
    """Dedicated skip-path test: single issue, already imported -> 0 imported, 1 skipped."""
    from ogham.tools.import_linear import import_linear_impl

    client = MagicMock()
    client.fetch_issues.return_value = [
        {
            "id": "u1",
            "identifier": "TBU-1",
            "title": "First",
            "description": "",
            "state": {"name": "In Progress"},
            "priority": 2,
            "labels": {"nodes": []},
            "comments": {"nodes": []},
        },
    ]
    service = MagicMock()
    service.fetch_all_tracker_ids.return_value = {"u1"}

    result = import_linear_impl(
        client=client,
        service=service,
        team_key="TBU",
        since_days=30,
        profile="work",
    )
    assert result == {"imported": 0, "skipped": 1, "disabled": 0}
    service.store_memory.assert_not_called()


def test_import_linear_respects_disabled_status():
    """An operator with inscribe disabled must see a `disabled` count, not a false `imported`."""
    from ogham.tools.import_linear import import_linear_impl

    client = MagicMock()
    client.fetch_issues.return_value = [
        {
            "id": "u1",
            "identifier": "TBU-1",
            "title": "X",
            "description": "",
            "state": {"name": "In Progress"},
            "priority": 2,
            "labels": {"nodes": []},
            "comments": {"nodes": []},
        },
    ]
    service = MagicMock()
    service.fetch_all_tracker_ids.return_value = set()
    service.store_memory.return_value = {"status": "disabled", "tool": "inscribe"}

    result = import_linear_impl(
        client=client,
        service=service,
        team_key="TBU",
        since_days=30,
        profile="work",
    )
    assert result == {"imported": 0, "skipped": 0, "disabled": 1}


def test_import_linear_falls_back_to_per_issue_lookup_without_batch_scan():
    """Services that only implement find_by_metadata_kv (no fetch_all_tracker_ids) still work."""
    from ogham.tools.import_linear import import_linear_impl

    client = MagicMock()
    client.fetch_issues.return_value = [
        {
            "id": "u1",
            "identifier": "TBU-1",
            "title": "First",
            "description": "",
            "state": {"name": "In Progress"},
            "priority": 2,
            "labels": {"nodes": []},
            "comments": {"nodes": []},
        },
        {
            "id": "u2",
            "identifier": "TBU-2",
            "title": "Second",
            "description": "",
            "state": {"name": "Done"},
            "priority": 3,
            "labels": {"nodes": []},
            "comments": {"nodes": []},
        },
    ]

    class _MinimalService:
        def find_by_metadata_kv(self, key, value, profile):
            return {"id": "existing-uuid"} if value == "u2" else None

        def store_memory(self, content, metadata, tags, profile):
            return {"id": "new-uuid"}

    result = import_linear_impl(
        client=client,
        service=_MinimalService(),
        team_key="TBU",
        since_days=30,
        profile="work",
    )
    assert result == {"imported": 1, "skipped": 1, "disabled": 0}


def test_import_linear_tool_raises_when_token_missing(monkeypatch):
    """MCP tool must fail cleanly (not silently no-op) when LINEAR_API_TOKEN is unset."""
    from ogham.tools.import_linear import import_linear

    monkeypatch.delenv("LINEAR_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="LINEAR_API_TOKEN"):
        import_linear(team_key="TBU")
