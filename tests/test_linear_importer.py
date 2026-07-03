"""Linear importer tests -- mapper + client, no live API calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock

FIXTURE = Path(__file__).parent / "fixtures" / "linear_sample.json"


def _load_issue():
    data = json.loads(FIXTURE.read_text())
    return data["data"]["issue"]


def test_map_issue_to_memory_content_includes_title_and_body():
    from ogham.importers.linear import map_issue_to_memory

    issue = _load_issue()
    memory = map_issue_to_memory(issue)
    assert issue["title"] in memory["content"]
    assert issue["description"] in memory["content"]


def test_map_issue_to_memory_metadata_carries_tracker_external_id():
    from ogham.importers.linear import map_issue_to_memory

    issue = _load_issue()
    memory = map_issue_to_memory(issue)
    assert memory["metadata"]["tracker_external_id"] == issue["id"]
    assert memory["metadata"]["identifier"] == issue["identifier"]


def test_map_issue_to_memory_tags_include_linear_labels():
    from ogham.importers.linear import map_issue_to_memory

    issue = _load_issue()
    memory = map_issue_to_memory(issue)
    for label in issue.get("labels", {}).get("nodes", []):
        assert f"linear:{label['name']}" in memory["tags"]


def test_map_issue_appends_comments_to_content():
    from ogham.importers.linear import map_issue_to_memory

    issue = _load_issue()
    memory = map_issue_to_memory(issue)
    for comment in issue.get("comments", {}).get("nodes", []):
        assert comment["body"] in memory["content"]


def test_client_authenticates_with_bearer_token():
    from ogham.importers.linear import LinearClient

    http = MagicMock()
    http.post.return_value.status_code = 200
    http.post.return_value.json.return_value = {"data": {"issues": {"nodes": []}}}

    client = LinearClient(token="fake-token", http_client=http)
    client.fetch_issues("TBU", since_days=30)
    call_args = http.post.call_args
    headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
    assert headers is not None
    # Tolerant assertion -- accepts raw token OR Bearer prefix. Investigation of
    # @linear/sdk (parseClientOptions, dist/index.cjs) confirms personal API keys
    # (LinearClient({apiKey: ...}), which is what linearis and this importer use)
    # send the raw token with no "Bearer " prefix; only OAuth accessToken gets one.
    assert "fake-token" in headers.get("Authorization", "")
