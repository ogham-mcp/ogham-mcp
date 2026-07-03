"""Linear importer -- fetch issues via GraphQL and map to Ogham memory shape."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

LINEAR_ENDPOINT = "https://api.linear.app/graphql"


class LinearClient:
    """Thin GraphQL client scoped to what the importer needs."""

    def __init__(self, token: str, http_client: httpx.Client | None = None):
        self._token = token
        self._http = http_client or httpx.Client(timeout=30.0)

    def fetch_issues(self, team_key: str, since_days: int) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        query = """
            query IssuesForTeam($teamKey: String!, $cutoff: DateTimeOrDuration) {
              issues(
                filter: { team: { key: { eq: $teamKey } }, updatedAt: { gt: $cutoff } }
                first: 250
              ) {
                nodes {
                  id
                  identifier
                  title
                  description
                  state { name }
                  priority
                  assignee { name }
                  labels { nodes { name } }
                  comments { nodes { id body user { name } } }
                }
              }
            }
        """
        response = self._http.post(
            LINEAR_ENDPOINT,
            json={"query": query, "variables": {"teamKey": team_key, "cutoff": cutoff}},
            # Personal API keys authenticate with the raw token, no "Bearer "
            # prefix -- confirmed against @linear/sdk's parseClientOptions
            # (only OAuth accessToken gets the Bearer prefix; LinearClient({apiKey})
            # sends `Authorization: <apiKey>`), and linearis (the local `linear`
            # CLI) constructs its client the same way.
            headers={"Authorization": self._token, "Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("issues", {}).get("nodes", [])


def map_issue_to_memory(issue: dict[str, Any]) -> dict[str, Any]:
    """Convert a Linear issue dict to Ogham memory shape.

    Returns a dict suitable for passing to ``store_memory``:
    ``{content, metadata, tags}``.
    """
    lines: list[str] = [f"# {issue['title']}", "", issue.get("description") or ""]

    comments = (issue.get("comments") or {}).get("nodes") or []
    if comments:
        lines.append("")
        lines.append("## Comments")
        for c in comments:
            user = ((c.get("user") or {}).get("name")) or "unknown"
            lines.append(f"\n**{user}**: {c.get('body') or ''}")

    content = "\n".join(lines).strip()

    labels = (issue.get("labels") or {}).get("nodes") or []
    tags = [f"linear:{label['name']}" for label in labels]
    tags.append("type:task")

    metadata: dict[str, Any] = {
        "tracker": "linear",
        "tracker_external_id": issue["id"],
        "identifier": issue.get("identifier"),
        "status": (issue.get("state") or {}).get("name"),
        "priority": issue.get("priority"),
    }
    assignee = issue.get("assignee")
    if assignee:
        metadata["assignee"] = assignee.get("name")

    return {"content": content, "metadata": metadata, "tags": tags}
