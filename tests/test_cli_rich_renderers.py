from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4
import sys

from typer.testing import CliRunner

from ogham import cli


runner = CliRunner()


def test_store_rich_output_handles_uuid_conflicts_and_datetime_expiry(monkeypatch) -> None:
    monkeypatch.setattr(cli, "console", cli.Console(record=True))
    monkeypatch.setitem(sys.modules, "ogham.config", SimpleNamespace(settings=SimpleNamespace(default_profile="default")))
    monkeypatch.setitem(
        sys.modules,
        "ogham.service",
        SimpleNamespace(
            store_memory_enriched=lambda **_kwargs: {
                "id": uuid4(),
                "profile": "demo",
                "expires_at": datetime.now(UTC),
                "conflict_warning": "Found 1 existing memory(s) with >75% similarity.",
                "conflicts": [
                    {
                        "id": uuid4(),
                        "similarity": 1.0,
                        "content_preview": "preview text",
                    }
                ],
            }
        ),
    )

    result = runner.invoke(cli.app, ["store", "content", "--profile", "demo"])

    assert result.exit_code == 0
    assert "Stored memory" in result.stdout
    assert "Found 1 existing memory(s)" in result.stdout
    assert "preview text" in result.stdout


def test_list_rich_output_handles_datetime_created_at(monkeypatch) -> None:
    monkeypatch.setattr(cli, "console", cli.Console(record=True))
    monkeypatch.setitem(sys.modules, "ogham.config", SimpleNamespace(settings=SimpleNamespace(default_profile="default")))
    monkeypatch.setitem(
        sys.modules,
        "ogham.database",
        SimpleNamespace(
            list_recent_memories=lambda **_kwargs: [
                {
                    "id": uuid4(),
                    "created_at": datetime.now(UTC),
                    "content": "memory content",
                    "tags": ["tag-a"],
                    "source": "unit-test",
                }
            ]
        ),
    )

    result = runner.invoke(cli.app, ["list", "--profile", "demo", "--limit", "1"])

    assert result.exit_code == 0
    assert "Recent Memories" in result.stdout
    assert "memory content" in result.stdout


def test_store_json_output_remains_structured(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ogham.config", SimpleNamespace(settings=SimpleNamespace(default_profile="default")))
    monkeypatch.setitem(
        sys.modules,
        "ogham.service",
        SimpleNamespace(
            store_memory_enriched=lambda **_kwargs: {
                "id": uuid4(),
                "profile": "demo",
                "expires_at": datetime.now(UTC),
                "conflicts": [],
            }
        ),
    )

    result = runner.invoke(cli.app, ["store", "content", "--profile", "demo", "--json"])

    assert result.exit_code == 0
    assert '"profile": "demo"' in result.stdout
