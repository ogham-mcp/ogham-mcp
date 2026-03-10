import asyncio
import json
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")
    monkeypatch.setenv("ENABLE_HTTP_HEALTH", "true")
    monkeypatch.setenv("HEALTH_PORT", "0")


def test_health_endpoint_healthy():
    """GET /health should return 200 when all checks pass"""
    from ogham.http_health import handle_health_request

    with patch("ogham.http_health.full_health_check") as mock_health:
        mock_health.return_value = {
            "database": {"status": "ok", "connected": True},
            "embedding": {"status": "ok", "provider": "ollama"},
            "config": {"status": "ok", "issues": []},
        }
        status, body = asyncio.run(handle_health_request())

    assert status == 200
    data = json.loads(body)
    assert data["database"]["status"] == "ok"


def test_health_endpoint_unhealthy():
    """GET /health should return 503 when a check fails"""
    from ogham.http_health import handle_health_request

    with patch("ogham.http_health.full_health_check") as mock_health:
        mock_health.return_value = {
            "database": {"status": "error", "connected": False, "error": "refused"},
            "embedding": {"status": "ok", "provider": "ollama"},
            "config": {"status": "ok", "issues": []},
        }
        status, body = asyncio.run(handle_health_request())

    assert status == 503


def test_health_endpoint_warning_is_ok():
    """GET /health should return 200 when config has warnings but no errors"""
    from ogham.http_health import handle_health_request

    with patch("ogham.http_health.full_health_check") as mock_health:
        mock_health.return_value = {
            "database": {"status": "ok", "connected": True},
            "embedding": {"status": "ok", "provider": "ollama"},
            "config": {"status": "warning", "issues": ["Unusual dim"]},
        }
        status, body = asyncio.run(handle_health_request())

    assert status == 200
