"""Tests for Ogham config transport settings."""

import pytest


@pytest.fixture(autouse=True)
def clean_ogham_env(monkeypatch):
    """Clear OGHAM_ transport env vars before each test."""
    for key in (
        "OGHAM_TRANSPORT",
        "OGHAM_HOST",
        "OGHAM_PORT",
        "OGHAM_RECALL_ENABLED",
        "OGHAM_INSCRIBE_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)


def test_transport_defaults(monkeypatch):
    """Default transport settings when no env vars set."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    s = Settings()
    assert s.server_transport == "stdio"
    assert s.server_host == "127.0.0.1"
    assert s.server_port == 8742
    assert s.recall_enabled is True
    assert s.inscribe_enabled is True


def test_transport_env_override(monkeypatch):
    """OGHAM_TRANSPORT env var overrides default."""
    monkeypatch.setenv("OGHAM_TRANSPORT", "sse")
    monkeypatch.setenv("OGHAM_HOST", "0.0.0.0")
    monkeypatch.setenv("OGHAM_PORT", "9000")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    s = Settings()
    assert s.server_transport == "sse"
    assert s.server_host == "0.0.0.0"
    assert s.server_port == 9000


def test_transport_invalid_rejected(monkeypatch):
    """Invalid transport value raises ValueError."""
    monkeypatch.setenv("OGHAM_TRANSPORT", "websocket")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    with pytest.raises(Exception):
        Settings()


def test_flow_control_env_defaults(monkeypatch):
    """Recall/inscribe controls default to enabled."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    s = Settings()
    assert s.recall_enabled is True
    assert s.inscribe_enabled is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_flow_control_env_false_values(monkeypatch, value):
    """Common false spellings disable recall and inscribe."""
    monkeypatch.setenv("OGHAM_RECALL_ENABLED", value)
    monkeypatch.setenv("OGHAM_INSCRIBE_ENABLED", value)
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    s = Settings()
    assert s.recall_enabled is False
    assert s.inscribe_enabled is False


def test_flow_control_invalid_bool_rejected(monkeypatch):
    """Invalid bool config is rejected by Pydantic."""
    monkeypatch.setenv("OGHAM_RECALL_ENABLED", "sometimes")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    with pytest.raises(Exception):
        Settings()
