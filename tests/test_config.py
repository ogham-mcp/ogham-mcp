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


# --- env-file -> os.environ export (TBU: config.env invisible to adapters) ---
#
# pydantic-settings parses env files into the Settings object only; it never
# populates os.environ. The ingestion adapters (Telegram, Slack, GitHub, Beads)
# read their credentials straight from os.environ, so a token placed in
# ~/.ogham/config.env -- exactly where the v0.17 docs say to put it -- was
# invisible to them.


def test_env_file_values_exported_to_environ(tmp_path, monkeypatch):
    """A key present only in an env file becomes visible via os.environ."""
    import os

    from ogham.config import _export_env_files

    env_file = tmp_path / "config.env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=from-env-file\n")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    _export_env_files((str(env_file),))

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "from-env-file"


def test_real_env_var_wins_over_env_file(tmp_path, monkeypatch):
    """An explicitly set environment variable is never overwritten."""
    import os

    from ogham.config import _export_env_files

    env_file = tmp_path / "config.env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=from-env-file\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-real-env")

    _export_env_files((str(env_file),))

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "from-real-env"


def test_earlier_env_file_wins(tmp_path, monkeypatch):
    """Project .env takes precedence over the global config.env fallback."""
    import os

    from ogham.config import _export_env_files

    project = tmp_path / ".env"
    project.write_text("SLACK_BOT_TOKEN=from-project\n")
    global_env = tmp_path / "config.env"
    global_env.write_text("SLACK_BOT_TOKEN=from-global\n")
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    _export_env_files((str(project), str(global_env)))

    assert os.environ["SLACK_BOT_TOKEN"] == "from-project"


def test_missing_env_file_is_skipped(tmp_path, monkeypatch):
    """A non-existent env file path does not raise."""
    from ogham.config import _export_env_files

    _export_env_files((str(tmp_path / "nope.env"),))


def test_adapter_sees_config_env_token(tmp_path, monkeypatch):
    """Regression: import_telegram reads a token set only in an env file."""
    import os

    from ogham.config import _export_env_files

    env_file = tmp_path / "config.env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=adapter-visible\n")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    _export_env_files((str(env_file),))

    # The adapter resolves its credential from os.environ directly.
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "adapter-visible"


@pytest.mark.parametrize("value", ["stdio", "sse", "http", "streamable-http"])
def test_transport_accepts_supported_values(monkeypatch, value):
    """Every transport FastMCP supports must pass config validation."""
    monkeypatch.setenv("OGHAM_TRANSPORT", value)
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake")

    from ogham.config import Settings

    assert Settings().server_transport == value
