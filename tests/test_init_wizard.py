"""Code review follow-up (TBU-159): _run_schema must not crash on a bad
EMBEDDING_DIM. _adjust_schema_dim delegates to schema_apply.render_schema_sql,
which raises for a non-positive dim -- both call sites in _run_schema are
outside any try/except, so `ogham init --dim -5` used to crash the whole
wizard with an unhandled traceback instead of printing a friendly error and
falling back to "run manually" guidance.
"""

import pytest

from ogham.config import PROVIDER_DEFAULT_DIMS
from ogham.init_wizard import _prompt_embeddings, _run_schema, run_init


@pytest.mark.parametrize("provider", sorted(PROVIDER_DEFAULT_DIMS))
def test_prompt_embeddings_dim_matches_provider_default(monkeypatch, provider):
    """TBU-160: EMBEDDING_DIM written by the wizard must match
    PROVIDER_DEFAULT_DIMS (config.py, the single source of truth) for the
    chosen provider. Previously hardcoded to "512" for every provider except
    mistral, contradicting openai/voyage's real default of 1024 and writing
    a .env that disagrees with the schema the TBU-159 boot-time
    schema-fingerprint guard expects.
    """
    calls = {"n": 0}

    def fake_ask(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return provider
        return kwargs.get("default")

    monkeypatch.setattr("ogham.init_wizard.Prompt.ask", fake_ask)

    env_vars = _prompt_embeddings()

    assert env_vars["EMBEDDING_DIM"] == str(PROVIDER_DEFAULT_DIMS[provider])


@pytest.mark.parametrize("provider", sorted(PROVIDER_DEFAULT_DIMS))
def test_run_init_noninteractive_dim_matches_provider_default(monkeypatch, provider):
    """TBU-160 (non-interactive path): `env_vars["EMBEDDING_DIM"] = str(dim
    or 512)` in run_init() ignored PROVIDER_DEFAULT_DIMS whenever the caller
    didn't pass --dim, so e.g. `ogham init --provider openai` (no --dim)
    wrote EMBEDDING_DIM=512 -- same user-facing bug as the interactive
    wizard, just via the CLI-args entry point. When dim is unset (None) it
    must fall back to the provider's default, not a bare 512.
    """
    captured: dict = {}
    monkeypatch.setattr(
        "ogham.init_wizard._write_env_file", lambda env_vars: captured.update(env_vars)
    )

    run_init(
        supabase_url="https://example.supabase.co",
        supabase_key="secret",
        provider=provider,
        dim=None,
        skip_schema=True,
        skip_clients=True,
        skip_test=True,
    )

    assert captured["EMBEDDING_DIM"] == str(PROVIDER_DEFAULT_DIMS[provider])


def test_run_init_noninteractive_explicit_dim_zero_falls_back_to_512(monkeypatch):
    """An explicit `--dim 0` keeps its pre-existing (if odd) behavior of
    falling back to 512 -- only the *unset* (None) case should switch to the
    provider-aware default. Don't silently change this documented edge case
    while fixing the None case."""
    captured: dict = {}
    monkeypatch.setattr(
        "ogham.init_wizard._write_env_file", lambda env_vars: captured.update(env_vars)
    )

    run_init(
        supabase_url="https://example.supabase.co",
        supabase_key="secret",
        provider="openai",
        dim=0,
        skip_schema=True,
        skip_clients=True,
        skip_test=True,
    )

    assert captured["EMBEDDING_DIM"] == "512"


def test_run_schema_negative_dim_does_not_crash_supabase_path(monkeypatch):
    """Supabase manual-paste path: a bad dim must print a friendly error and
    fall through to the normal confirm/guidance flow, not raise."""
    monkeypatch.setattr("ogham.init_wizard.Confirm.ask", lambda *a, **k: False)

    result = _run_schema({"DATABASE_BACKEND": "supabase", "EMBEDDING_DIM": "-5"})

    assert result is False


def test_run_schema_negative_dim_does_not_crash_postgres_path(monkeypatch):
    """Postgres-direct path: a bad dim must print a friendly error and return
    False before ever attempting a real DB connection, not raise."""
    monkeypatch.setattr("ogham.init_wizard.Confirm.ask", lambda *a, **k: True)

    result = _run_schema(
        {
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql://fake:fake@localhost/fake",
            "EMBEDDING_DIM": "-5",
        }
    )

    assert result is False
