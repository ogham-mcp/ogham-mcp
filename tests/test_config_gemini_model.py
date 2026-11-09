"""Gemini Embedding 2 Preview was retired on 2026-11-06.

Google now redirects `gemini-embedding-2-preview` to the GA
`gemini-embedding-2`: same API schema, same embedding space, no re-index.
They also warn that the old name may stop working, so a config still naming
the preview is rewritten to the GA name at load time, with a warning, rather
than being sent to the API.
"""

import logging

import pytest

from ogham.config import Settings


@pytest.fixture(autouse=True)
def _no_env_model(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBED_MODEL", raising=False)


def test_preview_name_is_rewritten_to_ga(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
    with caplog.at_level(logging.WARNING, logger="ogham.config"):
        s = Settings()
    assert s.gemini_embed_model == "gemini-embedding-2"
    assert "gemini-embedding-2-preview" in caplog.text
    assert "gemini-embedding-2" in caplog.text


def test_ga_name_passes_through_silently(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
    with caplog.at_level(logging.WARNING, logger="ogham.config"):
        s = Settings()
    assert s.gemini_embed_model == "gemini-embedding-2"
    assert caplog.text == ""


def test_other_models_are_left_alone(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    assert Settings().gemini_embed_model == "gemini-embedding-001"


def test_default_is_ga():
    assert Settings().gemini_embed_model == "gemini-embedding-2"
