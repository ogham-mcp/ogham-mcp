"""Tests for wiki-layer YAML localisation (T1.7).

Validates that:
  * en.yaml carries the wiki_compile + wiki_messages sections
  * loader accessors return expected English values
  * unknown locales fall back to English (not empty string)
  * recompute prompt builder loads from YAML (not hardcoded constants)
  * tools.wiki user-facing messages format with placeholders correctly

The anti-injection guard test for the system prompt is preserved here
(was previously in test_recompute.py via _COMPILE_SYSTEM_PROMPT) so a
future YAML edit can't silently drop the prompt-override warning.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# --------------------------------------------------------------------- #
# loader accessors
# --------------------------------------------------------------------- #


def test_get_wiki_compile_returns_system_prompt():
    from ogham.data.loader import get_wiki_compile

    out = get_wiki_compile("system_prompt")
    assert out  # non-empty
    assert "<source" in out  # the anti-injection guard references the wrapper
    assert "wiki" in out.lower()


def test_get_wiki_compile_returns_prompt_template():
    from ogham.data.loader import get_wiki_compile

    out = get_wiki_compile("prompt_template")
    assert "{topic_key}" in out
    assert "{sources}" in out


def test_get_wiki_compile_unknown_lang_falls_back_to_english():
    """Unknown locale must not break the compile path."""
    from ogham.data.loader import get_wiki_compile

    out = get_wiki_compile("system_prompt", lang="xx")  # xx never exists
    assert out  # English fallback returns non-empty


def test_get_wiki_compile_unknown_key_returns_empty_string():
    from ogham.data.loader import get_wiki_compile

    assert get_wiki_compile("nonexistent_key") == ""


def test_get_wiki_message_returns_known_keys():
    from ogham.data.loader import get_wiki_message

    for key in ("no_sources", "not_cached", "row_missing", "invalid_direction"):
        out = get_wiki_message(key)
        assert out, f"wiki_messages[{key!r}] missing in en.yaml"


def test_get_wiki_message_no_sources_template_has_placeholders():
    """Format-string sanity: must accept topic + profile."""
    from ogham.data.loader import get_wiki_message

    template = get_wiki_message("no_sources")
    formatted = template.format(topic="quantum", profile="work")
    assert "quantum" in formatted
    assert "work" in formatted


def test_get_wiki_message_unknown_lang_falls_back_to_english():
    from ogham.data.loader import get_wiki_message

    out = get_wiki_message("no_sources", lang="xx")
    assert out  # falls back


# --------------------------------------------------------------------- #
# system prompt anti-injection invariant -- preserved across the refactor
# --------------------------------------------------------------------- #


def test_compile_system_prompt_carries_anti_injection_guard():
    """Critical: the YAML refactor must not drop the prompt-override warning.

    A future contributor editing en.yaml could accidentally remove the
    "ignore previous instructions" / "you are now..." guard text. This
    test asserts the guard remains present so prompt-tuning changes
    can't strip it silently.
    """
    from ogham.recompute import _compile_system_prompt

    sp = _compile_system_prompt()
    assert "ignore previous instructions" in sp.lower()
    assert "you are now" in sp.lower() or "disregard the system prompt" in sp.lower()
    assert "Flagged content" in sp


def test_compile_prompt_template_renders_with_sources():
    from ogham.recompute import _render_compile_prompt

    rendered = _render_compile_prompt(
        "quantum",
        [{"id": "abc", "content": "claim 1"}, {"id": "def", "content": "claim 2"}],
    )
    assert "Topic: quantum" in rendered
    assert '<source id="abc">' in rendered
    assert '<source id="def">' in rendered
    assert "claim 1" in rendered
    assert "claim 2" in rendered


# --------------------------------------------------------------------- #
# tools.wiki user-facing messages
# --------------------------------------------------------------------- #


def test_compile_wiki_no_sources_message_localises():
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(
            wiki,
            "recompute_topic_summary",
            return_value={"action": "no_sources"},
        ),
    ):
        out = wiki.compile_wiki(topic="ghost")

    # Message comes from YAML, not from a hardcoded f-string.
    assert "ghost" in out["message"]
    assert "work" in out["message"]


def test_walk_knowledge_invalid_direction_message_localises():
    from ogham.tools import wiki

    out = wiki.walk_knowledge(start_id="abc", direction="sideways")
    assert "sideways" in out["message"]
    # Still mentions valid options.
    assert "outgoing" in out["message"]


def test_query_topic_summary_not_cached_message_localises():
    from ogham.tools import wiki

    with (
        patch.object(wiki, "get_active_profile", return_value="work"),
        patch.object(wiki, "get_summary_by_topic", return_value=None),
    ):
        out = wiki.query_topic_summary(topic="ghost")

    assert "ghost" in out["message"]
    assert "compile_wiki" in out["message"]


# --------------------------------------------------------------------- #
# locale knob
# --------------------------------------------------------------------- #


def test_locale_setting_defaults_to_english():
    """Default locale must remain 'en' so existing deployments don't shift."""
    from ogham.config import settings

    assert settings.locale == "en"


def test_locale_setting_drives_loader_lookup():
    """When settings.locale changes, the recompute path queries that locale."""
    from ogham import recompute

    captured = {}

    def fake_get(key, lang="en"):
        captured.setdefault(key, []).append(lang)
        return "stub"

    with patch.object(recompute, "get_wiki_compile", side_effect=fake_get):
        with patch("ogham.config.settings.locale", "de"):
            recompute._compile_system_prompt()
            recompute._render_compile_prompt("topic", [])

    assert captured.get("system_prompt") == ["de"]
    assert captured.get("prompt_template") == ["de"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
