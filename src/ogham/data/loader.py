"""Language data loader for Ogham MCP word lists.

Loads YAML language files with LRU caching. Falls back to English
when a requested language file is not found.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_LANGUAGES_DIR = Path(__file__).parent / "languages"


@lru_cache(maxsize=32)
def _load_language_file(lang: str) -> dict[str, Any]:
    """Load and cache a single language YAML file.

    Falls back to English if the requested language is unavailable.
    Raises FileNotFoundError if English is also missing.
    """
    path = _LANGUAGES_DIR / f"{lang}.yaml"
    if not path.exists():
        if lang != "en":
            logger.debug("Language file %s not found, falling back to English", lang)
            return _load_language_file("en")
        raise FileNotFoundError(f"English language file not found at {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Language file {path} did not produce a dict")

    return data


def _available_languages() -> list[str]:
    """Return list of available language codes (stems of .yaml files)."""
    return [p.stem for p in sorted(_LANGUAGES_DIR.glob("*.yaml"))]


# --- Public accessors ---


def get_day_names(lang: str = "en") -> dict[str, int]:
    """Return day_name -> day_index mapping for a language."""
    return _load_language_file(lang).get("day_names", {})


def get_all_day_names() -> dict[str, int]:
    """Return merged day_name -> day_index mapping from all languages."""
    merged: dict[str, int] = {}
    for lang in _available_languages():
        merged.update(get_day_names(lang))
    return merged


def get_every_words(lang: str = "en") -> list[str]:
    """Return 'every/each/weekly' keywords for a language."""
    return _load_language_file(lang).get("every_words", [])


def get_all_every_words() -> set[str]:
    """Return merged every-words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_every_words(lang))
    return merged


def get_temporal_keywords(lang: str = "en") -> list[str]:
    """Return temporal keyword list for a language."""
    return _load_language_file(lang).get("temporal_keywords", [])


def get_decision_words(lang: str = "en") -> list[str]:
    """Return decision word list for a language."""
    return _load_language_file(lang).get("decision_words", [])


def get_all_decision_words() -> set[str]:
    """Return merged decision words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_decision_words(lang))
    return merged


def get_error_words(lang: str = "en") -> list[str]:
    """Return error word list for a language."""
    return _load_language_file(lang).get("error_words", [])


def get_all_error_words() -> set[str]:
    """Return merged error words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_error_words(lang))
    return merged


def get_architecture_words(lang: str = "en") -> list[str]:
    """Return architecture word list for a language."""
    return _load_language_file(lang).get("architecture_words", [])


def get_all_architecture_words() -> set[str]:
    """Return merged architecture words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_architecture_words(lang))
    return merged


def get_direction_words(lang: str = "en") -> dict[str, list[str]]:
    """Return direction_words mapping (after/before -> word lists)."""
    return _load_language_file(lang).get("direction_words", {})


def get_month_names(lang: str = "en") -> dict[str, int]:
    """Return month_name -> month_number mapping."""
    return _load_language_file(lang).get("month_names", {})


def get_word_numbers(lang: str = "en") -> dict[str, int]:
    """Return word -> number mapping (one->1, two->2, etc.)."""
    return _load_language_file(lang).get("word_numbers", {})


def get_query_hints(lang: str = "en", hint_type: str = "") -> list[str]:
    """Return query hint phrases for a given hint type.

    hint_type is one of: multi_hop, ordering, summary.
    If empty, returns all hints as a flat list.
    """
    hints = _load_language_file(lang).get("query_hints", {})
    if hint_type:
        return hints.get(hint_type, [])
    # Flatten all hint types into a single list
    flat: list[str] = []
    for phrases in hints.values():
        flat.extend(phrases)
    return flat


def get_compression_decision_words(lang: str = "en") -> list[str]:
    """Return compression decision words for a language."""
    return _load_language_file(lang).get("compression_decision_words", [])


def get_event_words(lang: str = "en") -> list[str]:
    """Return event word list for a language."""
    return _load_language_file(lang).get("event_words", [])


def get_all_event_words() -> set[str]:
    """Return merged event words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_event_words(lang))
    return merged


def get_activity_words(lang: str = "en") -> list[str]:
    """Return activity word list for a language."""
    return _load_language_file(lang).get("activity_words", [])


def get_all_activity_words() -> set[str]:
    """Return merged activity words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_activity_words(lang))
    return merged


def get_emotion_words(lang: str = "en") -> list[str]:
    """Return emotion word list for a language."""
    return _load_language_file(lang).get("emotion_words", [])


def get_all_emotion_words() -> set[str]:
    """Return merged emotion words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_emotion_words(lang))
    return merged


def get_relationship_words(lang: str = "en") -> list[str]:
    """Return relationship word list for a language."""
    return _load_language_file(lang).get("relationship_words", [])


def get_all_relationship_words() -> set[str]:
    """Return merged relationship words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_relationship_words(lang))
    return merged


def get_possessive_triggers(lang: str = "en") -> list[str]:
    """Return possessive trigger words for a language."""
    return _load_language_file(lang).get("possessive_triggers", [])


def get_all_possessive_triggers() -> set[str]:
    """Return merged possessive triggers from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_possessive_triggers(lang))
    return merged


def get_quantity_units(lang: str = "en") -> list[str]:
    """Return quantity unit words for a language."""
    return _load_language_file(lang).get("quantity_units", [])


def get_all_quantity_units() -> set[str]:
    """Return merged quantity units from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_quantity_units(lang))
    return merged


def get_preference_words(lang: str = "en") -> list[str]:
    """Return preference trigger words for a language."""
    return _load_language_file(lang).get("preference_words", [])


def get_all_preference_words() -> set[str]:
    """Return merged preference words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_preference_words(lang))
    return merged


def get_negation_markers(lang: str = "en") -> list[str]:
    """Return negation / supersession markers for a language.

    Used by the contradiction producer: a stored memory containing any of
    these markers that also has high similarity to an existing memory with
    no such markers is treated as contradicting the earlier statement.
    """
    return _load_language_file(lang).get("negation_markers", [])


def get_all_negation_markers() -> set[str]:
    """Return merged negation markers from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_negation_markers(lang))
    return merged


def get_query_filler(lang: str = "en") -> list[str]:
    """Return query-specific filler words for a language."""
    return _load_language_file(lang).get("query_filler", [])


def get_all_query_filler() -> set[str]:
    """Return merged query filler words from all languages."""
    merged: set[str] = set()
    for lang in _available_languages():
        merged.update(get_query_filler(lang))
    return merged


def get_wiki_compile(key: str, lang: str = "en") -> str:
    """Return a wiki-compile prompt fragment for a language.

    `key` is one of the keys under the `wiki_compile` section in the
    language YAML (e.g. 'system_prompt', 'prompt_template'). Falls back
    to the English value if the requested language doesn't override it,
    so partial localisations don't break the compile pipeline.
    """
    data = _load_language_file(lang).get("wiki_compile", {})
    value = data.get(key)
    if value is None and lang != "en":
        value = _load_language_file("en").get("wiki_compile", {}).get(key)
    return value or ""


def get_wiki_message(key: str, lang: str = "en") -> str:
    """Return a wiki-tool user-facing message template for a language.

    `key` is one of the keys under `wiki_messages` (e.g. 'no_sources',
    'not_cached'). Falls back to English on miss. Returned strings are
    Python format-strings; callers fill placeholders via .format(...).
    """
    data = _load_language_file(lang).get("wiki_messages", {})
    value = data.get(key)
    if value is None and lang != "en":
        value = _load_language_file("en").get("wiki_messages", {}).get(key)
    return value or ""


def invalidate_cache() -> None:
    """Clear the language file LRU cache."""
    _load_language_file.cache_clear()
