"""Unit tests for the Obsidian exporter."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from ogham.exporters.obsidian import (
    ExportResult,
    _format_index,
    _format_topic_file,
    _rewrite_wikilinks,
    export_to_vault,
    slugify,
)


def test_slugify_preserves_simple_keys():
    assert slugify("wiki-tier1") == "wiki-tier1"
    assert slugify("user_profile") == "user_profile"


def test_slugify_strips_path_separators():
    # Slashes would create unintended subfolders; backslashes break Windows.
    assert slugify("Auth/Redesign") == "Auth-Redesign"
    assert slugify("a\\b\\c") == "a-b-c"


def test_slugify_collapses_runs_and_strips_edges():
    assert slugify("---weird-key---") == "weird-key"
    assert slugify("a:::b") == "a-b"


def test_slugify_falls_back_to_untitled_for_empty():
    assert slugify("") == "untitled"
    assert slugify("///") == "untitled"


def test_slugify_keeps_unicode():
    # Obsidian handles unicode filenames; we shouldn't ASCII-fold.
    assert slugify("你好") == "你好"
    assert slugify("café") == "café"


def test_rewrite_wikilinks_wraps_known_topics():
    body = "See wiki-tier1 for details and also auth-redesign."
    out = _rewrite_wikilinks(body, {"wiki-tier1", "auth-redesign", "self"}, "self")
    assert "[[wiki-tier1]]" in out
    assert "[[auth-redesign]]" in out


def test_rewrite_wikilinks_does_not_link_self():
    body = "This page is about wiki-tier1."
    out = _rewrite_wikilinks(body, {"wiki-tier1"}, "wiki-tier1")
    assert "[[wiki-tier1]]" not in out
    assert "wiki-tier1" in out


def test_rewrite_wikilinks_skips_inside_code_fence():
    body = textwrap.dedent("""
        Outside: wiki-tier1 here.
        ```
        Inside: wiki-tier1 should not be wrapped.
        ```
        Back outside: wiki-tier1 again.
    """).strip()
    out = _rewrite_wikilinks(body, {"wiki-tier1"}, "self")
    # Outside the fence: wrapped twice.
    assert out.count("[[wiki-tier1]]") == 2
    # Inside the fence: still raw.
    assert "Inside: wiki-tier1 should not be wrapped." in out


def test_rewrite_wikilinks_respects_word_boundaries():
    body = "wiki-tier12345 should not match wiki-tier1."
    out = _rewrite_wikilinks(body, {"wiki-tier1"}, "self")
    # Boundary check: longer token stays untouched.
    assert "wiki-tier12345" in out
    assert "[[wiki-tier12345]]" not in out


def test_rewrite_wikilinks_skips_keys_with_unsafe_chars():
    # Topic keys with slashes can't be matched by a simple word-boundary
    # rule -- the rewriter should leave them alone rather than guessing.
    body = "See foo/bar in the doc."
    out = _rewrite_wikilinks(body, {"foo/bar"}, "self")
    assert "[[foo/bar]]" not in out


def _fake_summary(**overrides) -> dict:
    base = {
        "id": "52fadcb0-3c4c-4163-b98d-38d95da28e33",
        "topic_key": "wiki-tier1",
        "profile_id": "work",
        "version": 2,
        "status": "fresh",
        "source_count": 3,
        "model_used": "gemini/gemini-2.5-flash",
        "updated_at": "2026-04-27T11:17:55.88876+00:00",
        "source_hash": b"\xf3\x26\x40\xd0",
        "content": "Body about wiki-tier1.",
    }
    base.update(overrides)
    return base


def test_format_topic_file_emits_frontmatter_and_body():
    out = _format_topic_file(_fake_summary(), {"wiki-tier1"})
    assert out.startswith("---\n")
    assert "ogham_id: 52fadcb0-3c4c-4163-b98d-38d95da28e33" in out
    assert "topic_key: wiki-tier1" in out
    assert "tags:" in out
    assert "  - ogham/wiki" in out
    assert "source_hash: f32640d0" in out
    assert "Body about wiki-tier1." in out


def test_format_topic_file_handles_postgrest_hex_string_hash():
    s = _fake_summary(source_hash="\\xdeadbeef")
    out = _format_topic_file(s, {"wiki-tier1"})
    assert "source_hash: deadbeef" in out


def test_format_topic_file_omits_source_hash_when_missing():
    s = _fake_summary(source_hash=None)
    out = _format_topic_file(s, {"wiki-tier1"})
    assert "source_hash:" not in out


def test_format_index_lists_topics():
    summaries = [
        _fake_summary(topic_key="alpha"),
        _fake_summary(topic_key="bravo", status="stale", source_count=7),
    ]
    out = _format_index("work", summaries)
    assert "profile `work`" in out
    assert "[[alpha|alpha]] -- fresh, 3 source(s)" in out
    assert "[[bravo|bravo]] -- stale, 7 source(s)" in out


def test_export_to_vault_writes_files_and_index(tmp_path: Path):
    summaries = [
        _fake_summary(topic_key="alpha", content="alpha body"),
        _fake_summary(topic_key="beta", content="beta body referencing alpha"),
    ]
    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = summaries
        result = export_to_vault(tmp_path / "vault", profile="work")

    assert isinstance(result, ExportResult)
    assert result.topics_written == 2
    assert result.errors == []

    vault = tmp_path / "vault"
    assert (vault / "alpha.md").is_file()
    assert (vault / "beta.md").is_file()
    assert (vault / "README.md").is_file()

    beta = (vault / "beta.md").read_text(encoding="utf-8")
    assert "[[alpha]]" in beta  # rewriter wired up correctly


def test_export_to_vault_errors_when_no_summaries(tmp_path: Path):
    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = []
        result = export_to_vault(tmp_path / "vault", profile="work")

    assert result.topics_written == 0
    assert any("no topic_summaries" in e for e in result.errors)


def test_export_to_vault_refuses_to_overwrite_unfamiliar_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # User dropped a non-export file in the vault.
    (vault / "personal-notes.txt").write_text("don't clobber me", encoding="utf-8")

    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = [
            _fake_summary(topic_key="alpha"),
        ]
        result = export_to_vault(vault, profile="work")

    assert result.topics_written == 0
    assert any("already contains files" in e for e in result.errors)
    # File should still be intact.
    assert (vault / "personal-notes.txt").read_text() == "don't clobber me"


def test_export_to_vault_force_overwrites(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "personal-notes.txt").write_text("clobber me", encoding="utf-8")

    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = [
            _fake_summary(topic_key="alpha"),
        ]
        result = export_to_vault(vault, profile="work", force=True)

    assert result.topics_written == 1
    # Non-export file is left alone (we never delete) -- force means
    # "go ahead and write alongside existing files".
    assert (vault / "alpha.md").is_file()


def test_export_to_vault_ignores_dotted_obsidian_dir(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # Obsidian's own metadata dir -- exporter should treat it as familiar.
    (vault / ".obsidian").mkdir()

    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = [
            _fake_summary(topic_key="alpha"),
        ]
        result = export_to_vault(vault, profile="work")

    assert result.topics_written == 1
    assert result.errors == []


def test_export_to_vault_slugifies_unsafe_topic_keys(tmp_path: Path):
    summary = _fake_summary(topic_key="Auth/Redesign", content="x")
    with patch("ogham.exporters.obsidian.get_backend") as get_backend:
        get_backend.return_value.wiki_topic_list_all.return_value = [summary]
        result = export_to_vault(tmp_path / "vault", profile="work")

    assert result.topics_written == 1
    # Slash in topic_key must not become a real subdirectory.
    assert (tmp_path / "vault" / "Auth-Redesign.md").is_file()
    assert not (tmp_path / "vault" / "Auth").exists()
