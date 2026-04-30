"""Tests for the Claude Code local-memory importer.

The parser tier (parse_memory_file, find_memory_files,
parse_claude_code_memories, _infer_project_tag) is pure-Python and tested
without a database. The top-level import_claude_code_memories function
is tested by mocking ogham.export_import.import_memories so we never
embed or hit a backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _sample_memory_file(name: str = "user_role.md", *, type_: str = "user") -> str:
    return (
        "---\n"
        "name: User role\n"
        "description: User is a senior backend dev focused on Python\n"
        f"type: {type_}\n"
        "---\n"
        "Senior backend engineer at a Berlin healthcare startup.\n"
        "Deep Python expertise, ~10 years.\n"
    )


# ---------- parse_memory_file ----------


def test_parse_memory_file_happy_path(tmp_path):
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "user_role.md"
    _write(f, _sample_memory_file())

    result = parse_memory_file(f)

    assert result is not None
    assert "Senior backend engineer" in result["content"]
    assert result["source"] == "claude-code-memory"
    assert "source:claude-code-memory" in result["tags"]
    assert "type:user" in result["tags"]
    assert result["metadata"]["name"] == "User role"
    assert result["metadata"]["claude_code_type"] == "user"
    assert result["metadata"]["source_file"] == str(f)


def test_parse_memory_file_no_frontmatter_returns_none(tmp_path, caplog):
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "no_fm.md"
    _write(f, "Just a body, no frontmatter at all.")

    with caplog.at_level("WARNING"):
        assert parse_memory_file(f) is None
    assert "no frontmatter" in caplog.text


def test_parse_memory_file_invalid_yaml_returns_none(tmp_path, caplog):
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "bad_yaml.md"
    _write(f, "---\nname: [unclosed\ntype: feedback\n---\nbody\n")

    with caplog.at_level("WARNING"):
        assert parse_memory_file(f) is None
    assert "invalid YAML" in caplog.text


def test_parse_memory_file_empty_body_returns_none(tmp_path, caplog):
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "empty_body.md"
    _write(f, "---\nname: x\ntype: project\n---\n\n   \n")

    with caplog.at_level("WARNING"):
        assert parse_memory_file(f) is None
    assert "empty body" in caplog.text


def test_parse_memory_file_carries_origin_session_id(tmp_path):
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "with_origin.md"
    _write(
        f,
        "---\n"
        "name: Decision\n"
        "description: Picked Postgres over Mongo\n"
        "type: project\n"
        "originSessionId: abc-123-uuid\n"
        "---\n"
        "Picked Postgres for ACID + pgvector.\n",
    )

    result = parse_memory_file(f)
    assert result is not None
    assert result["metadata"]["origin_session_id"] == "abc-123-uuid"


def test_parse_memory_file_falls_back_to_filename_for_name(tmp_path):
    """Frontmatter without a ``name`` field uses the filename stem."""
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "fallback_name.md"
    _write(f, "---\ntype: reference\n---\nA pointer to the docs.\n")

    result = parse_memory_file(f)
    assert result is not None
    assert result["metadata"]["name"] == "fallback_name"


def test_parse_memory_file_unknown_type_still_imports(tmp_path):
    """Frontmatter without ``type`` gets ``type:unknown`` and is still imported."""
    from ogham.claude_code_import import parse_memory_file

    f = tmp_path / "no_type.md"
    _write(f, "---\nname: Misc\n---\nA loose note.\n")

    result = parse_memory_file(f)
    assert result is not None
    assert "type:unknown" in result["tags"]


# ---------- find_memory_files ----------


def test_find_memory_files_excludes_index_and_dotfiles(tmp_path):
    from ogham.claude_code_import import find_memory_files

    _write(tmp_path / "MEMORY.md", "index, skip me")
    _write(tmp_path / "a.md", "---\ntype: user\n---\nbody\n")
    _write(tmp_path / "b.md", "---\ntype: feedback\n---\nbody\n")
    _write(tmp_path / ".DS_Store.md", "junk")

    found = find_memory_files(tmp_path)
    names = [p.name for p in found]
    assert names == ["a.md", "b.md"]


def test_find_memory_files_raises_for_non_directory(tmp_path):
    from ogham.claude_code_import import find_memory_files

    f = tmp_path / "single.md"
    _write(f, "")

    with pytest.raises(FileNotFoundError):
        find_memory_files(f)


# ---------- parse_claude_code_memories ----------


def test_parse_claude_code_memories_skips_invalid_files(tmp_path):
    from ogham.claude_code_import import parse_claude_code_memories

    _write(tmp_path / "good.md", _sample_memory_file())
    _write(tmp_path / "bad.md", "no frontmatter here")
    _write(tmp_path / "MEMORY.md", "index")

    out = parse_claude_code_memories(tmp_path)

    assert len(out) == 1
    assert out[0]["metadata"]["source_file"].endswith("good.md")


def test_parse_claude_code_memories_returns_sorted_order(tmp_path):
    from ogham.claude_code_import import parse_claude_code_memories

    _write(tmp_path / "z.md", _sample_memory_file())
    _write(tmp_path / "a.md", _sample_memory_file())
    _write(tmp_path / "m.md", _sample_memory_file())

    out = parse_claude_code_memories(tmp_path)
    files = [Path(m["metadata"]["source_file"]).name for m in out]
    assert files == ["a.md", "m.md", "z.md"]


# ---------- _infer_project_tag ----------


def test_infer_project_tag_from_claude_encoded_path(tmp_path):
    """A directory under ``-Users-<...>-foo/memory/`` yields ``project:foo``."""
    from ogham.claude_code_import import _infer_project_tag

    fake = tmp_path / "-Users-someone-Developer-web-projects-foo" / "memory"
    fake.mkdir(parents=True)

    assert _infer_project_tag(fake) == "foo"


def test_infer_project_tag_returns_none_for_non_claude_path(tmp_path):
    from ogham.claude_code_import import _infer_project_tag

    assert _infer_project_tag(tmp_path) is None


def test_parse_memory_file_includes_project_tag_when_under_encoded_path(tmp_path):
    from ogham.claude_code_import import parse_memory_file

    encoded = tmp_path / "-Users-someone-Developer-web-projects-bar" / "memory"
    encoded.mkdir(parents=True)
    f = encoded / "x.md"
    _write(f, _sample_memory_file())

    result = parse_memory_file(f)
    assert result is not None
    assert "project:bar" in result["tags"]


def test_parse_memory_file_explicit_project_tag_overrides_inferred(tmp_path):
    """The encoded-cwd heuristic loses hyphens; an explicit override wins."""
    from ogham.claude_code_import import parse_memory_file

    # Encoded path "openbrain-sharedmemory" naively decodes to "sharedmemory",
    # but the user wants the canonical "ogham" tag instead.
    encoded = tmp_path / "-Users-kev-Developer-web-projects-openbrain-sharedmemory" / "memory"
    encoded.mkdir(parents=True)
    f = encoded / "x.md"
    _write(f, _sample_memory_file())

    result = parse_memory_file(f, project_tag="ogham")
    assert result is not None
    assert "project:ogham" in result["tags"]
    assert "project:sharedmemory" not in result["tags"]


# ---------- import_claude_code_memories (top-level) ----------


def test_import_claude_code_memories_passes_envelope_to_import_memories(tmp_path):
    """Top-level wrapper builds a JSON envelope and forwards to export_import."""
    from ogham.claude_code_import import import_claude_code_memories

    _write(tmp_path / "a.md", _sample_memory_file())
    _write(tmp_path / "b.md", _sample_memory_file(name="b.md", type_="feedback"))

    fake_result = {
        "status": "complete",
        "profile": "test",
        "imported": 2,
        "skipped": 0,
        "total": 2,
    }

    with patch("ogham.claude_code_import._import_memories", return_value=fake_result) as mocked:
        out = import_claude_code_memories(tmp_path, profile="test", dedup_threshold=0.8)

    assert mocked.call_count == 1
    call = mocked.call_args
    envelope = call.args[0]
    import json

    parsed = json.loads(envelope)
    assert len(parsed["memories"]) == 2
    assert call.kwargs["profile"] == "test"
    assert call.kwargs["dedup_threshold"] == 0.8

    assert out["imported"] == 2
    assert out["directory"] == str(tmp_path)


def test_import_claude_code_memories_empty_directory_returns_warning(tmp_path):
    from ogham.claude_code_import import import_claude_code_memories

    out = import_claude_code_memories(tmp_path, profile="test")

    assert out["status"] == "complete"
    assert out["imported"] == 0
    assert out["total"] == 0
    assert out.get("warning") == "no memory files found"


def test_import_claude_code_memories_directory_with_only_invalid_files_returns_warning(
    tmp_path, caplog
):
    """All files unparseable -> empty parse list -> early-return with warning."""
    from ogham.claude_code_import import import_claude_code_memories

    _write(tmp_path / "broken.md", "no frontmatter")
    _write(tmp_path / "MEMORY.md", "index, skipped")

    with caplog.at_level("WARNING"):
        out = import_claude_code_memories(tmp_path, profile="test")

    assert out["imported"] == 0
    assert out.get("warning") == "no memory files found"
