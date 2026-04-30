"""Tests for the Claude.ai data-export importer.

Pure-Python parser tier (extract_turn_pairs, turn_pair_to_memory,
parse_export, _open_export, _message_text, _is_noise) is tested without a
DB. Top-level import_claude_ai_export is tested by mocking
ogham.export_import.import_memories so we never embed or hit a backend.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


def _msg(sender: str, text: str, uuid: str, *, created_at: str = "2026-04-01T12:00:00Z") -> dict:
    return {
        "uuid": uuid,
        "sender": sender,
        "text": text,
        "content": [{"type": "text", "text": text}],
        "created_at": created_at,
        "updated_at": created_at,
    }


def _conversation(
    uuid: str,
    name: str,
    messages: list[dict],
    *,
    updated_at: str = "2026-04-01T12:00:00Z",
) -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "summary": "",
        "created_at": updated_at,
        "updated_at": updated_at,
        "account": {"uuid": "acct-1"},
        "chat_messages": messages,
    }


def _write_export(tmp_path: Path, conversations: list[dict]) -> Path:
    target = tmp_path / "conversations.json"
    target.write_text(json.dumps(conversations), encoding="utf-8")
    return tmp_path


# ---------- _message_text ----------


def test_message_text_prefers_flat_text_field():
    from ogham.claude_ai_import import _message_text

    msg = {"text": "flat-string", "content": [{"type": "text", "text": "ignored"}]}
    assert _message_text(msg) == "flat-string"


def test_message_text_falls_back_to_content_blocks():
    from ogham.claude_ai_import import _message_text

    msg = {
        "text": "",
        "content": [
            {"type": "text", "text": "block-one"},
            {"type": "text", "text": "block-two"},
        ],
    }
    assert _message_text(msg) == "block-one\n\nblock-two"


def test_message_text_skips_non_text_blocks():
    from ogham.claude_ai_import import _message_text

    msg = {
        "content": [
            {"type": "tool_use", "name": "calculator"},
            {"type": "text", "text": "the answer"},
        ]
    }
    assert _message_text(msg) == "the answer"


def test_message_text_empty_when_nothing_usable():
    from ogham.claude_ai_import import _message_text

    assert _message_text({}) == ""
    assert _message_text({"text": None, "content": None}) == ""


# ---------- _is_noise ----------


def test_is_noise_flags_short_assistant():
    from ogham.claude_ai_import import _is_noise

    # 30 chars assistant → under 50-char threshold → noise.
    assert _is_noise("Tell me about it", "Sure, here you go.") is True


def test_is_noise_flags_pleasantry_human_with_short_real_assistant():
    from ogham.claude_ai_import import _is_noise

    # "thanks" matches noise stem AND human is short.
    short_assistant = "x" * 40
    assert _is_noise("thanks", short_assistant) is True


def test_is_noise_keeps_substantive_pair():
    from ogham.claude_ai_import import _is_noise

    human = "How does the BEAM benchmark score retrieval?"
    assistant = "BEAM checks whether returned IDs overlap the gold chat IDs..." * 3
    assert _is_noise(human, assistant) is False


def test_is_noise_keeps_long_assistant_even_with_short_human():
    from ogham.claude_ai_import import _is_noise

    # Short human that's not a noise stem ("show?" doesn't match).
    long_assistant = "y" * 200
    assert _is_noise("show?", long_assistant) is False


# ---------- extract_turn_pairs ----------


def test_extract_turn_pairs_basic_pairing():
    from ogham.claude_ai_import import extract_turn_pairs

    conv = _conversation(
        "c1",
        "test",
        [
            _msg("human", "What is RAG?", "h1"),
            _msg("assistant", "Retrieval Augmented Generation is when..." * 3, "a1"),
            _msg("human", "And how does it use vectors?", "h2"),
            _msg("assistant", "Vectors capture semantic similarity by..." * 3, "a2"),
        ],
    )
    pairs = list(extract_turn_pairs(conv))
    assert len(pairs) == 2
    assert pairs[0][0]["uuid"] == "h1" and pairs[0][1]["uuid"] == "a1"
    assert pairs[1][0]["uuid"] == "h2" and pairs[1][1]["uuid"] == "a2"


def test_extract_turn_pairs_drops_two_consecutive_humans():
    from ogham.claude_ai_import import extract_turn_pairs

    conv = _conversation(
        "c1",
        "test",
        [
            _msg("human", "first prompt — discarded by retry", "h1"),
            _msg("human", "What is RAG actually?", "h2"),
            _msg("assistant", "Retrieval Augmented Generation..." * 3, "a1"),
        ],
    )
    pairs = list(extract_turn_pairs(conv))
    # Only h2 → a1 is a valid pair; h1 is orphaned by the second human.
    assert len(pairs) == 1
    assert pairs[0][0]["uuid"] == "h2"


def test_extract_turn_pairs_drops_orphan_assistant():
    from ogham.claude_ai_import import extract_turn_pairs

    conv = _conversation(
        "c1",
        "test",
        [
            _msg("assistant", "I noticed something strange — " * 5, "a1"),
            _msg("human", "What did you notice?", "h1"),
            _msg("assistant", "On reflection, " * 10, "a2"),
        ],
    )
    pairs = list(extract_turn_pairs(conv))
    assert len(pairs) == 1
    assert pairs[0][0]["uuid"] == "h1" and pairs[0][1]["uuid"] == "a2"


def test_extract_turn_pairs_smart_filter_drops_pleasantries():
    from ogham.claude_ai_import import extract_turn_pairs

    conv = _conversation(
        "c1",
        "test",
        [
            _msg("human", "thanks", "h1"),
            _msg("assistant", "ok", "a1"),
            _msg("human", "Now explain why HNSW is non-deterministic.", "h2"),
            _msg("assistant", "HNSW indexes are graph-based and..." * 3, "a2"),
        ],
    )
    filtered = list(extract_turn_pairs(conv, smart_filter=True))
    raw = list(extract_turn_pairs(conv, smart_filter=False))
    assert len(filtered) == 1 and filtered[0][0]["uuid"] == "h2"
    assert len(raw) == 2


def test_extract_turn_pairs_handles_missing_chat_messages():
    from ogham.claude_ai_import import extract_turn_pairs

    assert list(extract_turn_pairs({"chat_messages": None})) == []
    assert list(extract_turn_pairs({})) == []


# ---------- turn_pair_to_memory ----------


def test_turn_pair_to_memory_shape():
    from ogham.claude_ai_import import turn_pair_to_memory

    human = _msg("human", "What is RAG?", "h1")
    assistant = _msg(
        "assistant",
        "Retrieval Augmented Generation is when..." * 3,
        "a1",
        created_at="2026-03-01T08:00:00Z",
    )
    conv = _conversation("c1", "RAG basics", [human, assistant])
    mem = turn_pair_to_memory(human, assistant, conv)

    assert mem["source"] == "claude-ai"
    assert mem["content"].startswith("Retrieval Augmented Generation")
    assert "source:claude-ai" in mem["tags"]
    assert "claude-conversation:rag-basics" in mem["tags"]
    md = mem["metadata"]
    assert md["user_prompt"] == "What is RAG?"
    assert md["claude_message_uuid"] == "a1"
    assert md["claude_human_uuid"] == "h1"
    assert md["claude_conversation_uuid"] == "c1"
    assert md["claude_conversation_title"] == "RAG basics"
    assert md["claude_created_at"] == "2026-03-01T08:00:00Z"


def test_turn_pair_to_memory_with_project_tag_override():
    from ogham.claude_ai_import import turn_pair_to_memory

    human = _msg("human", "q", "h1")
    assistant = _msg("assistant", "long answer..." * 10, "a1")
    conv = _conversation("c1", "n", [human, assistant])
    mem = turn_pair_to_memory(human, assistant, conv, project_tag="ogham")
    assert "project:ogham" in mem["tags"]


def test_turn_pair_to_memory_empty_title_falls_back_to_untitled():
    from ogham.claude_ai_import import turn_pair_to_memory

    human = _msg("human", "q", "h1")
    assistant = _msg("assistant", "long answer..." * 10, "a1")
    conv = _conversation("c1", "", [human, assistant])
    mem = turn_pair_to_memory(human, assistant, conv)
    assert "claude-conversation:untitled" in mem["tags"]


# ---------- _open_export ----------


def test_open_export_reads_zip(tmp_path):
    from ogham.claude_ai_import import _open_export

    convs = [
        _conversation(
            "c1",
            "n",
            [_msg("human", "q", "h1"), _msg("assistant", "a" * 60, "a1")],
        )
    ]
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(convs))
    result = list(_open_export(zip_path))
    assert len(result) == 1 and result[0]["uuid"] == "c1"


def test_open_export_reads_directory(tmp_path):
    from ogham.claude_ai_import import _open_export

    convs = [_conversation("c1", "n", [])]
    _write_export(tmp_path, convs)
    result = list(_open_export(tmp_path))
    assert len(result) == 1


def test_open_export_reads_json_file(tmp_path):
    from ogham.claude_ai_import import _open_export

    convs = [_conversation("c1", "n", [])]
    target = tmp_path / "conversations.json"
    target.write_text(json.dumps(convs), encoding="utf-8")
    result = list(_open_export(target))
    assert len(result) == 1


def test_open_export_zip_without_conversations_json_errors(tmp_path):
    from ogham.claude_ai_import import _open_export

    zip_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("something_else.json", "{}")
    import pytest

    with pytest.raises(FileNotFoundError, match="conversations.json"):
        list(_open_export(zip_path))


def test_open_export_directory_without_conversations_json_errors(tmp_path):
    import pytest

    from ogham.claude_ai_import import _open_export

    with pytest.raises(FileNotFoundError, match="conversations.json"):
        list(_open_export(tmp_path))


# ---------- parse_export ----------


def test_parse_export_end_to_end_smart_filter(tmp_path):
    from ogham.claude_ai_import import parse_export

    convs = [
        _conversation(
            "c1",
            "first conversation",
            [
                _msg("human", "thanks", "h1"),
                _msg("assistant", "ok", "a1"),
                _msg("human", "Explain HNSW non-determinism.", "h2"),
                _msg("assistant", "HNSW graph build path matters because..." * 3, "a2"),
            ],
        ),
        _conversation(
            "c2",
            "second conversation",
            [
                _msg("human", "Compare RRF and weighted sum.", "h3"),
                _msg("assistant", "RRF is rank-based whereas weighted sum..." * 3, "a3"),
            ],
        ),
    ]
    _write_export(tmp_path, convs)
    parsed = parse_export(tmp_path)
    assert len(parsed) == 2
    titles = {m["metadata"]["claude_conversation_title"] for m in parsed}
    assert titles == {"first conversation", "second conversation"}
    # turn_index annotation present and 0-based per-conversation
    indices = {
        m["metadata"]["claude_conversation_uuid"]: m["metadata"]["turn_index"] for m in parsed
    }
    assert all(i == 0 for i in indices.values())


def test_parse_export_since_filter(tmp_path):
    from ogham.claude_ai_import import parse_export

    convs = [
        _conversation(
            "old",
            "old chat",
            [_msg("human", "q", "h1"), _msg("assistant", "x" * 80, "a1")],
            updated_at="2025-01-01T00:00:00Z",
        ),
        _conversation(
            "new",
            "new chat",
            [_msg("human", "q", "h2"), _msg("assistant", "y" * 80, "a2")],
            updated_at="2026-04-01T00:00:00Z",
        ),
    ]
    _write_export(tmp_path, convs)
    parsed = parse_export(tmp_path, since="2026-01-01")
    assert len(parsed) == 1
    assert parsed[0]["metadata"]["claude_conversation_uuid"] == "new"


def test_parse_export_since_accepts_datetime(tmp_path):
    from ogham.claude_ai_import import parse_export

    convs = [
        _conversation(
            "old",
            "old",
            [_msg("human", "q", "h1"), _msg("assistant", "x" * 80, "a1")],
            updated_at="2025-01-01T00:00:00Z",
        ),
    ]
    _write_export(tmp_path, convs)
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    parsed = parse_export(tmp_path, since=cutoff)
    assert parsed == []


def test_parse_export_raw_mode_disables_smart_filter(tmp_path):
    from ogham.claude_ai_import import parse_export

    convs = [
        _conversation(
            "c1",
            "c",
            [
                _msg("human", "thanks", "h1"),
                _msg("assistant", "yw" * 30, "a1"),
            ],
        ),
    ]
    _write_export(tmp_path, convs)
    # smart_filter argument is True but mode='raw' must override it.
    parsed = parse_export(tmp_path, mode="raw", smart_filter=True)
    assert len(parsed) == 1
    assert parsed[0]["metadata"]["import_mode"] == "raw"


# ---------- import_claude_ai_export ----------


def test_import_claude_ai_export_passes_envelope_to_import_memories(tmp_path):
    """Top-level wiring: parse → JSON envelope → import_memories. No DB."""
    from ogham.claude_ai_import import import_claude_ai_export

    convs = [
        _conversation(
            "c1",
            "title",
            [_msg("human", "q?", "h1"), _msg("assistant", "real answer..." * 10, "a1")],
        )
    ]
    _write_export(tmp_path, convs)

    fake_result = {"status": "complete", "imported": 1, "skipped": 0, "total": 1, "profile": "p"}
    with patch("ogham.claude_ai_import._import_memories", return_value=fake_result) as mock_import:
        result = import_claude_ai_export(tmp_path, profile="p")

    assert mock_import.call_count == 1
    envelope_str, *_ = mock_import.call_args.args
    envelope = json.loads(envelope_str)
    assert len(envelope["memories"]) == 1
    assert envelope["memories"][0]["source"] == "claude-ai"

    assert result["imported"] == 1
    assert result["mode"] == "turn-pairs"
    assert result["path"] == str(tmp_path)


def test_import_claude_ai_export_short_circuits_on_empty_export(tmp_path):
    """No turn-pairs → return early with warning, do not call import_memories."""
    from ogham.claude_ai_import import import_claude_ai_export

    _write_export(tmp_path, [_conversation("c1", "empty", [])])

    with patch("ogham.claude_ai_import._import_memories") as mock_import:
        result = import_claude_ai_export(tmp_path, profile="p")

    mock_import.assert_not_called()
    assert result["imported"] == 0
    assert "no turn-pairs" in result["warning"]


def test_import_claude_ai_export_propagates_path_and_mode(tmp_path):
    from ogham.claude_ai_import import import_claude_ai_export

    convs = [
        _conversation(
            "c1",
            "t",
            [_msg("human", "q?", "h1"), _msg("assistant", "real answer..." * 10, "a1")],
        )
    ]
    _write_export(tmp_path, convs)
    fake_result = {"status": "complete", "imported": 1, "skipped": 0, "total": 1, "profile": "p"}
    with patch("ogham.claude_ai_import._import_memories", return_value=fake_result):
        result = import_claude_ai_export(tmp_path, profile="p", mode="raw")

    assert result["path"] == str(tmp_path)
    assert result["mode"] == "raw"
