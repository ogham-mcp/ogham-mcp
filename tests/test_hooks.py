"""Tests for lifecycle hooks."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_dedup_cache():
    """Clear the dedup cache between tests."""
    from ogham.hooks import _recent_actions

    _recent_actions.clear()
    yield
    _recent_actions.clear()


def test_session_start_returns_context():
    from ogham.hooks import session_start

    mock_results = [
        {
            "content": "Decided to use PostgreSQL",
            "tags": ["type:decision"],
            "similarity": 0.8,
            "created_at": "2026-03-20T10:00:00Z",
        },
        {
            "content": "psycopg needs check=check_connection",
            "tags": ["type:gotcha"],
            "similarity": 0.7,
            "created_at": "2026-03-19T10:00:00Z",
        },
    ]
    with (
        patch("ogham.database.hybrid_search_memories", return_value=mock_results),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        result = session_start(cwd="/Users/dev/myproject", profile="work")

    assert "Decided to use PostgreSQL" in result
    assert "psycopg" in result
    assert "## Session Context" in result
    assert "2 memories loaded" in result


def test_session_start_empty_db():
    from ogham.hooks import session_start

    with (
        patch("ogham.database.hybrid_search_memories", return_value=[]),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        result = session_start(cwd="/tmp/empty", profile="work")

    assert result == ""


def test_post_tool_stores_action():
    from ogham.hooks import post_tool

    hook_input = {
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -m 'fix: update config'"},
        "session_id": "abc123",
        "cwd": "/Users/dev/myproject",
    }
    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(hook_input, profile="work")

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert "Bash" in content
    assert "git commit" in content


def test_post_tool_skips_ogham_tools():
    from ogham.hooks import post_tool

    for tool_name in [
        "mcp__ogham__hybrid_search",
        "mcp__ogham__store_memory",
        "ogham_search",
        "store_memory",
        "hybrid_search",
    ]:
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(
                {"tool_name": tool_name, "tool_input": {"query": "test"}},
                profile="work",
            )
        mock_store.assert_not_called(), f"{tool_name} should be skipped"


def test_post_tool_skips_always_skip_tools():
    """ToolSearch, Skill, Read, Glob, Grep, ListDir are always skipped."""
    from ogham.hooks import post_tool

    for tool_name in [
        "ToolSearch",
        "Skill",
        "Read",
        "Glob",
        "Grep",
        "ListDir",
        "TaskCreate",
        "TaskUpdate",
        "AskUserQuestion",
    ]:
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(
                {
                    "tool_name": tool_name,
                    "tool_input": {"content": "some content with error keywords"},
                    "cwd": "/tmp",
                    "session_id": "s1",
                },
                profile="work",
            )
        mock_store.assert_not_called(), f"{tool_name} should be always-skipped"


def test_post_tool_captures_high_value_tools():
    """Write, Edit, Agent, WebFetch are always captured."""
    from ogham.hooks import post_tool

    for tool_name in ["Write", "Edit", "Agent", "WebFetch"]:
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(
                {
                    "tool_name": tool_name,
                    "tool_input": {"content": "some content"},
                    "cwd": "/tmp",
                    "session_id": "s1",
                },
                profile="work",
            )
        mock_store.assert_called_once(), f"{tool_name} should be captured"


def test_post_tool_skips_noise_bash():
    """ls, cat, pwd etc. should not be captured."""
    from ogham.hooks import post_tool

    for cmd in ["ls", "pwd", "cat foo.txt", "head -5 bar", "echo hello"]:
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": cmd},
                    "cwd": "/tmp",
                    "session_id": "s1",
                },
                profile="work",
            )
        mock_store.assert_not_called(), f"'{cmd}' should be skipped as noise"


def test_post_tool_captures_signal_bash():
    """Bash commands with signal keywords should be captured."""
    from ogham.hooks import post_tool

    for cmd in [
        "git commit -m 'fix bug'",
        "docker build -t myapp .",
        "pytest tests/ -v",
        "railway deploy",
    ]:
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": cmd},
                    "cwd": "/tmp",
                    "session_id": "s1",
                },
                profile="work",
            )
        mock_store.assert_called_once(), f"'{cmd}' should be captured as signal"


def test_post_tool_tags_include_tool_name():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Write",
                "tool_input": {"content": "new file content"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )

    tags = mock_store.call_args.kwargs["tags"]
    assert "tool:Write" in tags
    assert "type:action" in tags
    assert "session:s1" in tags


def test_post_tool_dedup_same_file():
    """Repeated edits to the same file within 5 min should be collapsed."""
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        # First edit -- captured
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py", "content": "change 1"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        # Second edit to same file -- deduped
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py", "content": "change 2"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        # Edit to different file -- captured
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/bar.py", "content": "change 3"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )

    assert mock_store.call_count == 2, "Second edit to foo.py should be deduped"


def test_post_tool_dedup_different_sessions():
    """Same file in different sessions should not dedup."""
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py", "content": "change 1"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/tmp/foo.py", "content": "change 2"},
                "cwd": "/tmp",
                "session_id": "s2",
            },
            profile="work",
        )

    assert mock_store.call_count == 2, "Different sessions should not dedup"


def test_post_tool_content_format():
    """Content should use readable format, not verbose Tool:/Input:/Directory:."""
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/Users/dev/myproject/src/main.py",
                    "content": "new code",
                },
                "cwd": "/Users/dev/myproject",
                "session_id": "s1",
            },
            profile="work",
        )

    content = mock_store.call_args.kwargs["content"]
    # Should have basename, not full path as the main identifier
    assert "main.py" in content
    # Should include project name
    assert "myproject" in content
    # Should NOT have the old verbose format
    assert "Directory:" not in content


def test_pre_compact_stores_summary():
    from ogham.hooks import pre_compact

    with patch("ogham.service.store_memory_enriched") as mock_store:
        pre_compact(session_id="abc", cwd="/Users/dev/myproject", profile="work")

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert "myproject" in content
    assert "abc" in content
    tags = mock_store.call_args.kwargs["tags"]
    assert "compaction:drain" in tags


def test_post_compact_returns_context():
    from ogham.hooks import post_compact

    mock_results = [
        {
            "content": "API uses REST with JWT",
            "tags": ["type:architecture"],
            "similarity": 0.9,
            "created_at": "2026-03-20T10:00:00Z",
        },
    ]
    with (
        patch("ogham.database.hybrid_search_memories", return_value=mock_results),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        result = post_compact(cwd="/Users/dev/myproject", profile="work")

    assert "API uses REST" in result
    assert "## Restored Context" in result
    assert "1 memories restored" in result


def test_post_compact_empty_db():
    from ogham.hooks import post_compact

    with (
        patch("ogham.database.hybrid_search_memories", return_value=[]),
        patch("ogham.embeddings.generate_embedding", return_value=[0.1] * 512),
    ):
        result = post_compact(cwd="/tmp/empty", profile="work")

    assert result == ""


def test_session_start_handles_errors():
    from ogham.hooks import session_start

    with patch("ogham.embeddings.generate_embedding", side_effect=RuntimeError("no provider")):
        result = session_start(cwd="/tmp", profile="work")

    assert result == ""


def test_mask_secrets_key_value():
    from ogham.hooks import _mask_secrets

    # API keys with assignment
    assert "***MASKED***" in _mask_secrets("api_key=sk-proj-abc123def456")
    assert "sk-proj-abc123def456" not in _mask_secrets("api_key=sk-proj-abc123def456")

    # Passwords
    assert "***MASKED***" in _mask_secrets("password=mysecretpass123")
    assert "mysecretpass123" not in _mask_secrets("password=mysecretpass123")

    # Database URLs
    assert "***MASKED***" in _mask_secrets("database_url=postgresql://user:pass@host/db")

    # Env vars
    assert "***MASKED***" in _mask_secrets("VOYAGE_API_KEY=pa-TKy5ucfpkIp99_lU1JaxlblJxl7pNTldP")
    assert "***MASKED***" in _mask_secrets("OPENAI_API_KEY=sk-proj-joiu0bp96xKIxxxxxxx")


def test_mask_secrets_bare_tokens():
    from ogham.hooks import _mask_secrets

    # GitHub PAT (bare, no KEY=)
    assert "***MASKED***" in _mask_secrets("ghp_ABCDEFghijklmnopqrstuvwxyz1234567890")

    # AWS access key
    assert "***MASKED***" in _mask_secrets("AKIA1234567890ABCDEF")

    # Slack token (values deliberately fake to avoid GitHub push protection)
    assert "***MASKED***" in _mask_secrets(
        "xoxb-FAKETOKEN00-FAKETOKEN0000-FAKEabcdefghijklmnopqrst"
    )

    # Anthropic key
    assert "***MASKED***" in _mask_secrets("sk-ant-api03-abc123def456ghi789jkl")

    # SendGrid
    assert "***MASKED***" in _mask_secrets("SG.abcdefghijklmnopqrstuvwxyz1234567890")


def test_mask_secrets_url_credentials():
    from ogham.hooks import _mask_secrets

    masked = _mask_secrets("postgresql://admin:s3cretP4ss@db.example.com:5432/mydb")
    assert "s3cretP4ss" not in masked
    assert "***MASKED***" in masked


def test_mask_secrets_safe_content():
    from ogham.hooks import _mask_secrets

    assert _mask_secrets("deployed to railway") == "deployed to railway"
    assert _mask_secrets("git commit -m 'fix bug'") == "git commit -m 'fix bug'"
    assert _mask_secrets("Kevin Burns worked on Ogham") == "Kevin Burns worked on Ogham"
    assert _mask_secrets("ls -la /tmp/foo") == "ls -la /tmp/foo"


def test_post_tool_masks_secrets_before_storing():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Write",
                "tool_input": {"content": "api_key=sk-proj-abc123def456ghi"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )

    content = mock_store.call_args.kwargs["content"]
    assert "sk-proj-abc123def456ghi" not in content
    assert "***MASKED***" in content


def test_post_tool_handles_errors():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched", side_effect=RuntimeError("db down")):
        post_tool(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": "/tmp"},
            profile="work",
        )
