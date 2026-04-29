"""Tests for lifecycle hooks."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_dedup_cache():
    """Clear the dedup cache between tests."""
    from ogham.flow_control import clear_flow_overrides
    from ogham.hooks import _recent_actions

    clear_flow_overrides()
    _recent_actions.clear()
    yield
    clear_flow_overrides()
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


def test_session_start_disabled_skips_recall_side_effects():
    from ogham.flow_control import temporary_flow_overrides
    from ogham.hooks import session_start

    with temporary_flow_overrides(recall=False):
        with (
            patch("ogham.database.hybrid_search_memories") as search,
            patch("ogham.embeddings.generate_embedding") as embed,
            patch("ogham.hooks.lifecycle_submit") as lifecycle,
        ):
            result = session_start(cwd="/Users/dev/myproject", profile="work")

    assert result == ""
    search.assert_not_called()
    embed.assert_not_called()
    lifecycle.assert_not_called()


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
    assert "git commit: fix: update config" in content


def test_post_tool_disabled_skips_store():
    from ogham.flow_control import temporary_flow_overrides
    from ogham.hooks import post_tool

    hook_input = {
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -m 'fix: update config'"},
        "session_id": "abc123",
        "cwd": "/Users/dev/myproject",
    }
    with temporary_flow_overrides(inscribe=False):
        with patch("ogham.service.store_memory_enriched") as mock_store:
            post_tool(hook_input, profile="work")

    mock_store.assert_not_called()


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
        assert mock_store.call_count == 0, f"{tool_name} should be skipped"


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
        assert mock_store.call_count == 0, f"{tool_name} should be always-skipped"


def test_post_tool_skips_non_response_gated_noise_tools():
    """Agent and WebFetch remain skipped -- too noisy for hook capture."""
    from ogham.hooks import post_tool

    for tool_name in ["Agent", "WebFetch"]:
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
        assert mock_store.call_count == 0, f"{tool_name} should be skipped"


def test_post_tool_captures_edit_added_to_collection():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/src/ogham/hooks.py",
                    "old_string": 'ALWAYS_SKIP_TOOLS = frozenset({"Read", "Glob"})',
                    "new_string": (
                        'ALWAYS_SKIP_TOOLS = frozenset({"Read", "Glob", "Edit", "Write"})'
                    ),
                },
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == "hooks.py: added Edit, Write to ALWAYS_SKIP_TOOLS [ogham-mcp]"
    tags = mock_store.call_args.kwargs["tags"]
    assert "tool:Edit" in tags
    assert "type:code-change" in tags


def test_post_tool_captures_edit_assignment_change():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/src/ogham/config.py",
                    "old_string": "x = 1",
                    "new_string": "x = 2",
                },
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == "config.py: changed x from 1 to 2 [ogham-mcp]"


def test_post_tool_skips_tiny_typo_edit():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/src/ogham/README.md",
                    "old_string": "teh",
                    "new_string": "the",
                },
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_not_called()


def test_post_tool_captures_write_new_file_docstring():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/src/ogham/dashboard_server.py",
                    "content": (
                        '"""FastAPI standalone dashboard server."""\n\n'
                        "from fastapi import FastAPI\n"
                    ),
                },
                "tool_response": "Created file",
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == (
        "created dashboard_server.py: FastAPI standalone dashboard server. [ogham-mcp]"
    )


def test_post_tool_skips_write_overwrite():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/src/ogham/dashboard_server.py",
                    "content": "# rewritten file\n",
                },
                "tool_response": "Updated existing file",
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_not_called()


def test_post_tool_skips_write_without_creation_response():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/src/ogham/dashboard_server.py",
                    "content": '"""FastAPI standalone dashboard server."""\n',
                },
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_not_called()


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
        assert mock_store.call_count == 0, f"'{cmd}' should be skipped as noise"


def test_post_tool_captures_signal_bash():
    """Bash commands matching the verb-based YAML signal lists should be captured.

    The YAML config was tightened on 2026-04-22 to drop noun-heavy
    categories (docker/pytest/railway/etc.) because they fired on every
    daily infra command. Signal capture is now verbs-only: errors,
    decisions (past-tense), architecture verbs, git_signal subcommands,
    and explicit annotations. Test commands updated to match.
    """
    from ogham.hooks import post_tool

    for cmd in [
        "git commit -m 'fix bug'",  # git_signal=commit
        "git push origin main",  # git_signal=push
        "git merge --no-ff feature/wiki",  # git_signal=merge
        "git revert HEAD",  # git_signal=revert
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
        assert mock_store.call_count == 1, f"{cmd!r} should be captured as signal"


def test_post_tool_captures_bash_error_from_response():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "uv run pytest", "exit_code": 1},
                "tool_response": (
                    "ImportError: cannot import name 'task_redis_prefix' "
                    "from 'fastmcp.server.tasks.keys'"
                ),
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content.startswith("error: ImportError cannot import name")
    tags = mock_store.call_args.kwargs["tags"]
    assert "type:error" in tags


def test_post_tool_captures_bash_deploy_outcome():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "uv publish --token pypi-FAKEFAKEFAKEFAKEFAKEFAKE"},
                "tool_response": "Published 0.10.2",
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == "published 0.10.2 [ogham-mcp]"
    tags = mock_store.call_args.kwargs["tags"]
    assert "type:deploy" in tags


def test_post_tool_captures_gh_pr_merge():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "gh pr merge 37 --squash"},
                "tool_response": "Merged pull request #37",
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == "merged PR #37 (squash) [ogham-mcp]"


@pytest.mark.parametrize(
    ("command", "response", "expected"),
    [
        (
            "gh pr create --title 'Add dashboard hooks'",
            "title: Add dashboard hooks\nurl: https://github.com/x/y/pull/12",
            "created PR: Add dashboard hooks [ogham-mcp]",
        ),
        (
            "gh pr close 12",
            "title: Close stale dashboard hooks PR",
            "closed PR: Close stale dashboard hooks PR [ogham-mcp]",
        ),
        (
            "gh issue close 44",
            "Closed issue #44",
            "closed issue #44 [ogham-mcp]",
        ),
    ],
)
def test_post_tool_captures_other_gh_actions(command, response, expected):
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": command},
                "tool_response": response,
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == expected
    tags = mock_store.call_args.kwargs["tags"]
    assert "type:decision" in tags


def test_post_tool_captures_gh_release_create():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "gh release create v0.10.3"},
                "tool_response": "Created release v0.10.3",
                "cwd": "/Users/dev/ogham-mcp",
                "session_id": "s1",
            },
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert content == "created GitHub release v0.10.3 [ogham-mcp]"
    tags = mock_store.call_args.kwargs["tags"]
    assert "type:deploy" in tags


def test_post_tool_dry_run_does_not_store_or_dedup():
    from ogham.hooks import post_tool

    hook_input = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "/src/ogham/config.py",
            "old_string": "x = 1",
            "new_string": "x = 2",
        },
        "cwd": "/Users/dev/ogham-mcp",
        "session_id": "s1",
    }

    with patch("ogham.service.store_memory_enriched") as mock_store:
        preview = post_tool(hook_input, profile="work", dry_run=True)
        post_tool(hook_input, profile="work")

    assert preview == "config.py: changed x from 1 to 2 [ogham-mcp]"
    mock_store.assert_called_once()


@pytest.mark.parametrize(
    ("prompt", "expected_tag"),
    [
        ("I prefer PostgreSQL over MySQL for this project", "type:preference"),
        ("let's go with vendor X because the pricing is better", "type:decision"),
        ("actually the API key goes in the header, not body", "type:correction"),
        ("the project deadline is April 30 2026 for the launch", "type:fact"),
        ("I'm based in Germany and work at Aldi on platform tooling", "type:context"),
    ],
)
def test_user_prompt_submit_captures_normal_user_signals(prompt, expected_tag):
    from ogham.hooks import user_prompt_submit

    with patch("ogham.service.store_memory_enriched") as mock_store:
        user_prompt_submit(
            prompt=prompt,
            cwd="/Users/dev/ogham-mcp",
            session_id="s1",
            profile="work",
        )

    mock_store.assert_called_once()
    content = mock_store.call_args.kwargs["content"]
    assert prompt in content
    tags = mock_store.call_args.kwargs["tags"]
    assert "type:prompt" in tags
    assert expected_tag in tags


@pytest.mark.parametrize(
    "prompt",
    [
        "yes",
        "what database should we use for the project?",
        "please run the tests again without changing anything",
    ],
)
def test_user_prompt_submit_skips_low_signal_prompts(prompt):
    from ogham.hooks import user_prompt_submit

    with patch("ogham.service.store_memory_enriched") as mock_store:
        result = user_prompt_submit(
            prompt=prompt,
            cwd="/Users/dev/ogham-mcp",
            session_id="s1",
            profile="work",
        )

    assert result is None
    mock_store.assert_not_called()


def test_user_prompt_submit_dry_run_does_not_store():
    from ogham.hooks import user_prompt_submit

    with patch("ogham.service.store_memory_enriched") as mock_store:
        preview = user_prompt_submit(
            prompt="I prefer PostgreSQL over MySQL for this project",
            cwd="/Users/dev/ogham-mcp",
            session_id="s1",
            profile="work",
            dry_run=True,
        )

    assert preview == "I prefer PostgreSQL over MySQL for this project [ogham-mcp]"
    mock_store.assert_not_called()


def test_post_tool_tags_include_tool_name():
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'deploy fix'"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )

    tags = mock_store.call_args.kwargs["tags"]
    assert "tool:Bash" in tags
    assert "type:action" in tags
    assert "type:decision" in tags
    assert "session:s1" in tags


def test_post_tool_dedup_same_file():
    """Repeated Bash commits on the same file within 5 min should be collapsed."""
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix deploy'", "file_path": "/tmp/foo.py"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix again'", "file_path": "/tmp/foo.py"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix bar'", "file_path": "/tmp/bar.py"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )

    assert mock_store.call_count == 2, "Second commit on foo.py should be deduped"


def test_post_tool_dedup_different_sessions():
    """Same file in different sessions should not dedup."""
    from ogham.hooks import post_tool

    with patch("ogham.service.store_memory_enriched") as mock_store:
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix'", "file_path": "/tmp/foo.py"},
                "cwd": "/tmp",
                "session_id": "s1",
            },
            profile="work",
        )
        post_tool(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'fix'", "file_path": "/tmp/foo.py"},
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
                "tool_name": "Bash",
                "tool_input": {
                    "command": "git push origin main",
                },
                "cwd": "/Users/dev/myproject",
                "session_id": "s1",
            },
            profile="work",
        )

    content = mock_store.call_args.kwargs["content"]
    assert "git push" in content
    assert "myproject" in content
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


def test_pre_compact_disabled_skips_store():
    from ogham.flow_control import temporary_flow_overrides
    from ogham.hooks import pre_compact

    with temporary_flow_overrides(inscribe=False):
        with patch("ogham.service.store_memory_enriched") as mock_store:
            pre_compact(session_id="abc", cwd="/Users/dev/myproject", profile="work")

    mock_store.assert_not_called()


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


def test_post_compact_disabled_skips_recall():
    from ogham.flow_control import temporary_flow_overrides
    from ogham.hooks import post_compact

    with temporary_flow_overrides(recall=False):
        with (
            patch("ogham.database.hybrid_search_memories") as search,
            patch("ogham.embeddings.generate_embedding") as embed,
        ):
            result = post_compact(cwd="/Users/dev/myproject", profile="work")

    assert result == ""
    search.assert_not_called()
    embed.assert_not_called()


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
    assert "***MASKED***" in _mask_secrets("VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
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
                "tool_name": "Bash",
                "tool_input": {"command": "git commit -m 'config api_key=sk-proj-abc123def456ghi'"},
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
