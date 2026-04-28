"""Lifecycle hooks for AI coding clients.

Called by `ogham hooks <event>` or shell wrappers. Each function reads
context from the Ogham database and either outputs markdown (for context
injection) or stores a memory (for capture).

Supported clients: Claude Code (native hooks), Kiro (Hook UI),
Codex/Cursor/OpenCode (CLAUDE.md fallback).
"""

import logging
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Lifecycle: advance_stages is scheduled to run off the hot path when a
# session starts. Imports hoisted to module top so tests can patch
# ogham.hooks.{advance_stages,lifecycle_submit} directly.
from ogham.lifecycle import advance_stages
from ogham.lifecycle_executor import submit as lifecycle_submit

logger = logging.getLogger(__name__)

# Tools we never capture (prevent infinite loops + pure noise)
_SKIP_PREFIXES = ("mcp__ogham__", "ogham_", "store_memory", "hybrid_search")

# Tools that are always noise -- never worth storing
_ALWAYS_SKIP_TOOLS = frozenset(
    {
        "ToolSearch",
        "Skill",
        "Read",
        "Glob",
        "Grep",
        "ListDir",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
        "Agent",
        "Monitor",
        "TaskCreate",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "AskUserQuestion",
        "SendMessage",
        "ScheduleWakeup",
        "CronCreate",
        "CronDelete",
        "CronList",
    }
)

_DEFAULT_RESPONSE_GATED_TOOLS = frozenset({"Edit", "Write"})
_HIGH_SIGNAL_TAGS = frozenset(
    {
        "type:code-change",
        "type:context",
        "type:correction",
        "type:decision",
        "type:deploy",
        "type:error",
        "type:fact",
        "type:git-signal",
        "type:preference",
    }
)
_QUESTION_STARTS = (
    "what ",
    "how ",
    "can you",
    "why ",
    "where ",
    "when ",
    "show me",
)
_PREFERENCE_MARKERS = (
    "prefer",
    "favorite",
    "favourite",
    "always use",
    "rather",
    "like better",
)
_DECISION_MARKERS = (
    "decided",
    "let's go with",
    "lets go with",
    "chose",
    "going with",
    "switching to",
)
_CORRECTION_MARKERS = (
    "actually",
    "correction",
    "wrong",
    "instead",
)
_PERSONAL_CONTEXT_MARKERS = (
    "i'm based in",
    "i am based in",
    "i work at",
    "my company",
    "my team",
)

# --- Config loading ---
_config_cache: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load hooks config from YAML file, with caching.

    Looks for hooks_config.yaml next to this module, then falls back
    to hardcoded defaults. Config is cached after first load.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = Path(__file__).parent / "hooks_config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                loaded = yaml.safe_load(f)
                _config_cache = loaded if isinstance(loaded, dict) else {}
                logger.debug("Loaded hooks config from %s", config_path)
                return _config_cache
        except ImportError:
            logger.debug("PyYAML not installed, using hardcoded defaults")
        except Exception as e:
            logger.debug("Failed to load hooks config: %s", e)

    # Return None to signal "use hardcoded defaults"
    _config_cache = {}
    return _config_cache


def _get_signal_keywords() -> frozenset[str]:
    """Get signal keywords from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "signals" in cfg:
        keywords = set()
        for category_words in cfg["signals"].values():
            if isinstance(category_words, list):
                keywords.update(category_words)
        return frozenset(keywords)
    return _DEFAULT_SIGNAL_KEYWORDS


def _get_noise_commands() -> frozenset[str]:
    """Get noise commands from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "noise_commands" in cfg:
        return frozenset(cfg["noise_commands"])
    return _DEFAULT_NOISE_COMMANDS


def _get_always_skip_tools() -> frozenset[str]:
    """Get always-skip tools from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "always_skip_tools" in cfg:
        return frozenset(cfg["always_skip_tools"])
    return _ALWAYS_SKIP_TOOLS


def _get_response_gated_tools() -> frozenset[str]:
    """Get tools captured only when their response/input yields a useful memory."""
    cfg = _load_config()
    if cfg and "response_gated_tools" in cfg:
        return frozenset(cfg["response_gated_tools"])
    return _DEFAULT_RESPONSE_GATED_TOOLS


def _get_routine_tools() -> frozenset[str]:
    """Get routine tools from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "routine_tools" in cfg:
        return frozenset(cfg["routine_tools"])
    return _DEFAULT_ROUTINE_TOOLS


def _get_git_signal() -> frozenset[str]:
    """Get git signal subcommands from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "git_signal" in cfg:
        return frozenset(cfg["git_signal"])
    return _DEFAULT_GIT_SIGNAL


def _get_git_noise() -> frozenset[str]:
    """Get git noise subcommands from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "git_noise" in cfg:
        return frozenset(cfg["git_noise"])
    return _DEFAULT_GIT_NOISE


def _build_bare_secret_patterns() -> re.Pattern:
    """Build bare token regex from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "secrets" in cfg and "bare_tokens" in cfg["secrets"]:
        patterns = [entry["pattern"] for entry in cfg["secrets"]["bare_tokens"]]
        combined = "|".join(f"(?:{p})" for p in patterns)
        return re.compile(combined)
    return _DEFAULT_BARE_SECRET_PATTERNS


def _get_env_secret_keys() -> frozenset[str]:
    """Get env secret key names from config or hardcoded defaults."""
    cfg = _load_config()
    if cfg and "secrets" in cfg and "env_keys" in cfg["secrets"]:
        return frozenset(cfg["secrets"]["env_keys"])
    return _DEFAULT_ENV_SECRET_KEYS


# --- Hardcoded defaults (used when YAML config is missing or PyYAML not installed) ---

_DEFAULT_ROUTINE_TOOLS = frozenset({"Bash"})

_DEFAULT_SIGNAL_KEYWORDS = frozenset(
    {
        # Errors and debugging
        "error",
        "fail",
        "fix",
        "bug",
        "broke",
        "crash",
        "exception",
        "traceback",
        "stacktrace",
        "segfault",
        "panic",
        # Decisions and changes
        "decided",
        "chose",
        "switch",
        "migrate",
        "replace",
        "refactor",
        "deprecated",
        "removed",
        "added",
        "changed",
        # Infrastructure
        "config",
        "deploy",
        "release",
        "install",
        "upgrade",
        "rollback",
        "permission",
        "denied",
        "timeout",
        "refused",
        "certificate",
        # DevOps
        "docker",
        "railway",
        "neon",
        "supabase",
        "vercel",
        "cloudflare",
        "terraform",
        "kubernetes",
        "k8s",
        "helm",
        # Testing
        "test",
        "pytest",
        "jest",
        "passed",
        "failed",
        "coverage",
        # Security
        "secret",
        "credential",
        "auth",
        "token",
        "vulnerability",
        "cve",
        # Database
        "migration",
        "schema",
        "index",
        "vacuum",
        "replication",
        # Workarounds
        "todo",
        "hack",
        "workaround",
        "gotcha",
        "caveat",
        "warning",
        # Package management
        "pip install",
        "npm install",
        "uv add",
        "go get",
        "cargo add",
    }
)

# Noise commands we never capture
_DEFAULT_NOISE_COMMANDS = frozenset(
    {
        "ls",
        "pwd",
        "cd",
        "cat",
        "head",
        "tail",
        "wc",
        "echo",
        "date",
        "whoami",
        "which",
        "type",
        "clear",
        "history",
    }
)

# Git subcommands worth capturing (commits, pushes, merges -- not maintenance)
_DEFAULT_GIT_SIGNAL = frozenset(
    {
        "commit",
        "push",
        "merge",
        "rebase",
        "tag",
        "release",
        "reset",
        "revert",
        "cherry-pick",
    }
)
# Git subcommands that are noise
_DEFAULT_GIT_NOISE = frozenset(
    {
        "add",
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "checkout",
        "switch",
        "fetch",
        "pull",
        "stash",
        "clean",
        "gc",
        "remote",
        "config",
    }
)

# Patterns that look like secrets -- mask these before storing.
# Two-tier approach:
# 1. _SECRET_PATTERNS: matches KEY=VALUE patterns (api_key=sk-proj-...)
# 2. _BARE_SECRET_PATTERNS: matches bare tokens without assignment (ghp_..., AKIA...)
# 3. _URL_CREDENTIALS: matches user:pass@host in URLs
# 4. _ENV_SECRET_KEYS: generic KEY=value matching for common env var names

_SECRET_PATTERNS = re.compile(
    r"(?i)"
    # Generic key=value patterns
    r"(?:api[_-]?key|secret[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token"
    r"|password|passwd|bearer|token"
    # Cloud provider prefixes
    r"|sk[_-]live|sk[_-]proj|pk[_-]live|sk[_-]test|pk[_-]test"
    # Service-specific prefixes
    r"|ghp_|gho_|github_pat_|glpat-|xoxb-|xoxp-|whsec_"
    r"|sb_secret_|ogham_live_"
    r"|SG\.[A-Za-z0-9_-]{20}"  # SendGrid
    r"|npm_[A-Za-z0-9]{20}"  # NPM
    r"|pypi-[A-Za-z0-9]{20}"  # PyPI
    # Voyage/Neon/custom
    r"|pa-[A-Za-z0-9_-]{20}"
    r"|npg_[A-Za-z0-9]{10}"
    # AWS
    r"|AKIA[A-Z0-9]{16}"
    # JWT
    r"|eyJ[A-Za-z0-9_-]{20,})"
    r"[=:\s]+\s*['\"]?([A-Za-z0-9_\-./+=]{8,})['\"]?"
)

# Bare tokens that don't need a KEY= prefix to be recognised
_DEFAULT_BARE_SECRET_PATTERNS = re.compile(
    r"(?:"
    r"ghp_[A-Za-z0-9]{36}"  # GitHub PAT
    r"|gho_[A-Za-z0-9]{36}"  # GitHub OAuth
    r"|github_pat_[A-Za-z0-9_]{20,}"  # GitHub fine-grained PAT
    r"|glpat-[A-Za-z0-9\-]{20,}"  # GitLab PAT
    r"|AKIA[A-Z0-9]{16}"  # AWS access key
    r"|SG\.[A-Za-z0-9_\-]{20,}"  # SendGrid
    r"|xox[bpars]-[A-Za-z0-9\-]{10,}"  # Slack tokens
    r"|sk-ant-[A-Za-z0-9\-]{20,}"  # Anthropic
    r"|sk-proj-[A-Za-z0-9\-]{20,}"  # OpenAI project
    r"|sk-[A-Za-z0-9]{40,}"  # OpenAI legacy
    r"|npm_[A-Za-z0-9]{36}"  # NPM
    r"|pypi-[A-Za-z0-9]{20,}"  # PyPI
    r"|whsec_[A-Za-z0-9]{20,}"  # Webhook secret (Clerk/Svix)
    r"|ogham_live_[A-Za-z0-9_\-]{20,}"  # Ogham API key
    r"|\d{8,12}:[A-Za-z0-9_-]{35}"  # Telegram bot token
    r"|[A-Za-z0-9]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}"  # Discord bot token
    r")"
)

# Basic auth in URLs: user:pass@host
_URL_CREDENTIALS = re.compile(r"://([^:]+):([^@]{3,})@")

_DEFAULT_ENV_SECRET_KEYS = frozenset(
    {
        "api_key",
        "secret_key",
        "access_key",
        "access_token",
        "auth_token",
        "password",
        "passwd",
        "bearer",
        "private_key",
        "database_url",
        "connection_string",
        "dsn",
        "redis_url",
        "valkey_url",
        "mongodb_uri",
        "encryption_key",
        "signing_key",
        "webhook_secret",
    }
)


@dataclass(frozen=True)
class _HookMemory:
    """A candidate memory extracted from one hook event."""

    content: str
    target: str = ""
    tags: tuple[str, ...] = ()


# --- Session dedup ---
# Track recent (tool, target) pairs to collapse repeated edits to the same file.
# Key: (session_id, tool_name, target_path) → timestamp
_recent_actions: dict[tuple[str, str, str], float] = {}
_DEDUP_WINDOW_SECONDS = 300  # 5 minutes


def _is_duplicate(session_id: str, tool_name: str, target: str) -> bool:
    """Check if this (tool, target) was already captured recently in this session."""
    import time

    key = (session_id, tool_name, target)
    now = time.time()

    # Prune old entries (> 30 min) to prevent unbounded growth
    stale = [k for k, t in _recent_actions.items() if now - t > 1800]
    for k in stale:
        del _recent_actions[k]

    if key in _recent_actions:
        if now - _recent_actions[key] < _DEDUP_WINDOW_SECONDS:
            _recent_actions[key] = now  # Refresh timestamp
            return True

    _recent_actions[key] = now
    return False


def _mask_secrets(text: str) -> str:
    """Replace anything that looks like a secret with a masked placeholder.

    Three detection layers:
    1. KEY=value patterns (api_key=sk-proj-...)
    2. Bare tokens without assignment (ghp_..., AKIA..., sk-ant-...)
    3. URL credentials (user:pass@host)
    4. Generic env var names (password=, database_url=)

    Captures the event ("set API key for Stripe") but never the value.
    """
    # Layer 1: KEY=value patterns
    masked = _SECRET_PATTERNS.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + "***MASKED***",
        text,
    )
    # Layer 2: Bare tokens (no KEY= prefix needed)
    masked = _build_bare_secret_patterns().sub("***MASKED***", masked)
    # Layer 3: URL credentials (user:pass@host)
    masked = _URL_CREDENTIALS.sub("://***MASKED***:***MASKED***@", masked)
    # Layer 4: Generic env var names
    for key in _get_env_secret_keys():
        pattern = re.compile(rf"(?i){re.escape(key)}\s*[=:]\s*['\"]?([^\s'\"]+)['\"]?")
        masked = pattern.sub(
            lambda m: m.group(0)[: m.start(1) - m.start(0)] + "***MASKED***",
            masked,
        )
    return masked


def _type_tags(memory: dict[str, Any]) -> list[str]:
    """Return display-safe type tags from a memory row."""
    raw_tags = memory.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [tag for tag in raw_tags if isinstance(tag, str) and tag.startswith("type:")]


def _squash(text: str, limit: int = 200) -> str:
    """Collapse whitespace and cap hook-derived snippets."""
    squashed = " ".join(text.strip().split())
    return squashed[:limit].rstrip()


def _string_literals(text: str) -> set[str]:
    return {
        match.group(1) or match.group(2)
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text)
        if match.group(1) or match.group(2)
    }


def _diff_change_size(old: str, new: str) -> int:
    """Approximate number of changed characters between two short snippets."""
    matcher = SequenceMatcher(a=old, b=new, autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    )


def _extract_assignment(line: str) -> tuple[str, str] | None:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*,?\s*$", line, re.S)
    if not match:
        return None
    return match.group(1), _squash(match.group(2), 80)


def _extract_edit_memory(tool_input: dict[str, Any], cwd: str) -> _HookMemory | None:
    file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    old = str(tool_input.get("old_string") or "")
    new = str(tool_input.get("new_string") or "")
    if not file_path or not old or not new or old == new:
        return None

    basename = os.path.basename(file_path)
    target = file_path

    old_literals = _string_literals(old)
    new_literals = _string_literals(new)
    added_literals = sorted(new_literals - old_literals)
    assignee = _extract_assignment(new) or _extract_assignment(old)
    if assignee and added_literals:
        added = ", ".join(added_literals[:5])
        return _HookMemory(
            f"{basename}: added {added} to {assignee[0]}",
            target=target,
            tags=("type:code-change",),
        )

    old_def = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", old)
    new_def = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", new)
    if old_def and new_def and old_def.group(1) == new_def.group(1):
        if old_def.group(2) != new_def.group(2):
            return _HookMemory(
                f"{basename}: changed signature for {new_def.group(1)}",
                target=target,
                tags=("type:code-change",),
            )

    old_assignment = _extract_assignment(old)
    new_assignment = _extract_assignment(new)
    if old_assignment and new_assignment and old_assignment[0] == new_assignment[0]:
        content = (
            f"{basename}: changed {new_assignment[0]} "
            f"from {old_assignment[1]} to {new_assignment[1]}"
        )
        return _HookMemory(
            content,
            target=target,
            tags=("type:code-change",),
        )

    if _diff_change_size(old, new) < 20:
        return None

    name_match = re.search(
        r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)|([A-Z_][A-Z0-9_]{2,})",
        new,
    )
    subject = name_match.group(1) or name_match.group(2) if name_match else ""
    if subject:
        content = f"{basename}: changed {subject}"
    else:
        content = f"{basename}: changed {_squash(new, 120)}"
    return _HookMemory(content, target=target, tags=("type:code-change",))


def _first_doc_or_comment(content: str) -> str:
    stripped = content.lstrip()
    doc_match = re.match(r'(?s)(?:"""|\'\'\')\s*(.+?)(?:"""|\'\'\')', stripped)
    if doc_match:
        return _squash(doc_match.group(1), 120)
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return _squash(line.lstrip("# "), 120)
        break
    return ""


def _extract_write_memory(
    tool_input: dict[str, Any],
    tool_response: str,
    cwd: str,
) -> _HookMemory | None:
    file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    content = str(tool_input.get("content") or "")
    if not file_path or not content:
        return None

    response_lower = tool_response.lower()
    if any(word in response_lower for word in ("overwrote", "overwrite", "updated existing")):
        return None
    if response_lower and not any(
        word in response_lower for word in ("created", "new file", "wrote")
    ):
        return None
    if not response_lower:
        return None

    basename = os.path.basename(file_path)
    description = _first_doc_or_comment(content)
    if description:
        memory = f"created {basename}: {description}"
    else:
        memory = f"created {basename}"
    return _HookMemory(memory, target=file_path, tags=("type:code-change",))


def _parse_shell(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _git_commit_message(parts: list[str]) -> str | None:
    for i, part in enumerate(parts):
        if part in ("-m", "--message") and i + 1 < len(parts):
            return parts[i + 1]
        if part.startswith("-m") and len(part) > 2:
            return part[2:]
        if part.startswith("--message="):
            return part.split("=", 1)[1]
    return None


def _extract_error_line(tool_response: str) -> str | None:
    for line in tool_response[:2000].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"\b\w*(?:Error|Exception)\b|Traceback|FAILED|error:", stripped, re.I):
            stripped = re.sub(r"\b(\w*(?:Error|Exception)):\s*", r"\1 ", stripped)
            return _squash(stripped, 200)
    return None


def _extract_publish_outcome(command: str, tool_response: str) -> str:
    combined = f"{command}\n{tool_response[:2000]}"
    version = re.search(r"\b(?:Published|version|v)(?:\s+)?v?(\d+\.\d+\.\d+)\b", combined, re.I)
    if version:
        return f"published {version.group(1)}"
    return f"deploy/release command completed: {_squash(command, 120)}"


def _extract_gh_memory(parts: list[str], tool_response: str) -> _HookMemory | None:
    if len(parts) < 3 or parts[0] != "gh":
        return None
    noun, action = parts[1], parts[2]
    if noun == "pr" and action == "merge":
        pr_number = next((p for p in parts[3:] if p.isdigit()), "")
        mode = "squash" if "--squash" in parts else "merge"
        suffix = f" #{pr_number}" if pr_number else ""
        return _HookMemory(f"merged PR{suffix} ({mode})", tags=("type:decision",))
    if noun == "pr" and action in {"create", "close"}:
        title_match = re.search(r"title:\s*(.+)", tool_response, re.I)
        title = f": {_squash(title_match.group(1), 120)}" if title_match else ""
        return _HookMemory(f"{action}d PR{title}", tags=("type:decision",))
    if noun == "issue" and action == "close":
        issue_number = next((p for p in parts[3:] if p.isdigit()), "")
        suffix = f" #{issue_number}" if issue_number else ""
        return _HookMemory(f"closed issue{suffix}", tags=("type:decision",))
    if noun == "release" and action == "create":
        version = parts[3] if len(parts) > 3 else ""
        suffix = f" {version}" if version else ""
        return _HookMemory(f"created GitHub release{suffix}", tags=("type:deploy",))
    return None


def _extract_bash_memory(
    tool_input: dict[str, Any],
    tool_response: str,
    cwd: str,
) -> _HookMemory | None:
    command = str(tool_input.get("command") or "")
    if not command:
        return None
    target = str(tool_input.get("file_path") or tool_input.get("path") or "")
    command_lower = command.lower().strip()
    parts = _parse_shell(command)
    first = parts[0].lower() if parts else ""

    if first in _get_noise_commands():
        return None

    exit_code = tool_input.get("exit_code")
    if exit_code is None:
        exit_code = tool_input.get("return_code")
    error_line = _extract_error_line(tool_response)
    if error_line and (exit_code not in (0, "0") or "error" in error_line.lower()):
        return _HookMemory(f"error: {error_line}", target=target, tags=("type:error",))

    gh_memory = _extract_gh_memory(parts, tool_response)
    if gh_memory:
        return _HookMemory(gh_memory.content, target=target, tags=gh_memory.tags)

    if first == "git" and len(parts) > 1:
        subcommand = parts[1].lower()
        if subcommand in _get_git_noise():
            return None
        if subcommand == "commit":
            message = _git_commit_message(parts)
            if not message:
                response_match = re.search(r"\[[^\]]+\]\s+(.+)", tool_response)
                message = response_match.group(1) if response_match else None
            if message:
                return _HookMemory(
                    f"git commit: {_squash(message, 160)}",
                    target=target,
                    tags=("type:decision",),
                )
        if subcommand in _get_git_signal():
            return _HookMemory(
                f"git {subcommand}: {_squash(command, 160)}",
                target=target,
                tags=("type:git-signal",),
            )
        if not any(kw in command_lower for kw in _get_signal_keywords()):
            return None

    if re.search(r"\b(publish|deploy|release)\b", command_lower):
        return _HookMemory(
            _extract_publish_outcome(command, tool_response),
            target=target,
            tags=("type:deploy",),
        )

    if first == "gh":
        return None

    if not any(kw in command_lower for kw in _get_signal_keywords()):
        return None
    return _HookMemory(f"Bash: {_squash(command, 160)}", target=target, tags=("type:action",))


def _extract_memory_content(
    event_type: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: str,
    cwd: str,
    prompt: str = "",
) -> _HookMemory | None:
    """Extract a useful memory from a hook event."""
    if event_type == "UserPromptSubmit":
        return _extract_user_prompt_memory(prompt, cwd)

    if tool_name == "Edit":
        return _extract_edit_memory(tool_input, cwd)
    if tool_name == "Write":
        return _extract_write_memory(tool_input, tool_response, cwd)
    if tool_name == "Bash":
        return _extract_bash_memory(tool_input, tool_response[:2000], cwd)
    return None


def _classify_user_prompt(prompt: str) -> tuple[bool, tuple[str, ...]]:
    prompt_lower = prompt.lower().strip()
    tags: list[str] = []

    if any(marker in prompt_lower for marker in _PREFERENCE_MARKERS):
        tags.append("type:preference")
    if any(marker in prompt_lower for marker in _DECISION_MARKERS):
        tags.append("type:decision")
    if any(marker in prompt_lower for marker in _CORRECTION_MARKERS):
        tags.append("type:correction")
    if any(marker in prompt_lower for marker in _PERSONAL_CONTEXT_MARKERS):
        tags.append("type:context")

    try:
        from ogham.extraction import extract_dates

        if extract_dates(prompt):
            tags.append("type:fact")
    except Exception:
        logger.debug("user prompt date extraction failed", exc_info=True)

    return bool(tags), tuple(dict.fromkeys(tags))


def _extract_user_prompt_memory(prompt: str, cwd: str) -> _HookMemory | None:
    """Extract durable user-stated context from natural-language prompts."""
    prompt = _squash(prompt, 500)
    if len(prompt) < 30:
        return None

    prompt_lower = prompt.lower().strip()
    if any(prompt_lower.startswith(q) for q in _QUESTION_STARTS):
        return None

    has_signal, tags = _classify_user_prompt(prompt)
    if not has_signal:
        return None

    return _HookMemory(prompt, tags=tags)


def _passes_importance_gate(content: str, tags: list[str]) -> bool:
    """Reuse Ogham's existing no-LLM importance heuristic for hook captures."""
    if any(tag in _HIGH_SIGNAL_TAGS for tag in tags):
        return True
    try:
        from ogham.extraction import compute_importance

        return compute_importance(content) >= 0.3
    except Exception:
        return True


def session_start(
    cwd: str,
    profile: str = "work",
    limit: int = 8,
) -> str:
    """Return markdown context for session injection.

    Searches for memories relevant to the current working directory and
    schedules a lifecycle advancement sweep on the background executor.
    The sweep is fire-and-forget -- session starts even if it fails.
    """
    from ogham.database import hybrid_search_memories
    from ogham.embeddings import generate_embedding

    project_name = os.path.basename(cwd)
    query = f"project context for {project_name}"

    # Schedule the lifecycle sweep BEFORE the search. It runs on the
    # background thread pool; the search proceeds on this thread. Any
    # exception the sweep raises is logged inside lifecycle_submit.
    try:
        lifecycle_submit(advance_stages, profile)
    except Exception:
        logger.warning("lifecycle: failed to schedule advance_stages sweep", exc_info=True)

    # Topic-summary stale sweep (T1.4 Phase 7). 30-day idle threshold
    # catches summaries the Phase 6 mutation hooks missed: raw SQL edits,
    # bulk imports, or the all-sources-deleted edge case. Runs on the
    # same lifecycle executor -- fire-and-forget, session proceeds
    # even if it fails.
    try:
        from ogham.topic_summaries import sweep_stale_summaries

        lifecycle_submit(sweep_stale_summaries, profile)
    except Exception:
        logger.warning("topic_summaries: failed to schedule nightly stale sweep", exc_info=True)

    try:
        embedding = generate_embedding(query)
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
        )
    except Exception:
        logger.debug("session_start: search failed, returning empty")
        return ""

    if not results:
        return ""

    lines = ["## Session Context", ""]
    for r in results:
        content = str(r.get("content", ""))[:200]
        tags = _type_tags(r)
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories loaded for {project_name}*")
    return "\n".join(lines)


def _event_type(hook_input: dict[str, Any]) -> str:
    raw = (
        hook_input.get("hook_event_name") or hook_input.get("event") or hook_input.get("event_name")
    )
    if raw:
        return str(raw)
    if hook_input.get("tool_name"):
        return "PostToolUse"
    if hook_input.get("prompt") or hook_input.get("user_prompt"):
        return "UserPromptSubmit"
    return ""


def post_tool(
    hook_input: dict[str, Any],
    profile: str = "work",
    dry_run: bool = False,
) -> str | None:
    """Capture a tool execution as a memory.

    Skips Ogham's own tools to prevent infinite loops.
    """
    tool_name = str(hook_input.get("tool_name", ""))

    # Skip Ogham's own tools (infinite loop prevention)
    if any(tool_name.startswith(p) for p in _SKIP_PREFIXES):
        return None

    # Skip tools that are always noise (reconnaissance, not action)
    if tool_name in _get_always_skip_tools():
        return None

    tool_input = hook_input.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {"input": tool_input}
    tool_response = str(
        hook_input.get("tool_response")
        or hook_input.get("response")
        or hook_input.get("tool_output")
        or hook_input.get("output")
        or ""
    )
    cwd = str(hook_input.get("cwd", ""))
    session_id = str(hook_input.get("session_id", ""))

    if tool_name not in _get_routine_tools() and tool_name not in _get_response_gated_tools():
        return None

    extracted = _extract_memory_content(
        "PostToolUse",
        tool_name,
        tool_input,
        tool_response[:2000],
        cwd,
    )
    if extracted is None:
        return None

    # Dedup: skip if same (tool, target) was captured recently in this session
    if extracted.target and not dry_run and _is_duplicate(session_id, tool_name, extracted.target):
        logger.debug("post_tool: dedup skip %s on %s", tool_name, extracted.target)
        return None

    # Mask any secrets before storing
    content = _mask_secrets(extracted.content)

    if cwd:
        project = os.path.basename(cwd)
        content += f" [{project}]"

    tags = [
        "type:action",
        f"tool:{tool_name}",
        f"session:{session_id}",
        *extracted.tags,
    ]
    if not _passes_importance_gate(content, list(extracted.tags)):
        return None

    if dry_run:
        return content

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:post-tool",
            tags=tags,
        )
    except Exception:
        logger.debug("post_tool: store failed, ignoring")
    return content


def user_prompt_submit(
    prompt: str,
    cwd: str,
    session_id: str = "",
    profile: str = "work",
    dry_run: bool = False,
) -> str | None:
    """Capture durable context from a user's natural-language prompt."""
    extracted = _extract_memory_content("UserPromptSubmit", "", {}, "", cwd, prompt=prompt)
    if extracted is None:
        return None

    content = _mask_secrets(extracted.content)
    if cwd:
        project = os.path.basename(cwd)
        content += f" [{project}]"

    tags = ["type:prompt", f"session:{session_id}", *extracted.tags]
    if not _passes_importance_gate(content, list(extracted.tags)):
        return None

    if dry_run:
        return content

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:user-prompt-submit",
            tags=tags,
        )
    except Exception:
        logger.debug("user_prompt_submit: store failed, ignoring")
    return content


def pre_compact(
    session_id: str,
    cwd: str,
    profile: str = "work",
    dry_run: bool = False,
) -> str | None:
    """Drain session context to Ogham before compaction."""
    project_name = os.path.basename(cwd)
    timestamp = datetime.now(timezone.utc).isoformat()

    content = (
        f"Session drain before compaction.\n"
        f"Project: {project_name}\n"
        f"Directory: {cwd}\n"
        f"Session: {session_id}\n"
        f"Time: {timestamp}"
    )

    if dry_run:
        return content

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:pre-compact",
            tags=["type:session", f"session:{session_id}", "compaction:drain"],
        )
    except Exception:
        logger.debug("pre_compact: store failed, ignoring")
    return content


def post_compact(
    cwd: str,
    profile: str = "work",
    limit: int = 10,
) -> str:
    """Rehydrate context after compaction.

    Returns markdown with the most relevant memories for the project.
    """
    from ogham.database import hybrid_search_memories
    from ogham.embeddings import generate_embedding

    project_name = os.path.basename(cwd)
    query = f"recent work and decisions for {project_name}"

    try:
        embedding = generate_embedding(query)
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
        )
    except Exception:
        logger.debug("post_compact: search failed, returning empty")
        return ""

    if not results:
        return ""

    lines = ["## Restored Context", ""]
    for r in results:
        content = str(r.get("content", ""))[:300]
        tags = _type_tags(r)
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories restored for {project_name} after compaction*")
    return "\n".join(lines)
