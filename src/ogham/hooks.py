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
from datetime import datetime, timezone
from pathlib import Path

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
        "Edit",
        "Write",
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

# --- Config loading ---
_config_cache: dict | None = None


def _load_config() -> dict:
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
                _config_cache = yaml.safe_load(f)
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

# High-value tool input fields to extract for summaries
_SUMMARY_FIELDS = ("command", "content", "query", "file_path", "url", "message")

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


def session_start(cwd: str, profile: str = "work", limit: int = 8) -> str:
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
        content = r.get("content", "")[:200]
        tags = [t for t in r.get("tags", []) if t.startswith("type:")]
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories loaded for {project_name}*")
    return "\n".join(lines)


def post_tool(hook_input: dict, profile: str = "work") -> None:
    """Capture a tool execution as a memory.

    Skips Ogham's own tools to prevent infinite loops.
    """
    tool_name = hook_input.get("tool_name", "")

    # Skip Ogham's own tools (infinite loop prevention)
    if any(tool_name.startswith(p) for p in _SKIP_PREFIXES):
        return

    # Skip tools that are always noise (reconnaissance, not action)
    if tool_name in _get_always_skip_tools():
        return

    tool_input = hook_input.get("tool_input", {})
    cwd = hook_input.get("cwd", "")
    session_id = hook_input.get("session_id", "")

    # Extract summary from tool input
    summary = ""
    target_path = ""
    if isinstance(tool_input, dict):
        # Grab the target file/path for dedup
        target_path = str(tool_input.get("file_path", tool_input.get("path", "")))
        for field in _SUMMARY_FIELDS:
            if field in tool_input:
                summary = str(tool_input[field])[:200]
                break
        if not summary:
            summary = str(tool_input)[:200]
    else:
        summary = str(tool_input)[:200]

    summary_lower = summary.lower()

    # Skip pure noise commands (ls, pwd, cat, etc.)
    git_signal_match = False
    if tool_name == "Bash":
        parts = summary_lower.strip().split()
        cmd_word = parts[0] if parts else ""
        if cmd_word in _get_noise_commands():
            return
        if cmd_word in ("git", "gh") and len(parts) > 1:
            git_sub = parts[1]
            if git_sub in _get_git_noise():
                return
            if git_sub in _get_git_signal():
                git_signal_match = True
            elif cmd_word == "gh":
                git_signal_match = True
            elif not any(kw in summary_lower for kw in _get_signal_keywords()):
                return

    if tool_name in _get_routine_tools() and not git_signal_match:
        if not any(kw in summary_lower for kw in _get_signal_keywords()):
            return

    # Dedup: skip if same (tool, target) was captured recently in this session
    if target_path and _is_duplicate(session_id, tool_name, target_path):
        logger.debug("post_tool: dedup skip %s on %s", tool_name, target_path)
        return

    # Mask any secrets before storing
    summary = _mask_secrets(summary)

    # Build content -- use file basename for readability
    target_display = os.path.basename(target_path) if target_path else ""
    if target_display:
        content = f"{tool_name} {target_display}: {summary[:150]}"
    else:
        content = f"{tool_name}: {summary[:150]}"
    if cwd:
        project = os.path.basename(cwd)
        content += f" [{project}]"

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:post-tool",
            tags=["type:action", f"tool:{tool_name}", f"session:{session_id}"],
        )
    except Exception:
        logger.debug("post_tool: store failed, ignoring")


def pre_compact(session_id: str, cwd: str, profile: str = "work") -> None:
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


def post_compact(cwd: str, profile: str = "work", limit: int = 10) -> str:
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
        content = r.get("content", "")[:300]
        tags = [t for t in r.get("tags", []) if t.startswith("type:")]
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories restored for {project_name} after compaction*")
    return "\n".join(lines)
