# Ogham MCP — Hooks CLI & Installation

Two modules handle hook lifecycle management for Claude Code integration.

## `hooks_cli.py` — CLI Commands

Typer-based CLI for managing Claude Code hooks. Commands:

### `install`
- Calls `hooks_install.install_hooks()`
- Prints success/failure status with rich formatting

### `recall`
- Runs recall hooks for session start / post-compaction context
- Supports `--recall/--no-recall` for one-off flow control
- Also honours `OGHAM_RECALL_ENABLED`

### `inscribe`
- Runs inscribe hooks for post-tool capture / pre-compaction drains
- Supports `--inscribe/--no-inscribe` for one-off flow control
- Also honours `OGHAM_INSCRIBE_ENABLED`

### `uninstall`
- Calls `hooks_install.uninstall_hooks()`
- Removes all Ogham hook entries from Claude Code settings

### `status`
- Reads Claude Code settings file
- Checks for presence of Ogham hooks in each event slot
- Reports installed/not-installed per hook event

### `test`
- Runs a quick integration test of the hook pipeline
- Calls `post_tool` with a synthetic Bash tool event containing signal keywords
- Verifies the hook processes it correctly (returns store vs skip decision)

## `hooks_install.py` — Hook Registration

Manages the Claude Code `settings.json` hooks configuration.

### Settings File Location
- Primary: `~/.claude/settings.json`
- Creates file/directory if not present

### Hook Events Registered

| Event | Hook Command | Timeout |
|-------|-------------|---------|
| `PreToolUse` | — | — |
| `PostToolUse` | `uvx ogham-mcp hooks post-tool` | 30s |
| `SessionStart` | `uvx ogham-mcp hooks session-start` | 15s |
| `PreCompact` | `uvx ogham-mcp hooks pre-compact` | 30s |
| `PostCompact` | `uvx ogham-mcp hooks post-compact` | 15s |

### `install_hooks()`
1. Reads existing settings (or creates empty)
2. For each hook event, checks if an Ogham hook already exists
3. Appends new hook entry with `type: "command"`, the command string, and timeout
4. Writes back to settings file
5. Returns list of what was installed vs already present

### `uninstall_hooks()`
1. Reads existing settings
2. Filters out any hook whose command contains `ogham`
3. Removes empty hook event arrays
4. Writes back cleaned settings

### Hook Entry Format
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "type": "command",
        "command": "uvx ogham-mcp hooks post-tool",
        "timeout": 30000
      }
    ]
  }
}
```

### Safety
- Preserves all non-Ogham hooks during install/uninstall
- Creates backup of settings before modification
- Idempotent — re-running install skips already-installed hooks
