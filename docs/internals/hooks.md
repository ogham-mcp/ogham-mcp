# Ogham MCP — Hook System

`hooks.py` integrates with Claude Code's lifecycle hooks to automatically capture context from AI coding sessions.

## Hook Events

### `session_start(cwd, profile, limit)`
- Searches ogham for memories relevant to the current project directory
- Returns markdown with top matches for context injection
- Used to prime the AI with project-specific knowledge at session start
- Returns nothing when recall is disabled

### `post_tool(hook_input, profile)`
- Called after every tool execution in a Claude Code session
- Decides whether the tool execution is worth storing as a memory
- Implements multi-layer filtering to avoid noise
- No-ops when inscribe is disabled

### `pre_compact(session_id, cwd, profile)`
- Drains session context to ogham before Claude Code compacts conversation
- Stores a timestamped "session drain" marker
- No-ops when inscribe is disabled

### `post_compact(cwd, profile, limit)`
- Rehydrates context after compaction
- Searches for recent decisions and work related to the project
- Returns markdown for re-injection
- Returns nothing when recall is disabled

## Filtering Pipeline (post_tool)

The `post_tool` hook applies several filters before storing:

### 1. Self-skip
- Skips any tool starting with `mcp__ogham__`, `ogham_`, `store_memory`, `hybrid_search`
- Prevents infinite loops

### 2. Always-skip tools
- `Read`, `Glob`, `Grep`, `ListDir` — reconnaissance, not action
- `TaskCreate`, `TaskUpdate`, `TaskGet`, `TaskList`, `TaskOutput` — task management noise
- `ToolSearch`, `Skill`, `AskUserQuestion`

### 3. Noise command filtering (Bash)
- Skips: `ls`, `pwd`, `cd`, `cat`, `head`, `tail`, `wc`, `echo`, `date`, `whoami`, `which`, `type`, `clear`, `history`

### 4. Git filtering
- **Signal** (capture): `commit`, `push`, `merge`, `rebase`, `tag`, `release`, `reset`, `revert`, `cherry-pick`
- **Noise** (skip): `add`, `status`, `diff`, `log`, `show`, `branch`, `checkout`, `switch`, `fetch`, `pull`, `stash`, `clean`, `gc`, `remote`, `config`
- Unknown git subcommands: only captured if they contain signal keywords

### 5. Signal keyword requirement (Bash)
- For routine tools (currently just Bash), content must contain at least one signal keyword to be captured
- Keywords cover: errors, decisions, infrastructure, DevOps, testing, security, database, workarounds, package management

### 6. Session dedup
- Tracks `(session_id, tool_name, target_path)` tuples with timestamps
- Same `(tool, target)` within 5 minutes is suppressed
- Entries older than 30 minutes are pruned

## Secret Masking (`_mask_secrets`)

Four detection layers applied before any content is stored:

1. **KEY=value patterns**: `api_key=sk-proj-...`, `password=...`, `bearer=...`
2. **Bare tokens**: `ghp_*` (GitHub PAT), `sk-ant-*` (Anthropic), `AKIA*` (AWS), `xox*-*` (Slack), `sk-proj-*` (OpenAI), Telegram bot tokens, Discord bot tokens, NPM/PyPI tokens, etc.
3. **URL credentials**: `://user:password@host`
4. **Generic env var names**: `password=`, `database_url=`, `private_key=`, `webhook_secret=`, etc.

All matches are replaced with `***MASKED***`. This runs on every store path — MCP tools, hooks, gateway, and CLI.

## Configuration

Hooks config is loaded from `hooks_config.yaml` (YAML file adjacent to `hooks.py`). Falls back to hardcoded defaults if YAML or PyYAML is unavailable. The config controls:
- Signal keywords
- Noise commands
- Always-skip tools
- Routine tools
- Git signal/noise subcommands
- Secret detection patterns
- Env secret key names

Runtime flow controls are separate from the YAML signal filters:

```bash
OGHAM_RECALL_ENABLED=false ogham hooks recall
OGHAM_INSCRIBE_ENABLED=false ogham hooks inscribe
ogham hooks recall --no-recall
ogham hooks inscribe --no-inscribe
```

Use `OGHAM_RECALL_ENABLED=false` in an MCP client config to prevent memory-derived context from reaching the LLM. Use `OGHAM_INSCRIBE_ENABLED=false` to prevent hook capture and content writes to Ogham. Admin commands remain available for inspection and cleanup.
