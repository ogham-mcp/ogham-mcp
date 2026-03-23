"""Detect AI client and install hooks configuration."""

import json
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def _detect_client() -> str:
    """Detect which AI coding client is in use."""
    if (Path.home() / ".claude" / "settings.json").exists() or shutil.which("claude"):
        return "claude-code"

    if (Path.home() / ".kiro").exists() or shutil.which("kiro"):
        return "kiro"

    if (Path.home() / ".cursor").exists():
        return "cursor"

    if (Path.home() / ".codex").exists() or shutil.which("codex"):
        return "codex"

    return "generic"


def _install_claude_code():
    """Write Claude Code hooks to settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hooks = settings.setdefault("hooks", {})
    ogham_hooks = {
        "SessionStart": {
            "matcher": "",
            "hooks": [{"type": "command", "command": "ogham hooks recall"}],
        },
        "PostToolUse": {
            "matcher": "",
            "hooks": [{"type": "command", "command": "ogham hooks inscribe"}],
        },
        "PreCompact": {
            "matcher": "",
            "hooks": [{"type": "command", "command": "ogham hooks inscribe"}],
        },
        "PostCompact": {
            "matcher": "",
            "hooks": [{"type": "command", "command": "ogham hooks recall"}],
        },
    }

    for event, hook_entry in ogham_hooks.items():
        existing = hooks.get(event, [])
        # Don't duplicate if already installed
        ogham_cmd = hook_entry["hooks"][0]["command"]
        already = any(ogham_cmd in str(e.get("hooks", [])) for e in existing)
        if not already:
            existing.append(hook_entry)
        hooks[event] = existing

    settings["hooks"] = hooks
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    console.print(f"[green]Claude Code hooks installed to {settings_path}[/green]")
    console.print("  SessionStart, PostToolUse, PreCompact, PostCompact")


def _install_kiro():
    """Print Kiro hook instructions (Kiro uses UI-based config)."""
    console.print("[bold]Kiro hooks -- manual setup via Hook UI:[/bold]")
    console.print("")
    console.print("  1. Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)")
    console.print("  2. Type 'Kiro: Open Kiro Hook UI'")
    console.print("  3. Create these hooks:")
    console.print("")
    console.print("  [bold]Hook 1: Recall[/bold]")
    console.print("    Event: Prompt Submit")
    console.print("    Action: Run Command")
    console.print("    Command: ogham hooks recall")
    console.print("    [dim](Debounced -- only fires once per 30 minutes per project)[/dim]")
    console.print("")
    console.print("  [bold]Hook 2: Inscribe[/bold]")
    console.print("    Event: Post Task Execution")
    console.print("    Action: Run Command")
    console.print("    Command: ogham hooks inscribe")
    console.print("    [dim](Fires on meaningful task completions, not every chat)[/dim]")
    console.print("")
    console.print(
        "[dim]Kiro doesn't support compaction hooks. Session start covers context loading.[/dim]"
    )


def _detect_instruction_file() -> tuple[Path, str]:
    """Find the right project instruction file for the current client.

    Returns (path, display_name).
    """
    cwd = Path.cwd()

    # Codex uses AGENTS.md
    if (cwd / "AGENTS.md").exists():
        return cwd / "AGENTS.md", "AGENTS.md"

    # Cursor uses .cursorrules
    if (cwd / ".cursorrules").exists():
        return cwd / ".cursorrules", ".cursorrules"

    # Kiro uses kiro.md or .kiro/
    if (cwd / "kiro.md").exists():
        return cwd / "kiro.md", "kiro.md"

    # Claude Code uses CLAUDE.md (also works as universal fallback)
    return cwd / "CLAUDE.md", "CLAUDE.md"


def _install_generic(client: str = "generic"):
    """Add instructions to the project instruction file."""
    target, name = _detect_instruction_file()

    # If no file exists, pick based on detected client
    if not target.exists():
        if client == "cursor":
            target = Path.cwd() / ".cursorrules"
            name = ".cursorrules"
        elif client == "codex":
            target = Path.cwd() / "AGENTS.md"
            name = "AGENTS.md"

    hook_section = """
## Ogham Memory Hooks

At the start of every session, search Ogham for relevant context:
- Call `hybrid_search` with a query describing the current project
- Review the top results for decisions, gotchas, and patterns

When you finish significant work, save learnings:
- Use `store_memory` for decisions, gotchas, and architectural patterns
- Tag with type:decision, type:gotcha, type:pattern
"""

    if target.exists():
        content = target.read_text()
        if "Ogham Memory Hooks" in content:
            console.print(f"[yellow]{name} already has Ogham hook instructions[/yellow]")
            return
        with open(target, "a") as f:
            f.write(hook_section)
    else:
        target.write_text(hook_section)

    console.print(f"[green]{name} updated with Ogham hook instructions[/green]")
    console.print("  Works with Codex, Cursor, OpenCode, and any MCP client")


def install_hooks():
    """Detect client and install appropriate hooks."""
    client = _detect_client()
    console.print(f"Detected client: [bold]{client}[/bold]")
    console.print("")

    match client:
        case "claude-code":
            _install_claude_code()
        case "kiro":
            _install_kiro()
        case _:
            _install_generic(client)
            if client not in ("generic", "codex", "cursor"):
                console.print(
                    f"\n[dim]{client} doesn't support hooks natively."
                    " Instructions added to project file as fallback.[/dim]"
                )
