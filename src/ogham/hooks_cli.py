"""CLI sub-commands for ogham hooks."""

import json
import os
import select
import sys

import typer

hooks_app = typer.Typer(name="hooks", help="Lifecycle hooks for AI coding clients.")


def _read_stdin() -> dict:
    """Read hook input JSON from stdin with timeout.

    Kiro's Agent Stop event may not pipe JSON or may not close stdin,
    so we use select() to avoid blocking forever.
    """
    try:
        if sys.stdin.isatty():
            return {}
        ready, _, _ = select.select([sys.stdin], [], [], 1.0)
        if not ready:
            return {}
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, Exception):
        return {}


def _should_recall() -> bool:
    """Debounce: only recall once per 30 minutes per directory.

    Kiro's Prompt Submit fires on every prompt. Without debounce,
    recall would search Ogham on every single message.
    """
    import time
    from pathlib import Path

    debounce_dir = Path.home() / ".ogham"
    debounce_dir.mkdir(parents=True, exist_ok=True)

    # Hash the cwd to create a per-project debounce file
    import hashlib

    cwd_hash = hashlib.md5(os.getcwd().encode(), usedforsecurity=False).hexdigest()[:8]
    marker = debounce_dir / f".recall_{cwd_hash}"

    now = time.time()
    if marker.exists():
        last_run = float(marker.read_text().strip())
        if now - last_run < 1800:  # 30 minutes
            return False

    marker.write_text(str(now))
    return True


@hooks_app.command(name="recall")
def recall_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
    force: bool = typer.Option(False, help="Skip debounce, always recall"),
):
    """Read from the stone. Load relevant memories for the current project."""
    from ogham.hooks import post_compact, session_start

    # Debounce: only recall once per 30 min (Kiro fires on every prompt)
    if not force and not _should_recall():
        return

    data = _read_stdin()
    cwd = data.get("cwd", ".")

    output = session_start(cwd=cwd, profile=profile)
    if not output:
        output = post_compact(cwd=cwd, profile=profile)
    if output:
        typer.echo(output)


@hooks_app.command(name="inscribe")
def inscribe_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
):
    """Carve into the stone. Capture activity or drain session before compaction."""
    from ogham.hooks import post_tool, pre_compact

    data = _read_stdin()

    if not data:
        # Kiro Agent Stop or no stdin -- create a minimal marker
        import os

        data = {
            "tool_name": "AgentStop",
            "tool_input": {"summary": "Agent turn completed"},
            "cwd": os.getcwd(),
            "session_id": "kiro",
        }

    # If it looks like a tool call, capture as post_tool
    if "tool_name" in data:
        post_tool(data, profile=profile)
    else:
        # Otherwise treat as compaction drain
        pre_compact(
            session_id=data.get("session_id", "unknown"),
            cwd=data.get("cwd", "."),
            profile=profile,
        )


@hooks_app.command(name="install")
def install_cmd():
    """Detect AI client and install hooks configuration."""
    from ogham.hooks_install import install_hooks

    install_hooks()
