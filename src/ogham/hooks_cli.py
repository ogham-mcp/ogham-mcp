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


def _event_type(data: dict) -> str:
    raw = data.get("hook_event_name") or data.get("event") or data.get("event_name")
    if raw:
        return str(raw)
    if "tool_name" in data:
        return "PostToolUse"
    if _prompt_from_data(data):
        return "UserPromptSubmit"
    return ""


def _prompt_from_data(data: dict) -> str:
    for key in ("prompt", "user_prompt", "message"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _echo_dry_run(result: str | None) -> None:
    if result:
        typer.echo(result)
    else:
        typer.echo("No memory would be stored.")


@hooks_app.command(name="recall")
def recall_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
    force: bool = typer.Option(False, help="Skip debounce, always recall"),
    recall: bool | None = typer.Option(
        None,
        "--recall/--no-recall",
        help="Enable or disable recall for this hook invocation",
    ),
):
    """Read from the stone. Load relevant memories for the current project."""
    from ogham.flow_control import recall_enabled, temporary_flow_overrides
    from ogham.hooks import post_compact, session_start

    with temporary_flow_overrides(recall=recall):
        if not recall_enabled():
            return

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
    inscribe: bool | None = typer.Option(
        None,
        "--inscribe/--no-inscribe",
        help="Enable or disable inscribe for this hook invocation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview the memory that would be stored without writing it",
    ),
):
    """Carve into the stone. Capture activity or drain session before compaction."""
    from ogham.flow_control import inscribe_enabled, temporary_flow_overrides
    from ogham.hooks import post_tool, pre_compact, user_prompt_submit

    with temporary_flow_overrides(inscribe=inscribe):
        if not inscribe_enabled():
            return

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

        event = _event_type(data)

        if event == "UserPromptSubmit":
            result = user_prompt_submit(
                prompt=_prompt_from_data(data),
                cwd=data.get("cwd", "."),
                session_id=data.get("session_id", "unknown"),
                profile=profile,
                dry_run=dry_run,
            )
            if dry_run:
                _echo_dry_run(result)
        elif "tool_name" in data:
            result = post_tool(data, profile=profile, dry_run=dry_run)
            if dry_run:
                _echo_dry_run(result)
        else:
            # Otherwise treat as compaction drain
            result = pre_compact(
                session_id=data.get("session_id", "unknown"),
                cwd=data.get("cwd", "."),
                profile=profile,
                dry_run=dry_run,
            )
            if dry_run:
                _echo_dry_run(result)


@hooks_app.command(name="install")
def install_cmd():
    """Detect AI client and install hooks configuration."""
    from ogham.hooks_install import install_hooks

    install_hooks()
