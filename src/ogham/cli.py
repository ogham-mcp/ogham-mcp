"""CLI interface for Ogham memory operations."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ogham.config import settings
from ogham.database import cleanup_expired as db_cleanup_expired
from ogham.database import count_expired as db_count_expired
from ogham.database import (
    get_memory_stats,
    get_profile_ttl,
    hybrid_search_memories,
    list_recent_memories,
)
from ogham.database import list_profiles as db_list_profiles
from ogham.database import store_memory as db_store
from ogham.embeddings import generate_embedding
from ogham.health import full_health_check

app = typer.Typer(
    name="ogham",
    help="Ogham Shared Memory — persistent memory for AI clients.",
    invoke_without_command=True,
)
console = Console()


def _run_server():
    from ogham.server import main as server_main

    server_main()


@app.callback()
def main_callback(ctx: typer.Context):
    """Ogham MCP — persistent memory for AI clients."""
    if ctx.invoked_subcommand is None:
        _run_server()


@app.command()
def serve():
    """Start the MCP server (default behavior)."""
    _run_server()


@app.command()
def store(
    content: str = typer.Argument(help="The text content to remember"),
    profile: str = typer.Option(None, help="Profile to store in"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Tags for the memory"),
    source: str = typer.Option("cli", help="Source identifier"),
):
    """Store a new memory."""
    from datetime import datetime, timedelta, timezone

    target = profile or settings.default_profile
    embedding = generate_embedding(content)

    ttl_days = get_profile_ttl(target)
    expires_at = None
    if ttl_days is not None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=ttl_days)
        ).isoformat()

    result = db_store(
        content=content,
        embedding=embedding,
        profile=target,
        source=source,
        tags=tags,
        expires_at=expires_at,
    )
    console.print(f"[green]Stored memory {result['id']} in profile '{target}'[/green]")
    if expires_at:
        console.print(f"[dim]Expires: {expires_at[:19]}[/dim]")


@app.command()
def health():
    """Check connectivity to Supabase and embedding provider."""
    result = full_health_check()

    table = Table(title="Health Check")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    for component, details in result.items():
        status = details.get("status", "unknown")
        style = "green" if status == "ok" else "red" if status == "error" else "yellow"
        info = {k: v for k, v in details.items() if k != "status"}
        table.add_row(component, f"[{style}]{status}[/{style}]", str(info) if info else "")

    console.print(table)


@app.command()
def profiles():
    """List all memory profiles and their counts."""
    data = db_list_profiles()

    table = Table(title="Profiles")
    table.add_column("Profile", style="bold")
    table.add_column("Memories", justify="right")

    for row in data:
        table.add_row(row["profile"], str(row["count"]))

    console.print(table)


@app.command()
def stats(profile: str = typer.Option(None, help="Profile to show stats for")):
    """Show statistics for a memory profile."""
    target = profile or settings.default_profile
    data = get_memory_stats(profile=target)

    console.print(f"\n[bold]Profile:[/bold] {data.get('profile', target)}")
    console.print(f"[bold]Total memories:[/bold] {data.get('total', 0)}")

    sources = data.get("sources") or {}
    if sources:
        source_str = ", ".join(f"{k}: {v}" for k, v in sources.items())
        console.print(f"[bold]Sources:[/bold] {source_str}")

    top_tags = data.get("top_tags") or []
    if top_tags:
        tag_str = ", ".join(f"{t['tag']} ({t['count']})" for t in top_tags[:10])
        console.print(f"[bold]Top tags:[/bold] {tag_str}")

    console.print()


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
    profile: str = typer.Option(None, help="Profile to search"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Filter by tag"),
):
    """Search memories by meaning and keywords (hybrid search)."""
    target = profile or settings.default_profile
    embedding = generate_embedding(query)
    results = hybrid_search_memories(
        query_text=query,
        query_embedding=embedding,
        profile=target,
        limit=limit,
        tags=tags,
    )

    if not results:
        console.print("[yellow]No matching memories found.[/yellow]")
        return

    table = Table(title=f"Search Results ({len(results)} matches)")
    table.add_column("Relevance", justify="right", width=10)
    table.add_column("Content")
    table.add_column("Tags")

    for r in results:
        relevance = f"{r.get('relevance', 0):.3f}"
        content = r["content"][:120]
        tags_str = ", ".join(r.get("tags", []))
        table.add_row(relevance, content, tags_str)

    console.print(table)


@app.command(name="list")
def list_memories(
    limit: int = typer.Option(10, help="Max results"),
    profile: str = typer.Option(None, help="Profile to list"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Filter by tag"),
    source: Optional[str] = typer.Option(None, help="Filter by source"),
):
    """List recent memories."""
    target = profile or settings.default_profile
    results = list_recent_memories(profile=target, limit=limit, source=source, tags=tags)

    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(title=f"Recent Memories ({len(results)})")
    table.add_column("Created", width=20)
    table.add_column("Content")
    table.add_column("Tags")
    table.add_column("Source")

    for r in results:
        table.add_row(
            r.get("created_at", "")[:19],
            r["content"][:100],
            ", ".join(r.get("tags", [])),
            r.get("source", ""),
        )

    console.print(table)


@app.command()
def cleanup(
    profile: str = typer.Option(None, help="Profile to clean"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove expired memories."""
    target = profile or settings.default_profile
    count = db_count_expired(target)

    if count == 0:
        console.print(f"[green]No expired memories in profile '{target}'.[/green]")
        return

    console.print(f"Found [bold]{count}[/bold] expired memories in profile '{target}'.")

    if not yes:
        confirm = typer.confirm("Delete them?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    deleted = db_cleanup_expired(target)
    console.print(f"[green]Deleted {deleted} expired memories.[/green]")


@app.command(name="export")
def export_cmd(
    profile: str = typer.Option(None, help="Profile to export"),
    format: str = typer.Option("json", help="Output format: json or markdown"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (stdout if omitted)"
    ),
):
    """Export memories from a profile."""
    from ogham.export_import import export_memories

    target = profile or settings.default_profile
    data = export_memories(target, format=format)

    if output:
        with open(output, "w") as f:
            f.write(data)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(data)


@app.command(name="import")
def import_cmd(
    file: str = typer.Argument(help="JSON file to import"),
    profile: str = typer.Option(None, help="Profile to import into"),
    dedup: float = typer.Option(0.8, help="Dedup threshold (0 to disable)"),
):
    """Import memories from a JSON export file."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    from ogham.export_import import import_memories

    target = profile or settings.default_profile

    with open(file) as f:
        data = f.read()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        embed_task = progress.add_task("Embedding...", total=None)
        dedup_task = progress.add_task("Deduplicating...", total=None, visible=False)

        def on_embed_progress(embedded, total):
            if progress.tasks[embed_task].total is None:
                progress.update(embed_task, total=total)
            progress.update(embed_task, completed=embedded)

        def on_progress(imported, skipped, total):
            progress.update(embed_task, visible=False)
            if progress.tasks[dedup_task].total is None:
                progress.update(dedup_task, total=total, visible=True)
            progress.update(
                dedup_task,
                completed=imported + skipped,
                description=f"Processing ({imported} new, {skipped} skipped)",
            )

        result = import_memories(
            data,
            profile=target,
            dedup_threshold=dedup,
            on_progress=on_progress,
            on_embed_progress=on_embed_progress,
        )

    console.print(
        f"[green]Imported {result['imported']} memories, "
        f"skipped {result['skipped']} duplicates.[/green]"
    )


@app.command()
def openapi(
    output: str = typer.Option("docs/openapi.json", help="Output file path"),
):
    """Generate OpenAPI spec from MCP tool definitions."""
    from ogham.openapi import write_openapi_spec

    write_openapi_spec(output)
    console.print(f"[green]OpenAPI spec written to {output}[/green]")


def main():
    try:
        app()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
