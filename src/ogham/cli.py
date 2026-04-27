"""CLI interface for Ogham memory operations."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ogham",
    help="Ogham Shared Memory — persistent memory for AI clients.",
    invoke_without_command=True,
)
console = Console()


def _safe_text(value: object, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    return text[:limit] if limit is not None else text


def _run_server(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    from ogham.server import main as server_main

    server_main(transport=transport, host=host, port=port)


@app.callback()
def main_callback(ctx: typer.Context):
    """Ogham MCP — persistent memory for AI clients."""
    if ctx.invoked_subcommand is None:
        _run_server()


@app.command()
def serve(
    transport: Optional[str] = typer.Option(None, help="Transport: stdio or sse"),
    host: Optional[str] = typer.Option(None, help="SSE bind host (default 127.0.0.1)"),
    port: Optional[int] = typer.Option(None, help="SSE port (default 8742)"),
):
    """Start the MCP server."""
    _run_server(transport=transport, host=host, port=port)


@app.command()
def store(
    content: str = typer.Argument(help="The text content to remember"),
    profile: str = typer.Option(None, help="Profile to store in"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Tags for the memory"),
    tags_csv: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    source: str = typer.Option("cli", help="Source identifier"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich text"),
):
    """Store a new memory."""
    from ogham.config import settings
    from ogham.service import store_memory_enriched

    target = profile or settings.default_profile

    merged_tags = list(tags or [])
    if tags_csv:
        merged_tags.extend(t.strip() for t in tags_csv.split(",") if t.strip())

    result = store_memory_enriched(
        content=content,
        profile=target,
        source=source,
        tags=merged_tags or None,
    )

    if output_json:
        print(json.dumps(result, default=str))
        return

    console.print(f"[green]Stored memory {result['id']} in profile '{target}'[/green]")
    if result.get("expires_at"):
        console.print(f"[dim]Expires: {_safe_text(result['expires_at'], 19)}[/dim]")
    if result.get("conflicts"):
        console.print(f"[yellow]{result['conflict_warning']}[/yellow]")
        for c in result["conflicts"]:
            preview = _safe_text(c.get("content_preview", ""), 80)
            similarity = float(c.get("similarity", 0))
            console.print(
                f"  [dim]{_safe_text(c.get('id'), 8)}... ({similarity:.0%}) {preview}[/dim]"
            )


@app.command()
def config(
    output_json: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show current runtime configuration (secrets masked)."""
    from ogham.tools.stats import get_runtime_config

    data = get_runtime_config()

    if output_json:
        print(json.dumps(data, default=str, indent=2))
        return

    for section, values in data.items():
        if section == "config_sources":
            console.print("\n[bold]Config loaded from:[/bold]")
            for src in values:
                console.print(f"  {src}")
            continue
        console.print(f"\n[bold]{section}[/bold]")
        if isinstance(values, dict):
            for k, v in values.items():
                if v is not None:
                    console.print(f"  {k}: {v}")
        else:
            console.print(f"  {values}")


@app.command()
def health():
    """Check connectivity to Supabase and embedding provider."""
    from ogham.health import full_health_check

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
    from ogham.database import list_profiles as db_list_profiles

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
    from ogham.config import settings
    from ogham.database import get_memory_stats

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
    tags_csv: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
    extract: bool = typer.Option(False, "--extract", help="Extract query-relevant facts via LLM"),
):
    """Search memories by meaning and keywords (hybrid search)."""
    from ogham.config import settings

    merged_tags = list(tags or [])
    if tags_csv:
        merged_tags.extend(t.strip() for t in tags_csv.split(",") if t.strip())

    target = profile or settings.default_profile

    if extract:
        from ogham.service import search_memories_enriched

        results = search_memories_enriched(
            query=query,
            profile=target,
            limit=limit,
            tags=merged_tags or None,
            extract_facts=True,
        )
    else:
        from ogham.database import hybrid_search_memories
        from ogham.embeddings import generate_embedding

        embedding = generate_embedding(query)
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=target,
            limit=limit,
            tags=merged_tags or None,
        )

    if not results:
        if output_json:
            print("[]")
        else:
            console.print("[yellow]No matching memories found.[/yellow]")
        return

    if output_json:
        print(json.dumps(results, default=str))
        return

    table = Table(title=f"Search Results ({len(results)} matches)")
    table.add_column("ID", width=8)
    table.add_column("Relevance", justify="right", width=10)
    table.add_column("Content")
    table.add_column("Tags")

    for r in results:
        mem_id = str(r.get("id", ""))[:8]
        relevance = f"{r.get('relevance', 0):.3f}"
        content = r["content"][:120]
        tags_str = ", ".join(r.get("tags", []))
        table.add_row(mem_id, relevance, content, tags_str)

    console.print(table)


@app.command(name="list")
def list_memories(
    limit: int = typer.Option(10, help="Max results"),
    profile: str = typer.Option(None, help="Profile to list"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Filter by tag"),
    tags_csv: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    source: Optional[str] = typer.Option(None, help="Filter by source"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich table"),
):
    """List recent memories."""
    from ogham.config import settings
    from ogham.database import list_recent_memories

    merged_tags = list(tags or [])
    if tags_csv:
        merged_tags.extend(t.strip() for t in tags_csv.split(",") if t.strip())

    target = profile or settings.default_profile
    results = list_recent_memories(
        profile=target, limit=limit, source=source, tags=merged_tags or None
    )

    if not results:
        if output_json:
            print("[]")
        else:
            console.print("[yellow]No memories found.[/yellow]")
        return

    if output_json:
        print(json.dumps(results, default=str))
        return

    table = Table(title=f"Recent Memories ({len(results)})")
    table.add_column("ID", width=8)
    table.add_column("Created", width=20)
    table.add_column("Content")
    table.add_column("Tags")
    table.add_column("Source")

    for r in results:
        table.add_row(
            _safe_text(r.get("id", ""), 8),
            _safe_text(r.get("created_at", ""), 19),
            _safe_text(r.get("content", ""), 100),
            ", ".join(r.get("tags", [])),
            _safe_text(r.get("source", "")),
        )

    console.print(table)


@app.command()
def delete(
    memory_id: str = typer.Argument(help="Memory ID (full UUID or prefix)"),
    profile: str = typer.Option(None, help="Profile the memory belongs to"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a memory by ID."""
    from ogham.config import settings
    from ogham.database import delete_memory as db_delete

    target = profile or settings.default_profile

    if not yes:
        confirm = typer.confirm(f"Delete memory {memory_id[:8]}... from '{target}'?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    deleted = db_delete(memory_id, target)
    if deleted:
        console.print(f"[green]Deleted memory {memory_id[:8]}...[/green]")
    else:
        console.print(f"[red]Memory {memory_id[:8]}... not found in profile '{target}'.[/red]")


@app.command()
def use(
    profile: str = typer.Argument(help="Profile name to set as default"),
):
    """Set the default profile for subsequent commands."""
    from pathlib import Path

    env_file = Path.home() / ".ogham" / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing env file or start fresh
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    env_vars["DEFAULT_PROFILE"] = profile

    env_file.write_text("\n".join(f"{k}={v}" for k, v in env_vars.items()) + "\n")
    console.print(f"[green]Default profile set to '{profile}'[/green]")
    console.print(f"[dim]Saved to {env_file}[/dim]")


@app.command()
def cleanup(
    profile: str = typer.Option(None, help="Profile to clean"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove expired memories."""
    from ogham.config import settings
    from ogham.database import cleanup_expired as db_cleanup_expired
    from ogham.database import count_expired as db_count_expired

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
    from ogham.config import settings
    from ogham.export_import import export_memories

    target = profile or settings.default_profile
    data = export_memories(target, format=format)

    if output:
        with open(output, "w") as f:
            f.write(data)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(data)


@app.command(name="export-obsidian")
def export_obsidian_cmd(
    vault: str = typer.Argument(help="Path to the Obsidian vault directory"),
    profile: Optional[str] = typer.Option(None, help="Profile to export (default: active)"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite a vault directory that already contains non-export files",
    ),
):
    """Export wiki topic summaries to an Obsidian-compatible vault.

    Writes one markdown file per topic_summary (with YAML frontmatter
    and Obsidian wikilinks) plus a README.md index. Read-only -- the
    vault is a snapshot, not a sync target.
    """
    from pathlib import Path

    from ogham.config import settings
    from ogham.exporters.obsidian import export_to_vault

    target = profile or settings.default_profile
    result = export_to_vault(Path(vault), target, force=force)

    if result.errors:
        for err in result.errors:
            console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Wrote {result.topics_written} topic(s) to {result.vault_path}[/green]")
    if result.skipped:
        console.print(f"[yellow]Skipped: {', '.join(result.skipped)}[/yellow]")


@app.command(name="import")
def import_cmd(
    file: str = typer.Argument(help="JSON file to import"),
    profile: str = typer.Option(None, help="Profile to import into"),
    dedup: float = typer.Option(0.8, help="Dedup threshold (0 to disable)"),
):
    """Import memories from a JSON export file."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    from ogham.config import settings
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
def init(
    db_url: str = typer.Option(None, help="PostgreSQL connection string"),
    provider: str = typer.Option(None, help="Embedding provider (ollama/openai/voyage/mistral)"),
    api_key: str = typer.Option(None, help="Embedding provider API key"),
    backend: str = typer.Option(None, help="Database backend (supabase/postgres)"),
    supabase_url: str = typer.Option(None, help="Supabase project URL"),
    supabase_key: str = typer.Option(None, help="Supabase anon key"),
    dim: int = typer.Option(None, help="Embedding dimensions (default: 512)"),
    mode: str = typer.Option(None, help="Execution mode (uvx/docker)"),
    skip_schema: bool = typer.Option(False, help="Skip schema migration"),
    skip_clients: bool = typer.Option(False, help="Skip MCP client configuration"),
    skip_test: bool = typer.Option(False, help="Skip connection test"),
):
    """Interactive setup wizard. Configures database, embeddings, and MCP clients."""
    from ogham.init_wizard import run_init

    run_init(
        db_url=db_url,
        provider=provider,
        api_key=api_key,
        backend=backend,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        dim=dim,
        mode=mode,
        skip_schema=skip_schema,
        skip_clients=skip_clients,
        skip_test=skip_test,
    )


@app.command()
def openapi(
    output: str = typer.Option("docs/openapi.json", help="Output file path"),
):
    """Generate OpenAPI spec from MCP tool definitions."""
    from ogham.openapi import write_openapi_spec

    write_openapi_spec(output)
    console.print(f"[green]OpenAPI spec written to {output}[/green]")


def _register_subcommands():
    """Register sub-command groups (lazy to avoid import-time overhead)."""
    from ogham.hooks_cli import hooks_app

    app.add_typer(hooks_app)


_register_subcommands()


@app.command()
def decay(
    profile: str = typer.Option(None, help="Profile to decay"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Count eligible memories without decaying"
    ),
    batch_size: int = typer.Option(1000, help="Max memories to decay per run"),
):
    """Apply Hebbian decay to memories that haven't been accessed recently."""
    from ogham.config import settings
    from ogham.database import apply_hebbian_decay, count_decay_eligible

    target = profile or settings.default_profile

    if dry_run:
        eligible = count_decay_eligible(target)
        console.print(f"[cyan]{eligible} memories eligible for decay in profile '{target}'[/cyan]")
        return

    decayed = apply_hebbian_decay(target, batch_size=batch_size)
    console.print(f"[green]Decayed {decayed} memories in profile '{target}'[/green]")


@app.command()
def audit(
    profile: str = typer.Option(None, help="Profile to query"),
    limit: int = typer.Option(20, help="Max events"),
    operation: str = typer.Option(None, help="Filter by operation (store/search/delete/update)"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """View audit trail for a memory profile."""
    from ogham.config import settings
    from ogham.database import query_audit_log

    target = profile or settings.default_profile
    events = query_audit_log(target, limit=limit, operation=operation)

    if not events:
        if output_json:
            print("[]")
        else:
            console.print("[yellow]No audit events found.[/yellow]")
        return

    if output_json:
        print(json.dumps(events, default=str))
        return

    table = Table(title=f"Audit Trail ({len(events)} events)")
    table.add_column("Time", width=19)
    table.add_column("Op", width=8)
    table.add_column("Resource", width=10)
    table.add_column("Outcome", width=8)
    table.add_column("Results", width=8)
    table.add_column("Source", width=12)

    for e in events:
        event_time = str(e.get("event_time", ""))[:19]
        op = e.get("operation", "")
        resource = str(e.get("resource_id", "") or "")[:10]
        outcome = e.get("outcome", "")
        result_count = str(e.get("result_count", "") or "")
        source_val = e.get("source", "") or ""
        table.add_row(event_time, op, resource, outcome, result_count, source_val)

    console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(3113, help="Port to serve the dashboard on"),
    profile: str | None = typer.Option(
        None,
        help="Memory profile to display (defaults to DEFAULT_PROFILE / settings.default_profile).",
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
):
    """Start a visual dashboard in your browser. Requires ogham-mcp[dashboard]."""
    try:
        import uvicorn

        from ogham.config import settings
        from ogham.dashboard_server import create_app
    except ImportError:
        console.print(
            "[red]Dashboard requires extra dependencies.[/red]\n"
            "Install with: pip install ogham-mcp[dashboard]"
        )
        raise typer.Exit(1)

    # Fall back to the configured default profile when --profile is not
    # passed. A hardcoded "default" default would override DEFAULT_PROFILE
    # from env / config.env, which is surprising and broke the Go CLI's
    # profile handoff -- see
    # docs/plans/2026-04-16-go-cli-enterprise.md for the diagnosis.
    if not profile:
        profile = settings.default_profile

    dashboard_app = create_app(profile=profile)
    console.print(f"[green]Ogham dashboard ({profile}) → http://{host}:{port}[/green]")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="warning")


def main():
    try:
        app()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
