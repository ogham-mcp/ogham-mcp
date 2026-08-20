"""CLI interface for Ogham memory operations."""

import json
import re
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ogham.importers.beads import BeadsClient  # re-exported for CLI test monkeypatch
from ogham.importers.github_issues import GitHubClient  # re-exported for CLI test monkeypatch
from ogham.importers.slack import SlackClient  # re-exported for CLI test monkeypatch
from ogham.importers.telegram import TelegramClient  # re-exported for CLI test monkeypatch
from ogham.ingest import DefaultIngestService  # re-exported for CLI test monkeypatch

app = typer.Typer(
    name="ogham",
    help="Ogham Shared Memory — persistent memory for AI clients.",
    invoke_without_command=True,
)
console = Console()


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


# Downloadable model archives, pinned by digest. `sha256` and `size` are the
# values GitHub itself publishes for the release asset, so anyone can check
# them independently:
#
#   gh api repos/yuniko-software/bge-m3-onnx/releases/tags/1.01 \
#     --jq '.assets[] | "\(.name) \(.size) \(.digest)"'
#
# Both were also verified locally by downloading the archive and hashing it.
# Without this, `download-model` fetches 1.3 GB over the network and extracts it
# with no integrity check at all -- a tampered or truncated archive would be
# unpacked and used to generate embeddings.
MODEL_REGISTRY: dict[str, dict] = {
    "bge-m3": {
        "url": "https://github.com/yuniko-software/bge-m3-onnx/releases/download/1.01/onnx.zip",
        "sha256": "fef1d045ace47593bd7f149be2bfd72658625ad2786b0d3a79a90d48f7e5ed8e",
        "size": 1322654161,
        "expected_files": ["bge_m3_model.onnx", "bge_m3_model.onnx_data"],
    },
}


def verify_archive(path, expected_sha256: str, expected_size: int) -> list[str]:
    """Return integrity violations for a downloaded archive; empty means good.

    Size is checked first because it is free and catches the common case -- a
    truncated download -- without hashing 1.3 GB to find out.
    """
    import hashlib
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return [f"{path} does not exist"]

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return [f"size mismatch: expected {expected_size} bytes, got {actual_size}"]

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        return [f"sha256 mismatch: expected {expected_sha256}, got {actual}"]
    return []


def _resolve_memory_id(value: str, profile: str) -> str:
    """Return a full memory UUID for `value`, which may be a short ID prefix.

    `list` and `search` render only the first 8 characters of an ID, so a
    prefix is what users actually have to hand. Resolving it here keeps the
    backends dealing only in full UUIDs, and an ambiguous prefix is reported
    rather than guessed at -- deleting the wrong memory is unrecoverable.
    """
    import ogham.database as db

    if _UUID_RE.fullmatch(value):
        return value

    if not re.fullmatch(r"[0-9a-fA-F]{4,}", value.replace("-", "")):
        console.print(f"[red]{value!r} is not a memory ID or ID prefix.[/red]")
        raise typer.Exit(1)

    try:
        matches = db.find_memory_ids_by_prefix(value, profile)
    except NotImplementedError:
        console.print("[red]This backend cannot resolve ID prefixes.[/red]")
        console.print("Pass a full memory ID -- get one with: [cyan]ogham list --full-id[/cyan]")
        raise typer.Exit(1) from None

    if not matches:
        console.print(
            f"[red]Memory with ID prefix {value!r} not found in profile {profile!r}.[/red]"
        )
        raise typer.Exit(1)

    if len(matches) > 1:
        console.print(f"[yellow]Prefix {value!r} matches {len(matches)} memories:[/yellow]")
        for match in matches:
            console.print(f"  {match}")
        console.print("[yellow]Use a longer prefix, or the full ID.[/yellow]")
        raise typer.Exit(1)

    return matches[0]


def _safe_text(value: object, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    return text[:limit] if limit is not None else text


def _run_server(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
    recall: bool | None = None,
    inscribe: bool | None = None,
):
    from ogham.flow_control import set_flow_overrides
    from ogham.server import main as server_main

    set_flow_overrides(recall=recall, inscribe=inscribe)
    server_main(transport=transport, host=host, port=port)


@app.callback()
def main_callback(ctx: typer.Context):
    """Ogham MCP — persistent memory for AI clients."""
    if ctx.invoked_subcommand is None:
        _run_server()


@app.command()
def serve(
    transport: Optional[str] = typer.Option(
        None, help="Transport: stdio, streamable-http (recommended for remote), or sse (deprecated)"
    ),
    host: Optional[str] = typer.Option(None, help="Network bind host (default 127.0.0.1)"),
    port: Optional[int] = typer.Option(None, help="Network port (default 8742)"),
    recall: Optional[bool] = typer.Option(
        None,
        "--recall/--no-recall",
        help="Enable or disable recall for this server process",
    ),
    inscribe: Optional[bool] = typer.Option(
        None,
        "--inscribe/--no-inscribe",
        help="Enable or disable inscribe for this server process",
    ),
):
    """Start the MCP server."""
    _run_server(transport=transport, host=host, port=port, recall=recall, inscribe=inscribe)


@app.command()
def store(
    content: str = typer.Argument(help="The text content to remember"),
    profile: str = typer.Option(None, help="Profile to store in"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", help="Tags for the memory"),
    tags_csv: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    source: str = typer.Option("cli", help="Source identifier"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON instead of rich text"),
    inscribe: Optional[bool] = typer.Option(
        None,
        "--inscribe/--no-inscribe",
        help="Enable or disable inscribe for this command",
    ),
):
    """Store a new memory."""
    from ogham.flow_control import disabled_message, inscribe_enabled, temporary_flow_overrides

    with temporary_flow_overrides(inscribe=inscribe):
        if not inscribe_enabled():
            if output_json:
                print(
                    json.dumps(
                        {
                            "status": "disabled",
                            "flow": "inscribe",
                            "message": disabled_message("inscribe"),
                        }
                    )
                )
            else:
                console.print(f"[yellow]{disabled_message('inscribe')}[/yellow]")
            return

        _store_impl(content, profile, tags, tags_csv, source, output_json)


def _store_impl(
    content: str,
    profile: str | None,
    tags: list[str] | None,
    tags_csv: str | None,
    source: str,
    output_json: bool,
) -> None:
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


_ZONE_ICON = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
_ZONE_STYLE = {"GREEN": "green", "AMBER": "yellow", "RED": "red"}


@app.command()
def health(
    profile: Optional[str] = typer.Option(None, help="Profile to score (default: active profile)"),
    output_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """8-dimension health readout (score 0-10 per dim)."""
    from ogham.config import settings
    from ogham.health_dimensions import compose_health, overall_score

    target = profile or settings.default_profile
    results = compose_health(target)
    overall = overall_score(results)
    overall_zone = "GREEN" if overall >= 8.0 else "AMBER" if overall >= 5.0 else "RED"

    if output_json:
        payload = {
            "profile": target,
            "overall_score": overall,
            "overall_zone": overall_zone,
            "dimensions": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, default=str))
        return

    table = Table(title=f"Ogham Health -- profile '{target}'")
    table.add_column("#", justify="right", width=3)
    table.add_column("Name", style="bold", width=18)
    table.add_column("Zone", width=10)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Detail")

    for i, r in enumerate(results, start=1):
        icon = _ZONE_ICON.get(r.zone, "")
        style = _ZONE_STYLE.get(r.zone, "")
        zone_cell = f"{icon} [{style}]{r.zone}[/{style}]" if style else f"{icon} {r.zone}"
        table.add_row(str(i), r.name, zone_cell, f"{r.score:.1f}", r.detail)

    console.print(table)
    overall_style = _ZONE_STYLE.get(overall_zone, "")
    overall_icon = _ZONE_ICON.get(overall_zone, "")
    console.print(
        f"\nOverall: {overall_icon} "
        f"[{overall_style}]{overall_zone}[/{overall_style}] "
        f"(avg {overall:.1f} / 10)"
    )


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
    full_id: bool = typer.Option(False, "--full-id", help="Show full memory UUIDs"),
    extract: bool = typer.Option(False, "--extract", help="Extract query-relevant facts via LLM"),
    recall: Optional[bool] = typer.Option(
        None,
        "--recall/--no-recall",
        help="Enable or disable recall for this command",
    ),
):
    """Search memories by meaning and keywords (hybrid search)."""
    from ogham.flow_control import disabled_message, recall_enabled, temporary_flow_overrides

    with temporary_flow_overrides(recall=recall):
        if not recall_enabled():
            if output_json:
                print("[]")
            else:
                console.print(f"[yellow]{disabled_message('recall')}[/yellow]")
            return

        _search_impl(query, limit, profile, tags, tags_csv, output_json, extract, full_id)


def _search_impl(
    query: str,
    limit: int,
    profile: str | None,
    tags: list[str] | None,
    tags_csv: str | None,
    output_json: bool,
    extract: bool,
    full_id: bool = False,
) -> None:
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
    table.add_column("ID", width=36 if full_id else 8)
    table.add_column("Relevance", justify="right", width=10)
    table.add_column("Content")
    table.add_column("Tags")

    for r in results:
        mem_id = str(r.get("id", "")) if full_id else str(r.get("id", ""))[:8]
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
    full_id: bool = typer.Option(False, "--full-id", help="Show full memory UUIDs"),
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
    table.add_column("ID", width=36 if full_id else 8)
    table.add_column("Created", width=20)
    table.add_column("Content")
    table.add_column("Tags")
    table.add_column("Source")

    for r in results:
        table.add_row(
            _safe_text(r.get("id", ""), None if full_id else 8),
            _safe_text(r.get("created_at", ""), 19),
            _safe_text(r.get("content", ""), 100),
            ", ".join(r.get("tags", [])),
            _safe_text(r.get("source", "")),
        )

    console.print(table)


@app.command()
def delete(
    memory_id: str = typer.Argument(help="Memory ID (full UUID, or a unique ID prefix)"),
    profile: str = typer.Option(None, help="Profile the memory belongs to"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a memory by ID."""
    import ogham.database as db
    from ogham.config import settings

    target = profile or settings.default_profile
    resolved = _resolve_memory_id(memory_id, target)

    if not yes:
        confirm = typer.confirm(f"Delete memory {resolved[:8]}... from '{target}'?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    if db.delete_memory(resolved, target):
        console.print(f"[green]Deleted memory {resolved[:8]}...[/green]")
    else:
        console.print(f"[red]Memory {resolved[:8]}... not found in profile '{target}'.[/red]")


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
    format: str = typer.Option(
        "json", help="Output format: json, markdown, or okf (OKF v0.1 bundle directory)"
    ),
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
    with_graph: bool = typer.Option(
        False,
        "--with-graph",
        help="Also import the bundle's entities/ graph layer (OKF bundles only)",
    ),
    graph_dry_run: bool = typer.Option(
        False,
        "--graph-dry-run",
        help="With --with-graph, report what the graph import would do and write nothing",
    ),
):
    """Import memories from a JSON export file, or an OKF bundle directory.

    The bundle's entity graph is imported only with --with-graph, because
    `entities` is global -- it has no profile column -- so a graph import
    mutates rows every profile reads.

    There is no reliable undo: importing into a populated profile MERGES rather
    than restores, and snapshotting first does not help because the snapshot is
    built from the same export path. Use --graph-dry-run to look before you
    write.
    """
    from ogham.flow_control import disabled_message, inscribe_enabled

    if not inscribe_enabled():
        console.print(f"[yellow]{disabled_message('inscribe')}[/yellow]")
        return

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    from ogham.config import settings
    from ogham.export_import import import_memories

    target = profile or settings.default_profile

    # OKF v0.1 bundles are directories, not files. Hand the path through to
    # the importer as a string -- import_memories auto-detects bundle dirs.
    # JSON/markdown exports are still single files we read into memory.
    import os

    if os.path.isdir(file):
        data = file
    else:
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
            import_graph=with_graph,
            graph_dry_run=graph_dry_run,
        )

    console.print(
        f"[green]Imported {result['imported']} memories, "
        f"skipped {result['skipped']} duplicates.[/green]"
    )

    graph = result.get("graph")
    if graph:
        if not graph.get("graph_present"):
            console.print("[dim]No entities/ layer in this bundle.[/dim]")
        elif graph.get("dry_run"):
            console.print(
                f"[yellow]DRY RUN -- nothing written.[/yellow] Would create "
                f"{graph.get('entities_new', 0)} entities "
                f"({graph.get('entities_existing', 0)} already exist) and "
                f"{graph.get('edges_written', 0)} edges "
                f"({graph.get('edges_already_present', 0)} already present, "
                f"{graph.get('unresolved_edges', 0)} unresolved)."
            )
        else:
            console.print(
                f"[green]Graph: {graph.get('entities_new', 0)} entities created, "
                f"{graph.get('edges_written', 0)} edges written "
                f"({graph.get('edges_already_present', 0)} already present, "
                f"{graph.get('unresolved_edges', 0)} unresolved).[/green]"
            )


@app.command(name="import-claude-code")
def import_claude_code_cmd(
    directory: str = typer.Argument(
        help="Path to a Claude Code memory dir (e.g. ~/.claude/projects/<encoded-cwd>/memory/)"
    ),
    profile: str = typer.Option(None, help="Profile to import into"),
    dedup: float = typer.Option(0.8, help="Dedup threshold (0 to disable)"),
    project: str = typer.Option(
        None,
        help=(
            "Override the inferred project tag. The encoded-cwd heuristic is "
            "lossy on hyphenated repo names (e.g. 'openbrain-sharedmemory' -> "
            "'sharedmemory'); pass --project ogham to keep tags consistent."
        ),
    ),
):
    """Import memories from a Claude Code local-memory directory.

    Parses ``MEMORY.md``-companion ``.md`` files (each with YAML frontmatter
    carrying name/description/type/originSessionId) and imports the bodies
    as Ogham memories tagged ``source:claude-code-memory``, ``type:<frontmatter
    type>``, and ``project:<inferred from directory>``.
    """
    from ogham.flow_control import disabled_message, inscribe_enabled

    if not inscribe_enabled():
        console.print(f"[yellow]{disabled_message('inscribe')}[/yellow]")
        return

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    from ogham.claude_code_import import import_claude_code_memories
    from ogham.config import settings

    target = profile or settings.default_profile

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

        result = import_claude_code_memories(
            directory,
            profile=target,
            dedup_threshold=dedup,
            on_progress=on_progress,
            on_embed_progress=on_embed_progress,
            project_tag=project,
        )

    if result.get("warning"):
        console.print(f"[yellow]Warning: {result['warning']} ({result['directory']})[/yellow]")
        return
    console.print(
        f"[green]Imported {result['imported']} memories, "
        f"skipped {result['skipped']} duplicates from {result['directory']}.[/green]"
    )


@app.command(name="import-claude-ai")
def import_claude_ai_cmd(
    path: str = typer.Argument(
        help=(
            "Path to a Claude.ai data export — accepts the .zip Anthropic emails "
            "you, the unzipped directory, or conversations.json directly."
        )
    ),
    profile: str = typer.Option(None, help="Profile to import into"),
    dedup: float = typer.Option(0.8, help="Dedup threshold (0 to disable)"),
    mode: str = typer.Option(
        "turn-pairs",
        help=(
            "Granularity: 'turn-pairs' (default, one memory per human/assistant "
            "exchange), 'raw' (one per message), 'summarize' (placeholder, "
            "currently behaves like turn-pairs)."
        ),
    ),
    project: str = typer.Option(
        None,
        help="Override the project tag attached to every imported memory.",
    ),
    since: str = typer.Option(
        None,
        help=("Only import conversations updated on/after this date (ISO 8601, e.g. 2026-01-01)."),
    ),
    no_smart_filter: bool = typer.Option(
        False,
        "--no-smart-filter",
        help="Keep pleasantry turn-pairs that the default filter drops.",
    ),
):
    """Import memories from a Claude.ai data export.

    Anthropic offers a first-party export at Settings -> Privacy -> Request
    your data. After ~24-48h you receive a ZIP with conversations.json plus
    metadata. This command parses that export, walks each conversation as
    consecutive (human, assistant) turn-pairs, and stores one memory per
    pair (assistant turn as content, human prompt in metadata.user_prompt).

    Privacy note: a year of Claude.ai history can include sensitive content.
    Pre-prune the export ZIP before running if needed; --since narrows by
    date.
    """
    from ogham.flow_control import disabled_message, inscribe_enabled

    if not inscribe_enabled():
        console.print(f"[yellow]{disabled_message('inscribe')}[/yellow]")
        return

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    from ogham.claude_ai_import import import_claude_ai_export
    from ogham.config import settings

    target = profile or settings.default_profile

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

        result = import_claude_ai_export(
            path,
            profile=target,
            mode=mode,  # type: ignore[arg-type]
            smart_filter=not no_smart_filter,
            project_tag=project,
            since=since,
            dedup_threshold=dedup,
            on_progress=on_progress,
            on_embed_progress=on_embed_progress,
        )

    if result.get("warning"):
        console.print(f"[yellow]Warning: {result['warning']} ({result['path']})[/yellow]")
        return
    console.print(
        f"[green]Imported {result['imported']} memories, "
        f"skipped {result['skipped']} duplicates from {result['path']} "
        f"(mode={result['mode']}).[/green]"
    )


@app.command(name="import-linear")
def import_linear_cmd(
    team: str = typer.Option(..., "--team", help="Linear team key, e.g. TBU"),
    since: int = typer.Option(30, "--since", help="Days lookback for updatedAt filter"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
):
    """Import Linear issues into Ogham as memories, deduped by tracker_external_id."""
    import os

    from ogham.importers.linear import LinearClient
    from ogham.tools.import_linear import _DefaultMemoryService, import_linear_impl
    from ogham.tools.memory import get_active_profile

    token = os.environ.get("LINEAR_API_TOKEN")
    if not token:
        console.print("[red]LINEAR_API_TOKEN not set[/red]")
        raise typer.Exit(code=1)

    target = profile or get_active_profile()
    client = LinearClient(token=token)
    service = _DefaultMemoryService()
    result = import_linear_impl(
        client=client,
        service=service,
        team_key=team,
        since_days=since,
        profile=target,
    )
    console.print(
        f"[green]imported={result['imported']} skipped={result['skipped']} "
        f"disabled={result['disabled']}[/green]"
    )


@app.command(name="ingest-obsidian")
def ingest_obsidian_cmd(
    vault: str = typer.Argument(..., help="Path to the Obsidian vault root"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
    source: str = typer.Option("obsidian", "--source", help="Source label stored on each memory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would store without storing"
    ),
):
    """Ingest an Obsidian vault into Ogham. Idempotent -- safe to run on a timer."""
    from ogham.tools.import_obsidian import DefaultIngestService, ingest_obsidian_impl
    from ogham.tools.memory import get_active_profile

    target = profile or get_active_profile()
    try:
        result = ingest_obsidian_impl(
            vault_path=vault,
            service=DefaultIngestService(),
            profile=target,
            source=source,
            dry_run=dry_run,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    console.print(
        f"[green]scanned={result['scanned']} stored={result['stored']} "
        f"dup={result['skipped_duplicate']} ignored={result['skipped_ignored']} "
        f"disabled={result['disabled']} errors={result['errors']}[/green]"
    )


@app.command(name="ingest-telegram")
def ingest_telegram_cmd(
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
    source: str = typer.Option("telegram", "--source", help="Source label stored on each memory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would store without storing"
    ),
):
    """Ingest Telegram messages into Ogham (outbound getUpdates). Idempotent -- safe on a timer."""
    import os

    from ogham.tools.import_telegram import _allowed_chat_ids_from_env, ingest_telegram_impl
    from ogham.tools.memory import get_active_profile

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        console.print("[red]TELEGRAM_BOT_TOKEN not set[/red]")
        raise typer.Exit(code=1)

    target = profile or get_active_profile()
    try:
        allowed = _allowed_chat_ids_from_env()
        result = ingest_telegram_impl(
            client=TelegramClient(token=token),
            service=DefaultIngestService(),
            profile=target,
            source=source,
            allowed_chat_ids=allowed,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]scanned={result['scanned']} stored={result['stored']} "
        f"dup={result['skipped_duplicate']} ignored={result['skipped_ignored']} "
        f"disabled={result['disabled']} errors={result['errors']}[/green]"
    )


@app.command(name="ingest-slack")
def ingest_slack_cmd(
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
    source: str = typer.Option("slack", "--source", help="Source label stored on each memory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would store without storing"
    ),
):
    """Ingest Slack channel messages into Ogham (poll conversations.history). Idempotent."""
    import os

    from ogham.tools.import_slack import _channels_from_env, ingest_slack_impl
    from ogham.tools.memory import get_active_profile

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        console.print("[red]SLACK_BOT_TOKEN not set[/red]")
        raise typer.Exit(code=1)

    target = profile or get_active_profile()
    try:
        channels = _channels_from_env()
        result = ingest_slack_impl(
            client=SlackClient(token=token),
            service=DefaultIngestService(),
            profile=target,
            channels=channels,
            source=source,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]scanned={result['scanned']} stored={result['stored']} "
        f"dup={result['skipped_duplicate']} ignored={result['skipped_ignored']} "
        f"disabled={result['disabled']} errors={result['errors']}[/green]"
    )


@app.command(name="import-github")
def import_github_cmd(
    since_days: int = typer.Option(
        30, "--since-days", help="Import issues updated in the last N days"
    ),
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
    source: str = typer.Option("github", "--source", help="Source label stored on each memory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would store without storing"
    ),
):
    """Import GitHub issues + comments into Ogham (REST). Idempotent -- safe on a timer."""
    import os

    from ogham.tools.import_github import _repos_from_env, import_github_impl
    from ogham.tools.memory import get_active_profile

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print("[red]GITHUB_TOKEN not set[/red]")
        raise typer.Exit(code=1)

    target = profile or get_active_profile()
    try:
        repos = _repos_from_env()
        result = import_github_impl(
            client=GitHubClient(token=token),
            service=DefaultIngestService(),
            profile=target,
            repos=repos,
            since_days=since_days,
            source=source,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]scanned={result['scanned']} stored={result['stored']} "
        f"dup={result['skipped_duplicate']} ignored={result['skipped_ignored']} "
        f"disabled={result['disabled']} errors={result['errors']}[/green]"
    )


@app.command(name="import-beads")
def import_beads_cmd(
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile to store into"),
    source: str = typer.Option("beads", "--source", help="Source label stored on each memory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would store without storing"
    ),
):
    """Import Beads issues + comments into Ogham (via the `bd` CLI). Idempotent."""
    from ogham.tools.import_beads import _beads_dir_from_env, import_beads_impl
    from ogham.tools.memory import get_active_profile

    target = profile or get_active_profile()
    try:
        beads_dir = _beads_dir_from_env()
        result = import_beads_impl(
            client=BeadsClient(beads_dir=beads_dir),
            service=DefaultIngestService(),
            profile=target,
            source=source,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]scanned={result['scanned']} stored={result['stored']} "
        f"dup={result['skipped_duplicate']} ignored={result['skipped_ignored']} "
        f"disabled={result['disabled']} errors={result['errors']}[/green]"
    )


@app.command(name="predicates")
def predicates_cmd():
    """List the typed-edge predicate vocabulary with their portable URIs."""
    from ogham.entity_graph import PREDICATE_URIS
    from ogham.tools.entity_graph import describe_predicates_impl

    for row in describe_predicates_impl(uris=PREDICATE_URIS):
        schema_org = row["schema_org_uri"] or "-"
        iirds = row["iirds_uri"] or "-"
        console.print(
            f"{row['predicate']:16} {row['ogham_uri']}  schema.org={schema_org}  iirds={iirds}"
        )


@app.command(name="trace-provenance")
def trace_provenance_cmd(
    edge_id: int = typer.Argument(..., help="Edge id to trace"),
    max_depth: int = typer.Option(10, help="Maximum BFS depth"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile"),
):
    """Walk an edge's derivation lineage back to root evidence (memories)."""
    from ogham.database import get_entity_graph_and_vocab
    from ogham.tools.entity_graph import trace_provenance_impl
    from ogham.tools.memory import get_active_profile

    target = profile or get_active_profile()
    graph, _allowed = get_entity_graph_and_vocab()
    result = trace_provenance_impl(
        graph=graph, edge_id=edge_id, profile=target, max_depth=max_depth
    )
    print(json.dumps(result, default=str))


@app.command(name="find-derivatives")
def find_derivatives_cmd(
    source_id: str = typer.Argument(..., help="Edge id or memory uuid to check impact for"),
    max_depth: int = typer.Option(10, help="Maximum BFS depth"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Ogham profile"),
):
    """Find every edge that (transitively) cites source_id -- impact analysis."""
    from ogham.database import get_entity_graph_and_vocab
    from ogham.tools.entity_graph import find_derivatives_impl
    from ogham.tools.memory import get_active_profile

    target = profile or get_active_profile()
    graph, _allowed = get_entity_graph_and_vocab()
    sid: int | str = int(source_id) if source_id.isdigit() else source_id
    result = find_derivatives_impl(graph=graph, source_id=sid, profile=target, max_depth=max_depth)
    print(json.dumps(result, default=str))


@app.command(name="backfill-entities")
def backfill_entities_cmd(
    profile: str = typer.Option(None, help="Profile to backfill (default: all)"),
    batch_size: int = typer.Option(200, help="Memory rows per batch"),
):
    """Populate entities + memory_entities for existing memory rows.

    One-shot per deployment after applying migration 036. New writes after
    v0.14 are linked automatically by the live store_memory path; this
    command covers anything written before that.
    """
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from ogham.entity_backfill import backfill_entities

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[edges]} edges"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Linking entities...", total=None, edges=0)

        def on_progress(processed: int, edges: int, total: int) -> None:
            if progress.tasks[task].total != total:
                progress.update(task, total=total)
            progress.update(task, completed=processed, edges=edges)

        result = backfill_entities(
            profile=profile,
            batch_size=batch_size,
            on_progress=on_progress,
        )

    scope = f" in profile {result['profile']!r}" if result.get("profile") else ""
    console.print(
        f"[green]Backfill complete: {result['edges_added']} edges added across "
        f"{result['memories_with_entities']}/{result['total']} memories{scope}.[/green]"
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


@app.command(name="download-model")
def download_model(
    model: str = typer.Argument("bge-m3", help="Model to download (only bge-m3 supported)"),
    path: str = typer.Option(
        None, "--path", help="Download directory (default: ~/.cache/ogham/bge-m3-onnx)"
    ),
):
    """Download ONNX model files for local embedding."""
    import os
    import shutil
    import tempfile
    import urllib.request
    import zipfile
    from pathlib import Path

    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

    models = MODEL_REGISTRY

    if model not in models:
        console.print(f"[red]Unknown model {model!r}. Available: {', '.join(models)}[/red]")
        raise typer.Exit(1)

    info = models[model]
    default_dir = Path.home() / ".cache" / "ogham" / "bge-m3-onnx"
    dest = Path(path) if path else default_dir

    if all((dest / f).exists() for f in info["expected_files"]):
        console.print(f"[green]Model {model!r} already exists at {dest}[/green]")
        raise typer.Exit(0)

    console.print(f"Downloading [bold]{model}[/bold] to {dest}...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0 and progress.tasks[task].total is None:
                progress.update(task, total=total_size)
            downloaded = block_num * block_size
            if total_size > 0:
                downloaded = min(downloaded, total_size)
            progress.update(task, completed=downloaded)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "model.zip"
            # S310: the URL is a fixed https GitHub release asset defined above,
            # never user input, so no scheme-injection surface exists.
            urllib.request.urlretrieve(info["url"], zip_path, reporthook=_reporthook)  # noqa: S310

            # Integrity gate: refuse to extract an archive we cannot vouch for.
            progress.update(task, description="Verifying...")
            problems = verify_archive(zip_path, info["sha256"], info["size"])
            if problems:
                console.print(f"[red]Downloaded archive failed verification: {problems[0]}[/red]")
                console.print(
                    "The download was discarded and nothing was extracted. This usually means "
                    "a truncated or interrupted transfer -- try again. If it repeats, the "
                    "upstream release asset may have changed and the pinned digest in "
                    "MODEL_REGISTRY needs review before trusting it."
                )
                raise typer.Exit(1)

            progress.update(task, description="Extracting...")
            with zipfile.ZipFile(zip_path) as zf:
                # Refuse archive members that would escape the extraction root.
                for member in zf.namelist():
                    if os.path.isabs(member) or ".." in member.split("/"):
                        console.print(f"[red]Unsafe zip member: {member}[/red]")
                        raise typer.Exit(1)
                zf.extractall(tmpdir)

            # Members may sit at the archive root or under a subdirectory, so
            # locate each expected file rather than assuming a layout.
            dest.mkdir(parents=True, exist_ok=True)
            try:
                for expected in info["expected_files"]:
                    candidates = list(Path(tmpdir).rglob(expected))
                    if not candidates:
                        console.print(f"[red]Expected file {expected!r} not found in archive[/red]")
                        raise typer.Exit(1)
                    shutil.copy2(candidates[0], dest / expected)
            except Exception:
                # Never leave a half-installed model behind.
                for f in info["expected_files"]:
                    (dest / f).unlink(missing_ok=True)
                raise

    console.print(f"[green]Model {model!r} downloaded to {dest}[/green]")


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
    active_profile = profile or settings.default_profile

    dashboard_app = create_app(profile=active_profile)
    console.print(f"[green]Ogham dashboard ({active_profile}) → http://{host}:{port}[/green]")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="warning")


def main():
    try:
        app()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
