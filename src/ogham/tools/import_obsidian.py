"""Obsidian ingest MCP tool + CLI backend -- walk a vault, store new notes.

Thin adapter over the shared ``ogham.ingest`` contract: it supplies the raw
items (vault note paths) and a ``to_record`` mapper; ``run_ingest`` owns the
loop, dedup, dry-run, and per-item resilience.
"""

from __future__ import annotations

from pathlib import Path

from ogham.app import mcp
from ogham.importers.obsidian import iter_vault_notes, parse_note
from ogham.ingest import DefaultIngestService, IngestRecord, IngestService, run_ingest
from ogham.tools.memory import get_active_profile


def _path_to_record(path: Path, root: Path) -> IngestRecord:
    """Read + parse one vault note into an ``IngestRecord``. May raise on an
    unreadable / non-UTF8 file -- ``run_ingest`` catches it as an error."""
    raw = path.read_text(encoding="utf-8")
    note = parse_note(str(path.relative_to(root)), raw)
    return IngestRecord(content=note.content, tags=note.tags, metadata=note.metadata)


def ingest_obsidian_impl(
    *,
    vault_path: str,
    service: IngestService,
    profile: str,
    source: str = "obsidian",
    dry_run: bool = False,
) -> dict:
    """Scan a vault and store new/changed notes. Idempotent; safe to re-run.

    Dedup by ``content_fingerprint``. A changed note gets a new memory; no
    supersession. One bad file is counted under ``errors`` and never aborts.

    Raises:
        ValueError: if ``vault_path`` is not a directory.
    """
    if not Path(vault_path).is_dir():
        raise ValueError(f"vault path is not a directory: {vault_path}")

    root = Path(vault_path)
    summary = run_ingest(
        items=iter_vault_notes(vault_path),
        to_record=lambda path: _path_to_record(path, root),
        service=service,
        profile=profile,
        source=source,
        dedup_key_field="content_fingerprint",
        dry_run=dry_run,
    )
    # run_ingest's own contract includes an internal "stopped" key (used by
    # adapters that pass stop_on_store_error=True); Obsidian doesn't use that
    # mode, so only surface the 6 public counts this adapter documents.
    public_keys = (
        "scanned",
        "stored",
        "skipped_duplicate",
        "skipped_ignored",
        "disabled",
        "errors",
    )
    return {key: summary[key] for key in public_keys}


@mcp.tool
def ingest_obsidian(
    vault_path: str,
    profile: str | None = None,
    source: str = "obsidian",
    dry_run: bool = False,
) -> dict:
    """Ingest an Obsidian vault into Ogham as memories, deduped by content fingerprint.

    Args:
        vault_path: Absolute path to the vault root on this machine.
        profile: Target profile. Defaults to the active profile.
        source: Source label stored on each memory (default ``"obsidian"``).
        dry_run: When true, report counts without storing.

    Returns:
        ``{"scanned","stored","skipped_duplicate","skipped_ignored","disabled","errors"}``.
    """
    target = profile or get_active_profile()
    return ingest_obsidian_impl(
        vault_path=vault_path,
        service=DefaultIngestService(),
        profile=target,
        source=source,
        dry_run=dry_run,
    )
