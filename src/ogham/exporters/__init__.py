"""Filesystem-shaped views of Ogham memory.

Each exporter takes the live database and writes a self-contained,
human-readable artifact: an Obsidian vault, a static-site bundle, etc.
Read-only operations -- exporters never mutate memory state.
"""

from ogham.exporters.obsidian import ExportResult, export_to_vault

__all__ = ["ExportResult", "export_to_vault"]
