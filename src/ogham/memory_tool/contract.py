"""Factory for the Anthropic Memory Tool 6-op filesystem contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = "~/.ogham/memory-tool"


def make_memory_contract(root: str | Path | None = None) -> Any:
    """Return a filesystem-backed Memory Tool implementation.

    Uses the reference ``FilesystemMemory`` from ``memory-tool-conformance``
    (dev/CI dependency). Paths are virtual ``/memories/...`` paths per the
    Anthropic context-editing spec.
    """
    try:
        from memory_tool_conformance.reference.fs_memory import FilesystemMemory
    except ImportError as exc:
        raise ImportError(
            "memory-tool-conformance is required for the Memory Tool contract. "
            "Install dev deps: uv sync --extra dev --extra postgres --group dev"
        ) from exc

    if root is None:
        root = os.environ.get("OGHAM_MEMORY_TOOL_ROOT", _DEFAULT_ROOT)
    return FilesystemMemory(str(Path(root).expanduser()))
