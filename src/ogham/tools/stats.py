from typing import Any

from ogham.app import mcp
from ogham.database import get_memory_stats
from ogham.embeddings import get_cache_stats as _get_cache_stats
from ogham.tools.memory import get_active_profile


@mcp.tool
def get_stats() -> dict[str, Any]:
    """Get summary statistics for the active memory profile.

    Returns total count, source breakdown, and top tags.
    """
    return get_memory_stats(profile=get_active_profile())


@mcp.tool
def get_cache_stats() -> dict[str, int | float]:
    """Get embedding cache statistics.

    Returns current size, max size, hit/miss counts, eviction count, and hit rate.
    Useful for monitoring cache effectiveness.
    """
    return _get_cache_stats()
