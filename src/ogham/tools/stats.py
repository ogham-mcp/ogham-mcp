import re
from pathlib import Path
from typing import Any

from ogham.app import mcp
from ogham.database import (
    count_decay_eligible,
    get_all_memories_full,
    get_memory_stats,
    get_related_memories,
)
from ogham.embeddings import get_cache_stats as _get_cache_stats
from ogham.tools.memory import get_active_profile


def _mask_secret(value: str | None) -> str | None:
    """Mask sensitive values, preserving host/scheme for database URLs."""
    if not value:
        return None
    # Database URLs: show scheme + host, mask credentials
    m = re.match(r"(postgres(?:ql)?://)([^@]+)@(.+)", value)
    if m:
        return f"{m.group(1)}***@{m.group(3)}"
    # API keys: show first 8 and last 4 chars
    if len(value) > 16:
        return f"{value[:8]}...{value[-4:]}"
    return "***"


def get_runtime_config() -> dict[str, Any]:
    """Build the runtime config dict with masked secrets."""
    from ogham.config import settings

    # Find where config was loaded from
    config_sources = []
    env_file = Path(".env")
    if env_file.exists():
        config_sources.append(str(env_file.resolve()))
    global_env = Path.home() / ".ogham" / "config.env"
    if global_env.exists():
        config_sources.append(str(global_env))
    if not config_sources:
        config_sources.append("environment variables only (no .env file found)")

    config = {
        "config_sources": config_sources,
        "database": {
            "backend": settings.database_backend,
        },
        "embeddings": {
            "provider": settings.embedding_provider,
            "dimensions": settings.embedding_dim,
            "batch_size": settings.embedding_batch_size,
        },
        "profile": {
            "default": settings.default_profile,
        },
        "search": {
            "match_threshold": settings.default_match_threshold,
            "match_count": settings.default_match_count,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_alpha": settings.rerank_alpha if settings.rerank_enabled else None,
        },
        "server": {
            "transport": settings.server_transport,
            "host": settings.server_host,
            "port": settings.server_port,
        },
    }

    # Database-specific fields
    if settings.database_backend == "supabase":
        config["database"]["supabase_url"] = settings.supabase_url
        config["database"]["supabase_key"] = _mask_secret(settings.supabase_key)
    elif settings.database_backend == "postgres":
        config["database"]["database_url"] = _mask_secret(settings.database_url)
    elif settings.database_backend == "gateway":
        config["database"]["gateway_url"] = settings.gateway_url
        config["database"]["gateway_api_key"] = _mask_secret(settings.gateway_api_key)

    # Provider-specific fields
    provider = settings.embedding_provider
    if provider == "ollama":
        config["embeddings"]["ollama_url"] = settings.ollama_url
        config["embeddings"]["model"] = settings.ollama_embed_model
    elif provider == "openai":
        config["embeddings"]["api_key"] = _mask_secret(settings.openai_api_key)
    elif provider == "mistral":
        config["embeddings"]["model"] = settings.mistral_embed_model
        config["embeddings"]["api_key"] = _mask_secret(settings.mistral_api_key)
    elif provider == "voyage":
        config["embeddings"]["model"] = settings.voyage_embed_model
        config["embeddings"]["api_key"] = _mask_secret(settings.voyage_api_key)
    elif provider == "gemini":
        config["embeddings"]["model"] = settings.gemini_embed_model
        config["embeddings"]["api_key"] = _mask_secret(settings.gemini_api_key)

    return config


@mcp.tool
def get_config() -> dict[str, Any]:
    """Get the current runtime configuration (read-only).

    Returns database backend, embedding provider, dimensions, model, search
    settings, and server transport. Sensitive values (API keys, database
    passwords) are masked. Useful for AI assistants that need to discover
    how Ogham is configured without searching the filesystem.
    """
    return get_runtime_config()


def _build_tag_distribution(stats: dict[str, Any]) -> list[dict[str, Any]]:
    total = stats.get("total")
    if not isinstance(total, int) or total <= 0:
        return []

    distribution: list[dict[str, Any]] = []
    for item in stats.get("top_tags") or []:
        tag = item.get("tag")
        count = item.get("count")
        if not tag or not isinstance(count, int):
            continue
        distribution.append({"tag": tag, "count": count, "share": count / total})
    return distribution


def _count_orphan_memories(profile: str) -> int | None:
    try:
        memories = get_all_memories_full(profile)
        orphan_count = 0
        for memory in memories:
            memory_id = memory.get("id")
            if not memory_id:
                continue
            if not get_related_memories(memory_id=memory_id, limit=1):
                orphan_count += 1
        return orphan_count
    except Exception:
        return None


def enrich_stats(profile: str, stats: dict[str, Any]) -> dict[str, Any]:
    stats = dict(stats)
    stats["decay_eligible"] = count_decay_eligible(profile)
    stats["orphan_count"] = _count_orphan_memories(profile)
    stats["tag_distribution"] = _build_tag_distribution(stats)
    return stats


@mcp.tool
def get_stats() -> dict[str, Any]:
    """Get summary statistics for the active memory profile.

    Returns total count, source breakdown, top tags, tag distribution,
    orphan count, and Hebbian decay eligibility.
    """
    profile = get_active_profile()
    return enrich_stats(profile, get_memory_stats(profile=profile))


@mcp.tool
def get_cache_stats() -> dict[str, int | float]:
    """Get embedding cache statistics.

    Returns current size, max size, hit/miss counts, eviction count, and hit rate.
    Useful for monitoring cache effectiveness.
    """
    return _get_cache_stats()
