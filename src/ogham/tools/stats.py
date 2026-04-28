import re
from pathlib import Path
from typing import Any

from ogham.app import mcp
from ogham.database import get_memory_stats
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

    from ogham.flow_control import flow_status

    config["memory_flows"] = flow_status()

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
