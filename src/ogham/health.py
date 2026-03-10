import logging

from ogham.config import settings
from ogham.database import get_backend

logger = logging.getLogger(__name__)


def check_database() -> dict[str, str | bool]:
    """Check database connectivity with a lightweight query."""
    try:
        backend = get_backend()
        if hasattr(backend, "_get_client"):
            # Supabase backend
            client = backend._get_client()
            client.table("memories").select("id", count="exact").limit(1).execute()
        elif hasattr(backend, "_execute"):
            # Postgres backend
            backend._execute("SELECT 1 FROM memories LIMIT 1", fetch="one")
        else:
            return {"status": "error", "connected": False, "error": "Unknown backend type"}
        return {"status": "ok", "connected": True, "backend": settings.database_backend}
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        return {"status": "error", "connected": False, "error": str(e)}


def check_embedding_provider() -> dict[str, str | bool]:
    """Check embedding provider availability."""
    provider = settings.embedding_provider

    if provider == "ollama":
        try:
            import ollama

            client = ollama.Client(host=settings.ollama_url)
            client.list()
            return {"status": "ok", "provider": "ollama", "url": settings.ollama_url}
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return {
                "status": "error",
                "provider": "ollama",
                "url": settings.ollama_url,
                "error": str(e),
                "hint": "Is Ollama running? Try: ollama serve",
            }

    elif provider == "openai":
        if not settings.openai_api_key:
            return {
                "status": "error",
                "provider": "openai",
                "error": "OPENAI_API_KEY not set",
            }
        return {"status": "ok", "provider": "openai"}

    elif provider == "mistral":
        try:
            import mistralai  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "provider": "mistral",
                "error": "mistralai package not installed",
                "hint": "Install with: uv add ogham-mcp[mistral]",
            }
        if not settings.mistral_api_key:
            return {
                "status": "error",
                "provider": "mistral",
                "error": "MISTRAL_API_KEY not set",
            }
        return {"status": "ok", "provider": "mistral"}

    elif provider == "voyage":
        try:
            import voyageai  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "provider": "voyage",
                "error": "voyageai package not installed",
                "hint": "Install with: uv add ogham-mcp[voyage]",
            }
        if not settings.voyage_api_key:
            return {
                "status": "error",
                "provider": "voyage",
                "error": "VOYAGE_API_KEY not set",
            }
        valid_voyage_dims = {256, 512, 1024, 2048}
        if settings.embedding_dim not in valid_voyage_dims:
            return {
                "status": "error",
                "provider": "voyage",
                "error": f"EMBEDDING_DIM={settings.embedding_dim} not supported by Voyage",
                "hint": f"Voyage supports: {sorted(valid_voyage_dims)}",
            }
        return {"status": "ok", "provider": "voyage"}

    return {"status": "unknown", "provider": provider}


def check_config() -> dict[str, str | bool | list[str]]:
    """Validate configuration values."""
    issues = []

    if settings.database_backend == "supabase":
        if not settings.supabase_url:
            issues.append("SUPABASE_URL not set")
        if not settings.supabase_key:
            issues.append("SUPABASE_KEY not set")
    elif settings.database_backend == "postgres":
        if not settings.database_url:
            issues.append("DATABASE_URL not set")

    if settings.embedding_dim not in (256, 512, 768, 1024, 1536, 2048):
        issues.append(f"Unusual embedding_dim: {settings.embedding_dim}")

    return {
        "status": "ok" if not issues else "warning",
        "issues": issues,
    }


# Keep old name as alias for backwards compatibility
check_supabase = check_database


def full_health_check() -> dict[str, dict]:
    """Run all health checks."""
    return {
        "database": check_database(),
        "embedding": check_embedding_provider(),
        "config": check_config(),
    }
