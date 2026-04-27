import logging
from typing import Any, cast

from ogham.config import settings
from ogham.database import get_backend

logger = logging.getLogger(__name__)


def check_database() -> dict[str, str | bool]:
    """Check database connectivity with a lightweight query."""
    try:
        backend = cast(Any, get_backend())
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
    except ModuleNotFoundError as e:
        mod = e.name or str(e)
        if "psycopg" in mod:
            hint = "Install with: uvx --from 'ogham-mcp[postgres]' ogham-mcp health"
        else:
            hint = f"Missing module: {mod}"
        logger.error("Database health check failed: %s", e)
        return {"status": "error", "connected": False, "error": str(e), "hint": hint}
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

    elif provider == "gemini":
        try:
            from google import genai  # pyright: ignore[reportAttributeAccessIssue]  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "provider": "gemini",
                "error": "google-genai package not installed",
                "hint": 'Install with: pip install "ogham-mcp[gemini]"',
            }
        if not settings.gemini_api_key:
            return {
                "status": "error",
                "provider": "gemini",
                "error": "GEMINI_API_KEY not set",
            }
        return {"status": "ok", "provider": "gemini"}

    elif provider == "onnx":
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "provider": "onnx",
                "error": "onnxruntime package not installed",
                "hint": "Install with: uv add onnxruntime",
            }
        from pathlib import Path

        model_path = settings.onnx_model_path
        if not model_path:
            # Use same default as onnx_embedder._get_model()
            model_path = str(Path.home() / ".cache" / "ogham" / "bge-m3-onnx" / "bge_m3_model.onnx")

        if not Path(model_path).exists():
            return {
                "status": "error",
                "provider": "onnx",
                "error": f"Model not found: {model_path}",
            }
        return {"status": "ok", "provider": "onnx", "model_path": model_path}

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
