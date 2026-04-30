import asyncio
import logging
import os
import sys
import threading

from ogham.app import mcp
from ogham.config import settings
from ogham.health import check_embedding_provider, check_supabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

import ogham.prompts  # noqa: F401, E402
import ogham.tools.dashboard  # noqa: F401, E402
import ogham.tools.memory  # noqa: F401, E402
import ogham.tools.stats  # noqa: F401, E402
import ogham.tools.wiki  # noqa: F401, E402


def validate_startup() -> None:
    """Run lightweight health checks before starting MCP server."""
    logger = logging.getLogger(__name__)

    db = check_supabase()
    if not db.get("connected"):
        logger.error("Database connection failed: %s", db.get("error"))
        logger.error("Check your database configuration in .env")
        sys.exit(1)
    logger.info("Database: connected (%s)", db.get("backend", "supabase"))

    embedding = check_embedding_provider()
    if embedding.get("status") != "ok":
        logger.error("Embedding provider check failed: %s", embedding.get("error"))
        if "hint" in embedding:
            logger.error(embedding["hint"])
        sys.exit(1)
    logger.info("Embedding provider: %s", embedding.get("provider"))


def _warm_hybrid_search() -> None:
    """Run one discardable hybrid_search at boot to warm the connection.

    Supabase HTTPS keep-alive + HNSW page cache need a first roundtrip
    before steady-state latency kicks in. Without this, the user's
    first query lands cold (often 1-2s on free-tier Supabase that
    auto-pauses compute). Running a throwaway query at server start
    moves the cold-start cost off the critical path.

    Failures here are non-fatal -- if the warm-up errors (no embedding
    provider, bad credentials), the server still boots and the next
    real query gets the cold-start penalty. validate_startup() already
    caught the hard configuration errors above.
    """
    try:
        from ogham.database import hybrid_search_memories
        from ogham.embeddings import generate_embedding

        emb = generate_embedding("ogham boot warmup")
        hybrid_search_memories(
            query_text="ogham boot warmup",
            query_embedding=emb,
            profile=settings.default_profile,
            limit=1,
        )
        logger = logging.getLogger(__name__)
        logger.info("Warmed hybrid_search connection")
    except Exception as exc:
        # Don't block server start on a slow/cold backend. The first
        # real user query will pay the cold cost; that's no worse than
        # without the warmup.
        logging.getLogger(__name__).debug(
            "Boot warmup skipped (%s); next call will warm the connection",
            type(exc).__name__,
        )


def main(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    validate_startup()

    actual_transport = transport or settings.server_transport
    actual_host = host or settings.server_host
    actual_port = port or settings.server_port

    logger = logging.getLogger(__name__)

    # v0.14: warm the hybrid_search path before the server starts taking
    # client traffic. See _warm_hybrid_search docstring. Disable with
    # OGHAM_BOOT_WARMUP=false (e.g. in tests, or environments with
    # known-fast steady-state).
    if os.environ.get("OGHAM_BOOT_WARMUP", "true").lower() not in {"false", "0", "no"}:
        _warm_hybrid_search()

    if actual_transport == "sse":
        logger.info("Starting SSE server on %s:%s", actual_host, actual_port)
        mcp.run(transport="sse", host=actual_host, port=actual_port)
    else:
        if settings.enable_http_health:
            from ogham.http_health import start_health_server

            logger.info("Starting health check endpoint on port %s", settings.health_port)

            async def _serve_health():
                server = await start_health_server(settings.health_port)
                await server.serve_forever()

            def run_health_server():
                asyncio.run(_serve_health())

            health_thread = threading.Thread(target=run_health_server, daemon=True)
            health_thread.start()

        try:
            mcp.run()
        except KeyboardInterrupt:
            import sys

            sys.exit(0)


if __name__ == "__main__":
    main()
