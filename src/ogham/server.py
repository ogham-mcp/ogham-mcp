import asyncio
import logging
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
