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
import ogham.tools.memory  # noqa: F401, E402
import ogham.tools.stats  # noqa: F401, E402


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


def main():
    validate_startup()

    if settings.enable_http_health:
        from ogham.http_health import start_health_server

        logger = logging.getLogger(__name__)
        logger.info("Starting health check endpoint on port %s", settings.health_port)

        def run_health_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_health_server(settings.health_port))
            loop.run_forever()

        health_thread = threading.Thread(target=run_health_server, daemon=True)
        health_thread.start()

    try:
        mcp.run()
    except KeyboardInterrupt:
        os._exit(0)


if __name__ == "__main__":
    main()
