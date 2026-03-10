"""Lightweight HTTP health endpoint for container orchestration."""

import asyncio
import json
import logging

from ogham.health import full_health_check

logger = logging.getLogger(__name__)


async def handle_health_request() -> tuple[int, str]:
    """Run health checks and return (status_code, json_body)."""
    result = full_health_check()
    has_error = any(v.get("status") == "error" for v in result.values())
    status = 503 if has_error else 200
    body = json.dumps(result, default=str)
    return status, body


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single HTTP request."""
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        request_str = request_line.decode("utf-8", errors="replace").strip()

        # Read remaining headers (discard)
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if line == b"\r\n" or line == b"\n" or not line:
                break

        if request_str.startswith("GET /health"):
            status, body = await handle_health_request()
            status_text = "OK" if status == 200 else "Service Unavailable"
        else:
            status = 404
            status_text = "Not Found"
            body = json.dumps({"error": "Not found. Try GET /health"})

        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()
    except Exception:
        logger.exception("Error handling health check request")
    finally:
        writer.close()
        await writer.wait_closed()


async def start_health_server(port: int) -> asyncio.Server:
    """Start the HTTP health check server."""
    server = await asyncio.start_server(_handle_client, "0.0.0.0", port)
    addr = server.sockets[0].getsockname()
    logger.info("Health check server listening on %s:%s", addr[0], addr[1])
    return server
