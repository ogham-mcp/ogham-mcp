"""Generate OpenAPI spec from MCP tool definitions."""

import json
from typing import Any

import ogham.tools.memory  # noqa: F401
import ogham.tools.stats  # noqa: F401
from ogham.app import mcp


def generate_openapi_spec() -> dict[str, Any]:
    """Generate an OpenAPI 3.1 spec from registered MCP tools."""
    spec: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {
            "title": "Ogham MCP",
            "description": (
                "MCP server providing persistent shared memory across AI clients. "
                "This spec documents the tool interfaces — the actual transport is "
                "MCP over stdio, not HTTP."
            ),
            "version": "0.1.0",
        },
        "paths": {},
    }

    tools = _get_tools()

    for tool in tools:
        tool_name = tool.name
        description = tool.description or ""
        path = f"/tools/{tool_name}"

        input_schema = {}
        if hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
            input_schema = tool.parameters

        spec["paths"][path] = {
            "post": {
                "operationId": tool_name,
                "summary": (description.split("\n")[0][:100] if description else tool_name),
                "description": description,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": input_schema if input_schema else {"type": "object"},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"},
                            }
                        },
                    }
                },
            }
        }

    return spec


def _get_tools() -> list:
    """Get list of registered tools from FastMCP."""
    # FastMCP stores components in _local_provider._components with keys like "tool:name@"
    provider = mcp._local_provider
    components = provider._components
    return [v for k, v in components.items() if k.startswith("tool:")]


def write_openapi_spec(output_path: str) -> None:
    """Generate and write OpenAPI spec to a file."""
    spec = generate_openapi_spec()

    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)
