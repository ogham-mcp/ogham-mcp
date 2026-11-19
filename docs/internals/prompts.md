# Ogham MCP — Prompts & OpenAPI

## `prompts.py` — MCP Prompt Templates

Defines MCP prompt resources that clients can request. These are pre-built prompt templates for common memory operations.

### Available Prompts

| Prompt | Arguments | Purpose |
|--------|-----------|---------|
| `store-context` | `context` (text) | Wraps user-provided context into a store instruction |
| `recall` | `topic` (text) | Generates a search query for a topic |
| `session-summary` | `session_notes` (text) | Asks for a session summary to store |

Each prompt returns a `GetPromptResult` with a descriptive name and a list of messages (role + content) that the client can use to invoke the corresponding memory operation.

### Example

The `recall` prompt for topic "authentication" would return:

```text
Search your memory for everything you know about: authentication
Include related context and any decisions made.
```

## `openapi.py` — OpenAPI Schema Generator

Generates an OpenAPI 3.1.0 specification from the registered MCP tools.

### How It Works

1. Iterates over all tools registered on the FastMCP server
2. For each tool, creates a POST endpoint at `/tools/{tool_name}`
3. Extracts parameter schemas from the tool's input model (Pydantic)
4. Maps each tool's description to the OpenAPI operation description
5. Outputs valid OpenAPI JSON

### CLI Integration

Available via `ogham openapi` CLI command. Useful for:

- Documentation generation
- Gateway/proxy configuration
- Client SDK generation
- API testing tools (Postman, etc.)

### Output Structure

```json
{
  "openapi": "3.1.0",
  "info": {"title": "Ogham MCP", "version": "..."},
  "paths": {
    "/tools/store_memory": {
      "post": {
        "summary": "store_memory",
        "requestBody": { "content": { "application/json": { "schema": {...} } } },
        "responses": { "200": {...} }
      }
    }
  }
}
```
