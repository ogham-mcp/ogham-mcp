# Ogham MCP — Init Wizard

`init_wizard.py` provides an interactive setup wizard that walks users through complete Ogham configuration.

## Supported Clients

| Client | Config Format | Key |
|--------|--------------|-----|
| Claude Desktop | JSON | `mcpServers` |
| Claude Code | JSON | `mcpServers` |
| Cursor | JSON | `mcpServers` |
| VS Code (Copilot) | JSON | `servers` |
| Codex CLI | JSON | `mcpServers` |
| Kiro | JSON | `mcpServers` |
| OpenCode | TOML | `[mcpServers.ogham]` |

Platform-aware config paths for macOS, Linux, and Windows.

## Execution Modes

| Mode | Command | Notes |
|------|---------|-------|
| `uvx` (default) | `uvx ogham-mcp serve` | Recommended for most users |
| Docker | `docker run ghcr.io/ogham-mcp/ogham-mcp:latest` | GHCR image, env vars passed via `--env` |

## Wizard Steps

### 1. Database Configuration (`_configure_database`)

- Prompts for PostgreSQL connection string
- Tests connection with `psycopg.connect()` + pgvector extension check (`SELECT 'vector'::regtype`)
- Validates that the connection actually works before proceeding

### 2. Embedding Provider (`_configure_embeddings`)

- Choice of: Ollama (local/free), OpenAI, Mistral, Voyage
- For Ollama: tests model availability via `ollama.Client().show(model)`
- For API providers: prompts for API key
- Sets `EMBEDDING_DIM` based on provider defaults

### 3. Schema Migration (`_run_migrations`)

- Checks for `sql/schema_postgres.sql` in package directory
- Runs full schema via `psycopg` if user approves
- Falls back to printing manual instructions if file not found

### 4. Client Configuration (`_configure_clients`)

- Auto-detects installed MCP clients by checking config file existence
- Lets user select which clients to configure
- Generates appropriate JSON/TOML config block
- Merges into existing config (preserves other MCP servers)
- Creates backup before modifying any config file

### 5. Env File (`_write_env`)

- Writes configuration to `~/.ogham/.env`
- Creates directory if needed
- Stores: `DATABASE_URL`, `DATABASE_BACKEND=postgres`, `EMBEDDING_PROVIDER`, provider-specific settings, `EMBEDDING_DIM`

## Client Config Generation

For uvx mode:

```json
{
  "command": "uvx",
  "args": ["ogham-mcp", "serve"],
  "env": {
    "DATABASE_URL": "...",
    "EMBEDDING_PROVIDER": "ollama"
  }
}
```

For Docker mode:

```json
{
  "command": "docker",
  "args": ["run", "-i", "--rm", "--env", "DATABASE_URL=...", "ghcr.io/ogham-mcp/ogham-mcp:latest"]
}
```

OpenCode uses TOML format with `type = "stdio"`.

## Safety

- Backs up existing config files before modification
- Tests database connection before saving
- Tests Ollama model availability before saving
- Validates pgvector extension is installed
