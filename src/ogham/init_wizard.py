"""Interactive setup wizard for Ogham MCP.

Walks users through database configuration, embedding provider selection,
schema migration, and MCP client configuration.

Supports: Claude Desktop, Claude Code, Cursor, VS Code (Copilot), Codex CLI, Kiro, OpenCode.
Platforms: macOS, Linux, Windows.
Execution modes: uvx (default), Docker (GHCR image).
"""

import json
import os
import platform
import shutil
from importlib.metadata import version as pkg_version
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


def _get_version() -> str:
    """Get the installed ogham-mcp version."""
    try:
        return pkg_version("ogham-mcp")
    except Exception:
        return "dev"


console = Console()

GHCR_IMAGE = "ghcr.io/ogham-mcp/ogham-mcp:latest"


# ---------------------------------------------------------------------------
# Client detection
# ---------------------------------------------------------------------------


def _client_configs() -> list[dict]:
    """Return all known MCP client config locations for the current platform."""
    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        appdata = home / "Library" / "Application Support"
    elif system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    else:
        appdata = home / ".config"

    clients = [
        {
            "name": "Claude Desktop",
            "path": {
                "Darwin": appdata / "Claude" / "claude_desktop_config.json",
                "Linux": home / ".config" / "Claude" / "claude_desktop_config.json",
                "Windows": appdata / "Claude" / "claude_desktop_config.json",
            }.get(system),
            "format": "claude_desktop",
        },
        {
            "name": "Project .mcp.json (Claude Code + Cursor)",
            "path": Path.cwd() / ".mcp.json",
            "always_show": True,
            "format": "mcp_json",
        },
        {
            "name": "Claude Code (global)",
            "path": home / ".claude" / ".mcp.json",
            "detect": home / ".claude",
            "format": "mcp_json",
        },
        {
            "name": "Cursor (global)",
            "path": home / ".cursor" / "mcp.json",
            "detect": home / ".cursor",
            "format": "mcp_json",
        },
        {
            "name": "VS Code (Copilot)",
            "path": Path.cwd() / ".vscode" / "mcp.json",
            "detect": Path.cwd() / ".vscode",
            "format": "vscode",
        },
        {
            "name": "Codex CLI",
            "path": home / ".codex" / "config.toml",
            "detect_cmd": "codex",
            "format": "codex_toml",
        },
        {
            "name": "Kiro",
            "path": home / ".kiro" / "settings" / "mcp.json",
            "detect": home / ".kiro",
            "format": "mcp_json",
        },
        {
            "name": "OpenCode",
            "path": home / ".config" / "opencode" / "opencode.json",
            "detect": home / ".config" / "opencode",
            "format": "opencode",
        },
    ]

    return [c for c in clients if c.get("path")]


def _detect_clients() -> list[dict]:
    """Find installed MCP clients."""
    detected = []
    for client in _client_configs():
        path = client["path"]
        # Check if the client directory or config exists
        detect_dir = client.get("detect")
        detect_cmd = client.get("detect_cmd")

        if client.get("always_show"):
            detected.append(client)
        elif detect_dir and detect_dir.exists():
            detected.append(client)
        elif detect_cmd and shutil.which(detect_cmd):
            detected.append(client)
        elif path.exists():
            detected.append(client)

    return detected


# ---------------------------------------------------------------------------
# Database prompts
# ---------------------------------------------------------------------------


def _prompt_database() -> dict:
    """Prompt for database backend and connection details."""
    console.print("\n[bold]2. Where should Ogham store your memories?[/bold]")
    console.print("   Both options use PostgreSQL with pgvector. Both have free tiers.\n")
    console.print("   [bold]1)[/bold] supabase  -- hosted Postgres (supabase.com)")
    console.print(
        "   [bold]2)[/bold] postgres  -- Neon, self-hosted, or any Postgres with pgvector\n"
    )

    choice = Prompt.ask(
        "   Choose",
        choices=["1", "2", "supabase", "postgres"],
        default="1",
    )
    backend = "supabase" if choice in ("1", "supabase") else "postgres"

    env_vars = {"DATABASE_BACKEND": backend}

    if backend == "supabase":
        console.print("\n   Great. You'll need two things from your Supabase dashboard:")
        console.print("   [cyan]Settings -> API Keys[/cyan]")
        console.print(
            "   [yellow]Ogham needs the secret key (formerly service_role) --"
            " the anon key is denied by RLS.[/yellow]"
        )
        default_url = os.environ.get("SUPABASE_URL", "")
        default_key = os.environ.get("SUPABASE_KEY", "")
        if default_url:
            console.print("   [cyan]Found SUPABASE_URL in environment.[/cyan]")
        if default_key:
            console.print("   [cyan]Found SUPABASE_KEY in environment.[/cyan]")
        url = Prompt.ask("   Supabase project URL", default=default_url or None)
        key = Prompt.ask(
            "   Supabase secret key (formerly service_role)",
            default=default_key or None,
        )
        env_vars["SUPABASE_URL"] = url
        env_vars["SUPABASE_KEY"] = key
    else:
        console.print("\n   Great. Paste your PostgreSQL connection string.")
        console.print("   [cyan]Neon: Dashboard -> Connection Details -> Connection string[/cyan]")
        console.print("   [cyan]Format: postgresql://user:pass@host:5432/dbname[/cyan]")
        console.print(
            "   [cyan]Neon tip: pooler endpoint works fine."
            " Use direct endpoint only for schema migrations.[/cyan]"
        )
        default_url = os.environ.get("DATABASE_URL", "")
        if default_url:
            console.print("   [cyan]Found DATABASE_URL in environment.[/cyan]")
        url = Prompt.ask("   Database URL", default=default_url or None)
        env_vars["DATABASE_URL"] = url

    return env_vars


# ---------------------------------------------------------------------------
# Embedding prompts
# ---------------------------------------------------------------------------


def _prompt_embeddings() -> dict:
    """Prompt for embedding provider and credentials."""
    console.print("\n[bold]1. Pick your embedding provider[/bold]")
    console.print("   Ogham turns text into vectors for search. Which provider do you want?\n")
    console.print("   [bold]1)[/bold] ollama   -- free, runs on your machine (must be running)")
    console.print("   [bold]2)[/bold] openai   -- $0.02/1M tokens (API key required)")
    console.print("   [bold]3)[/bold] voyage   -- $0.02/1M tokens + 200M free (API key required)")
    console.print("   [bold]4)[/bold] mistral  -- fixed 1024 dims (API key required)\n")

    provider_map = {"1": "ollama", "2": "openai", "3": "voyage", "4": "mistral"}
    choice = Prompt.ask(
        "   Choose",
        choices=["1", "2", "3", "4", "ollama", "openai", "voyage", "mistral"],
        default="1",
    )
    provider = provider_map.get(choice, choice)

    env_vars = {"EMBEDDING_PROVIDER": provider}

    if provider == "openai":
        default_key = os.environ.get("OPENAI_API_KEY", "")
        if default_key:
            console.print("   [cyan]Found OPENAI_API_KEY in environment.[/cyan]")
        key = Prompt.ask("   OpenAI API key", default=default_key or None)
        env_vars["OPENAI_API_KEY"] = key
    elif provider == "voyage":
        default_key = os.environ.get("VOYAGE_API_KEY", "")
        if default_key:
            console.print("   [cyan]Found VOYAGE_API_KEY in environment.[/cyan]")
        key = Prompt.ask("   Voyage AI API key", default=default_key or None)
        env_vars["VOYAGE_API_KEY"] = key
    elif provider == "mistral":
        default_key = os.environ.get("MISTRAL_API_KEY", "")
        if default_key:
            console.print("   [cyan]Found MISTRAL_API_KEY in environment.[/cyan]")
        key = Prompt.ask("   Mistral API key", default=default_key or None)
        env_vars["MISTRAL_API_KEY"] = key
    elif provider == "ollama":
        default_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        url = Prompt.ask("   Ollama URL", default=default_url)
        env_vars["OLLAMA_URL"] = url

    if provider == "mistral":
        env_vars["EMBEDDING_DIM"] = "1024"
        console.print(
            "   [yellow]Mistral uses fixed 1024 dimensions -- schema will be set to 1024.[/yellow]"
        )
    else:
        env_vars["EMBEDDING_DIM"] = "512"

    return env_vars


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------


def _prompt_transport() -> tuple[str, str, int]:
    """Ask transport mode. Returns (transport, host, port)."""
    console.print("\n[bold]3. Server transport[/bold]")
    console.print("   [bold]1)[/bold] Stdio  -- spawned per session (default, works everywhere)")
    console.print(
        "   [bold]2)[/bold] SSE    -- persistent background server (better for multiple agents)\n"
    )

    choice = Prompt.ask(
        "   Choose",
        choices=["1", "2", "stdio", "sse"],
        default="1",
    )

    if choice in ("2", "sse"):
        host = Prompt.ask("   SSE bind host", default="127.0.0.1")
        port_str = Prompt.ask("   SSE port", default="8742")
        return "sse", host, int(port_str)

    return "stdio", "127.0.0.1", 8742


def _prompt_execution_mode() -> str:
    """Ask whether to use uvx or Docker (stdio mode only)."""
    console.print("\n[bold]4. How should your AI clients run Ogham?[/bold]")
    console.print("   [bold]1)[/bold] uvx    -- Python package, lightweight, fast startup")
    console.print("   [bold]2)[/bold] docker -- container image, all dependencies included\n")

    choice = Prompt.ask(
        "   Choose",
        choices=["1", "2", "uvx", "docker"],
        default="1",
    )
    return "uvx" if choice in ("1", "uvx") else "docker"


def _build_mcp_entry(
    env_vars: dict,
    mode: str,
    transport: str = "stdio",
    sse_host: str = "127.0.0.1",
    sse_port: int = 8742,
) -> dict:
    """Build the MCP server config entry for Ogham."""
    if transport == "sse":
        return {"url": f"http://{sse_host}:{sse_port}/sse"}

    env = dict(env_vars)

    if mode == "docker":
        # Inside Docker, localhost means the container -- swap to host
        for key in env:
            env[key] = env[key].replace("localhost", "host.docker.internal")
            env[key] = env[key].replace("127.0.0.1", "host.docker.internal")
        return {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                *[arg for k, v in env.items() for arg in ["-e", f"{k}={v}"]],
                GHCR_IMAGE,
            ],
        }
    else:
        return {
            "command": "uvx",
            "args": ["ogham-mcp"],
            "env": env,
        }


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def _find_schema_file(backend: str) -> Path | None:
    """Locate the appropriate schema SQL file."""
    filename = "schema.sql" if backend == "supabase" else "schema_postgres.sql"

    candidates = [
        Path(__file__).parent.parent.parent / "sql" / filename,
        Path(__file__).parent / "sql" / filename,
        Path.cwd() / "sql" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _adjust_schema_dim(sql: str, dim: str) -> str:
    """Replace vector(512) with the configured dimension if different."""
    if dim and dim != "512":
        return sql.replace("vector(512)", f"vector({dim})")
    return sql


def _run_schema(env_vars: dict) -> bool:
    """Run the appropriate schema SQL against the database."""
    console.print("\n[bold]4. Database schema[/bold]")

    backend = env_vars.get("DATABASE_BACKEND", "supabase")
    dim = env_vars.get("EMBEDDING_DIM", "512")

    if backend == "supabase" and not env_vars.get("DATABASE_URL"):
        console.print("   Ogham needs tables and functions in your database.")
        console.print("   For Supabase, paste the schema SQL into the SQL Editor.")
        schema_path = _find_schema_file("supabase")
        if schema_path and dim != "512":
            # Write a dimension-adjusted copy next to the original
            adjusted_path = schema_path.parent / f"schema_{dim}d.sql"
            adjusted_sql = _adjust_schema_dim(schema_path.read_text(), dim)
            adjusted_path.write_text(adjusted_sql)
            console.print(
                f"   [yellow]Adjusted schema for {dim} dims:[/yellow] [bold]{adjusted_path}[/bold]"
            )
        elif schema_path:
            console.print(f"   Schema file: [bold]{schema_path}[/bold]")
        else:
            console.print("   Schema file: [bold]sql/schema.sql[/bold] (from the repo)")
        console.print(
            "   [cyan]Steps: Supabase dashboard -> SQL Editor -> New query ->"
            " paste contents -> Run[/cyan]"
        )
        if Confirm.ask("   Have you already run the schema?", default=False):
            return True
        console.print(
            "   [cyan]You can run it later. Ogham will work once the schema is in place.[/cyan]"
        )
        return False

    if backend == "postgres":
        db_url = env_vars.get("DATABASE_URL", "")
        if not db_url:
            console.print("   [red]No DATABASE_URL -- skipping schema migration.[/red]")
            return False

        console.print("   We'll create the tables, indexes, and functions for you.")
        if not Confirm.ask("   Run schema now?", default=True):
            return False

        try:
            import psycopg
        except ImportError:
            console.print(
                "   [yellow]psycopg not installed."
                " Install with: uv add ogham-mcp[postgres][/yellow]"
            )
            console.print("   Run the schema manually: sql/schema_postgres.sql")
            return False

        schema_path = _find_schema_file("postgres")
        if not schema_path:
            console.print(
                "   [red]Schema file not found. Run manually: sql/schema_postgres.sql[/red]"
            )
            return False

        schema_sql = _adjust_schema_dim(schema_path.read_text(), dim)
        if dim != "512":
            console.print(f"   [cyan]Schema adjusted for {dim} dimensions.[/cyan]")
        try:
            with console.status("   Running schema migration..."):
                conn = psycopg.connect(db_url)
                conn.autocommit = True
                conn.execute(schema_sql)
                conn.close()
            console.print("   [green]Schema migration complete.[/green]")
            return True
        except Exception as e:
            console.print(f"   [red]Schema migration failed: {e}[/red]")
            return False

    return False


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


def _write_codex_toml(config_path: Path, mcp_entry: dict):
    """Write [mcp_servers.ogham] into Codex config.toml.

    Codex sometimes writes non-standard TOML (inline Python dicts),
    so we can't reliably parse the full file. Instead, strip any
    existing [mcp_servers.ogham] block and append a clean one.
    """
    import re

    content = config_path.read_text() if config_path.exists() else ""

    # Remove existing [mcp_servers.ogham] and [mcp_servers.ogham.env]
    # blocks. Match from the header to the next section header or EOF.
    # The args line contains [] so we can't use [^\[]* -- instead match
    # lines that don't start with [
    content = re.sub(
        r"\n*\[mcp_servers\.ogham(?:\.\w+)?\]\n(?:(?!\[).*\n?)*",
        "",
        content,
    )
    content = content.rstrip() + "\n"

    # Build the new ogham section
    lines = ["\n[mcp_servers.ogham]"]
    lines.append(f'command = "{mcp_entry["command"]}"')
    if mcp_entry.get("args"):
        args_str = ", ".join(f'"{a}"' for a in mcp_entry["args"])
        lines.append(f"args = [{args_str}]")
    lines.append("enabled = true")
    if mcp_entry.get("env"):
        lines.append("")
        lines.append("[mcp_servers.ogham.env]")
        for k, v in mcp_entry["env"].items():
            lines.append(f'{k} = "{v}"')

    config_path.write_text(content + "\n".join(lines) + "\n")


def _write_mcp_config(client: dict, mcp_entry: dict):
    """Merge Ogham MCP entry into a client's config file."""
    config_path = client["path"]
    fmt = client["format"]

    config_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "codex_toml":
        _write_codex_toml(config_path, mcp_entry)
        return

    if config_path.exists():
        existing = json.loads(config_path.read_text())
    else:
        existing = {}

    if fmt == "opencode":
        # OpenCode uses "mcp" key with a different structure
        if "mcp" not in existing:
            existing["mcp"] = {}
        existing["mcp"]["ogham"] = {
            "type": "local",
            "command": [mcp_entry["command"]] + mcp_entry.get("args", []),
            "environment": mcp_entry.get("env", {}),
            "enabled": True,
        }
    elif fmt == "vscode":
        # VS Code uses "servers" key inside mcp.json
        if "servers" not in existing:
            existing["servers"] = {}
        existing["servers"]["ogham"] = mcp_entry
    else:
        # Standard mcpServers format (Claude Desktop, Claude Code, Cursor)
        if "mcpServers" not in existing:
            existing["mcpServers"] = {}
        existing["mcpServers"]["ogham"] = mcp_entry

    config_path.write_text(json.dumps(existing, indent=2) + "\n")


def _configure_clients(
    env_vars: dict,
    mode: str,
    transport: str = "stdio",
    sse_host: str = "127.0.0.1",
    sse_port: int = 8742,
) -> list[str]:
    """Write MCP config to detected AI clients."""
    step = "5" if transport == "stdio" else "4"
    console.print(f"\n[bold]{step}. Connect your AI clients[/bold]")

    detected = _detect_clients()
    if not detected:
        console.print("   [yellow]No MCP clients detected.[/yellow]")
        console.print(
            "   [cyan]You can configure manually later. See: ogham-mcp.dev/docs/quickstart[/cyan]"
        )
        return []

    console.print(f"   Found {len(detected)} client(s) on your machine.\n")
    if transport == "sse":
        console.print(f"   SSE mode: clients will connect to http://{sse_host}:{sse_port}/sse\n")
    else:
        console.print("   For each one, we'll add Ogham to its MCP config.\n")

    mcp_entry = _build_mcp_entry(env_vars, mode, transport, sse_host, sse_port)
    configured = []

    for client in detected:
        name = client["name"]
        path = client["path"]
        if Confirm.ask(f"   Configure {name}? ({path})", default=True):
            try:
                _write_mcp_config(client, mcp_entry)
                configured.append(name)
                console.print(f"   [green]{name} configured.[/green]")
            except Exception as e:
                console.print(f"   [red]Failed to configure {name}: {e}[/red]")

    return configured


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------


def _test_connection(env_vars: dict) -> bool:
    """Validate database and embedding connectivity."""
    console.print("\n[bold]6. Let's check everything works[/bold]")

    # Temporarily set env vars for the health check
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Force fresh settings instance from the temp env vars
        from ogham.config import settings

        settings._force()

        # Check database
        with console.status("   Checking database..."):
            from ogham.health import check_database

            db_result = check_database()

        if db_result["status"] == "ok":
            console.print("   [green]Database: connected[/green]")
        else:
            console.print(f"   [red]Database: {db_result.get('error', 'failed')}[/red]")
            return False

        # Check embedding provider
        with console.status("   Checking embedding provider..."):
            from ogham.health import check_embedding_provider

            emb_result = check_embedding_provider()

        if emb_result["status"] == "ok":
            prov = env_vars.get("EMBEDDING_PROVIDER", "ollama")
            console.print(f"   [green]Embeddings: {prov} ready[/green]")
        else:
            console.print(
                f"   [yellow]Embeddings: {emb_result.get('error', 'check failed')}[/yellow]"
            )
            if hint := emb_result.get("hint"):
                console.print(f"   [cyan]{hint}[/cyan]")

        return db_result["status"] == "ok"

    finally:
        # Restore original env and settings
        for key, original in original_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        settings._reset()


# ---------------------------------------------------------------------------
# .env file
# ---------------------------------------------------------------------------


def _write_env_file(env_vars: dict):
    """Write config to ~/.ogham/config.env (global) or .env (project)."""
    choices = {
        "global": "~/.ogham/config.env (works from any project)",
        "project": ".env in current directory",
        "skip": "Don't save",
    }
    console.print("\n   Where to save config?")
    for key, desc in choices.items():
        console.print(f"     [bold]{key}[/bold]: {desc}")

    choice = Prompt.ask("   Choice", choices=list(choices.keys()), default="global")

    if choice == "skip":
        return

    lines = [f"{k}={v}" for k, v in env_vars.items()]
    content = "\n".join(lines) + "\n"

    if choice == "global":
        ogham_dir = Path.home() / ".ogham"
        ogham_dir.mkdir(parents=True, exist_ok=True)
        env_path = ogham_dir / "config.env"
        env_path.write_text(content)
        # Restrict permissions -- contains API keys
        env_path.chmod(0o600)
        console.print(f"   [green]Saved to {env_path}[/green]")
        console.print(
            "   [dim]Hooks and CLI will find this automatically from any directory.[/dim]"
        )
    else:
        env_path = Path.cwd() / ".env"
        env_path.write_text(content)
        console.print(f"   [green]Saved to {env_path}[/green]")
        console.print("   [yellow]Add .env to your .gitignore -- it contains API keys.[/yellow]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_init(
    db_url: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    backend: str | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    dim: int | None = None,
    mode: str | None = None,
    skip_schema: bool = False,
    skip_clients: bool = False,
    skip_test: bool = False,
):
    """Run the init wizard, interactive or non-interactive."""
    console.clear()
    console.print(
        Panel(
            "[bold]Ogham MCP Setup[/bold]\n\n"
            "This wizard configures your database, embedding provider,\n"
            "and MCP client connections.\n\n"
            "You only need to run this once. Run it again if you want to\n"
            "switch database or embedding provider.\n\n"
            "[cyan]Detected platform: " + platform.system() + "[/cyan]\n"
            "[cyan]Version: " + _get_version() + "[/cyan]",
            border_style="yellow",
        )
    )

    # Non-interactive mode if key args provided
    if db_url or (supabase_url and supabase_key):
        env_vars = {}
        if db_url:
            env_vars["DATABASE_BACKEND"] = backend or "postgres"
            env_vars["DATABASE_URL"] = db_url
        elif supabase_url:
            env_vars["DATABASE_BACKEND"] = "supabase"
            env_vars["SUPABASE_URL"] = supabase_url
            env_vars["SUPABASE_KEY"] = supabase_key

        env_vars["EMBEDDING_PROVIDER"] = provider or "ollama"
        if api_key:
            key_map = {
                "openai": "OPENAI_API_KEY",
                "voyage": "VOYAGE_API_KEY",
                "mistral": "MISTRAL_API_KEY",
            }
            env_key = key_map.get(env_vars["EMBEDDING_PROVIDER"])
            if env_key:
                env_vars[env_key] = api_key

        env_vars["EMBEDDING_DIM"] = str(dim or 512)
        exec_mode = mode or "uvx"
        transport, sse_host, sse_port = "stdio", "127.0.0.1", 8742
    else:
        # Interactive mode — embeddings first so we know the dimension for schema
        env_vars = _prompt_embeddings()
        env_vars.update(_prompt_database())
        transport, sse_host, sse_port = _prompt_transport()
        if transport == "sse":
            exec_mode = "uvx"  # irrelevant for SSE but needs a value
            env_vars["OGHAM_TRANSPORT"] = "sse"
            env_vars["OGHAM_HOST"] = sse_host
            env_vars["OGHAM_PORT"] = str(sse_port)
        else:
            exec_mode = _prompt_execution_mode()

    # Schema migration
    if not skip_schema:
        _run_schema(env_vars)

    # Configure MCP clients
    configured = []
    if not skip_clients:
        configured = _configure_clients(env_vars, exec_mode, transport, sse_host, sse_port)

    # Test connection
    if not skip_test:
        _test_connection(env_vars)

    # Offer to save .env
    _write_env_file(env_vars)

    # Summary
    if transport == "sse":
        mode_str = f"SSE server on {sse_host}:{sse_port}"
        start_hint = (
            "[cyan]Start the server with:[/cyan] "
            "[bold]ogham serve --transport sse[/bold]\n"
            "[cyan]Then restart your MCP client(s) to connect.[/cyan]"
        )
    else:
        mode_str = "uvx ogham-mcp" if exec_mode == "uvx" else f"docker ({GHCR_IMAGE})"
        start_hint = "[cyan]Restart your MCP client(s) to pick up the new config.[/cyan]"

    console.print(
        Panel(
            "[bold green]You're all set![/bold green]\n\n"
            + (f"Clients configured: {', '.join(configured)}\n" if configured else "")
            + f"Database: {env_vars.get('DATABASE_BACKEND', 'supabase')}\n"
            + f"Embeddings: {env_vars.get('EMBEDDING_PROVIDER', 'ollama')}"
            + f" @ {env_vars.get('EMBEDDING_DIM', '512')}d\n"
            + f"Transport: {mode_str}\n\n"
            + start_hint,
            border_style="green",
        )
    )
