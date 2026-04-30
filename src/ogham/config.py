from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sensible batch defaults per provider. Each provider has different API limits:
#   OpenAI:  2,048 inputs / 300K tokens per request
#   Voyage:  1,000 inputs / 120K-1M tokens per request
#   Mistral: 16,384 total tokens per request (~32 typical memories)
#   Ollama:  no API limit, but CPU/memory bound locally
PROVIDER_BATCH_DEFAULTS: dict[str, int] = {
    "ollama": 10,
    "openai": 500,
    "mistral": 32,
    "voyage": 500,
    "gemini": 100,
}

# Default embedding dimensions per provider (used when EMBEDDING_DIM is not set).
PROVIDER_DEFAULT_DIMS: dict[str, int] = {
    "ollama": 512,
    "openai": 1024,
    "mistral": 1024,
    "voyage": 1024,
    "gemini": 512,
}


def _find_env_files() -> tuple[str, ...]:
    """Find env files: project .env first, then ~/.ogham/config.env as fallback."""
    from pathlib import Path

    files = []
    # Project-level .env (highest priority)
    if Path(".env").exists():
        files.append(".env")
    # Global fallback
    global_env = Path.home() / ".ogham" / "config.env"
    if global_env.exists():
        files.append(str(global_env))
    return tuple(files) if files else (".env",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_backend: str = "supabase"
    database_url: str | None = None

    supabase_url: str = ""
    supabase_key: str = ""

    embedding_provider: str = "ollama"
    embedding_dim: int | None = None

    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "embeddinggemma"

    openai_api_key: str | None = None
    mistral_api_key: str | None = None
    voyage_api_key: str | None = None
    gemini_api_key: str | None = None

    mistral_embed_model: str = "mistral-embed"
    voyage_embed_model: str = "voyage-4-lite"
    gemini_embed_model: str = "gemini-embedding-2-preview"

    onnx_model_path: str = ""

    default_match_threshold: float = 0.7
    default_match_count: int = 10

    default_profile: str = "default"

    ollama_timeout: int = 60
    embedding_batch_size: int | None = None

    # Persistent SQLite embedding cache at ~/.cache/ogham/embeddings.db by
    # default. 100K entries covers normal daily use (~300 MB at 768 dim).
    # For a full 500q LongMemEval clean-re-run (~260K memories), bump to
    # 500000 via EMBEDDING_CACHE_MAX_SIZE so the second run is cache-hit.
    embedding_cache_max_size: int = 100000
    embedding_cache_dir: str | None = None

    # Temporal LLM fallback: model for resolving complex date expressions.
    # Empty = parsedatetime only (no LLM calls). Set to an Ollama model name
    # (e.g. "llama3.2") for local LLM, or "gpt-4o-mini" for OpenAI, or any
    # litellm-compatible model string. Requires litellm installed.
    temporal_llm_model: str = ""

    rerank_enabled: bool = False
    rerank_model: str = "flashrank"  # "flashrank" or "bge"
    rerank_alpha: float = 0.55

    # Wiki Tier 1 (v0.12.1) -- prepend top-K matching topic_summaries to
    # the MCP `hybrid_search` tool's results as a context-injection layer.
    # Default True since v0.12.1: the injection happens at the tool layer
    # (tools/memory.py::hybrid_search), not in service.search_memories_enriched,
    # so benchmarks calling the service directly never see preamble
    # pollution. MCP clients (Claude / Cursor / OpenCode) get the
    # synthesized topic context as preamble before raw memories.
    wiki_injection_enabled: bool = True
    wiki_injection_top_k: int = 3
    wiki_injection_min_similarity: float = 0.4

    # Hard cap on source-memory count for compile_wiki. Mega-rollup tags
    # (e.g. "type:gotcha", "project:foo") accumulate hundreds of memories
    # over time; synthesizing all of them into one ~1500-word page produces
    # very large LLM outputs that fail JSON escaping reliability and saturate
    # context budgets without producing a coherent page. Surfaced 2026-04-29
    # by Hotfix C: project:ogham (687 mem, 1.4M char prompt) consistently
    # failed even on Gemini 2.5 Pro. Above this cap, compile_wiki refuses
    # by default; pass `force_oversize=True` to override for one-off intent.
    # 0 disables the check.
    compile_max_sources: int = 100

    # Locale for wiki-layer prompt templates and user-facing messages.
    # Two-letter language code matching a file under src/ogham/data/languages/
    # (en, de, fr, ja, ...). Falls back to English when a key is missing
    # in the requested locale, so partial localisations don't break the
    # compile pipeline.
    locale: str = "en"

    bare_postgrest: bool = False

    enable_http_health: bool = False
    health_port: int = 8080

    server_transport: str = Field(default="stdio", validation_alias="OGHAM_TRANSPORT")
    server_host: str = Field(default="127.0.0.1", validation_alias="OGHAM_HOST")
    server_port: int = Field(default=8742, validation_alias="OGHAM_PORT")

    recall_enabled: bool = Field(default=True, validation_alias="OGHAM_RECALL_ENABLED")
    inscribe_enabled: bool = Field(default=True, validation_alias="OGHAM_INSCRIBE_ENABLED")

    gateway_url: str = Field(default="", validation_alias="OGHAM_GATEWAY_URL")
    gateway_api_key: str = Field(default="", validation_alias="OGHAM_API_KEY")

    @field_validator("database_backend")
    @classmethod
    def check_database_backend(cls, v: str) -> str:
        allowed = {"supabase", "postgres", "gateway"}
        if v not in allowed:
            raise ValueError(f"database_backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def check_provider(cls, v: str) -> str:
        allowed = {"ollama", "openai", "mistral", "voyage", "gemini", "onnx"}
        if v not in allowed:
            raise ValueError(f"embedding_provider must be one of {allowed}, got {v!r}")
        return v

    @field_validator("server_transport")
    @classmethod
    def check_transport(cls, v: str) -> str:
        allowed = {"stdio", "sse"}
        if v not in allowed:
            raise ValueError(f"server_transport must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        """Set provider-aware defaults.

        Backend credential validation happens at backend/client initialization
        time so modules can inspect default settings without requiring a fully
        configured runtime environment.
        """
        if self.embedding_dim is None:
            self.embedding_dim = PROVIDER_DEFAULT_DIMS.get(self.embedding_provider, 1024)
        if self.embedding_batch_size is None:
            self.embedding_batch_size = PROVIDER_BATCH_DEFAULTS.get(self.embedding_provider, 50)
        return self


def _lazy_settings():
    """Lazy proxy so 'ogham init' can run before config exists."""
    _instance = None

    class _Proxy:
        def __getattr__(self, name):
            nonlocal _instance
            if _instance is None:
                _instance = Settings()
            return getattr(_instance, name)

        def _reset(self):
            """Discard cached instance so next access re-reads env vars."""
            nonlocal _instance
            _instance = None

        def _force(self):
            """Force a fresh Settings instance from current env vars."""
            nonlocal _instance
            _instance = Settings()

    return _Proxy()


settings = _lazy_settings()
