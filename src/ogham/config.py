from pydantic import field_validator, model_validator
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
}

# Default embedding dimensions per provider (used when EMBEDDING_DIM is not set).
PROVIDER_DEFAULT_DIMS: dict[str, int] = {
    "ollama": 512,
    "openai": 1024,
    "mistral": 1024,
    "voyage": 1024,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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

    mistral_embed_model: str = "mistral-embed"
    voyage_embed_model: str = "voyage-4-lite"

    default_match_threshold: float = 0.7
    default_match_count: int = 10

    default_profile: str = "default"

    ollama_timeout: int = 60
    embedding_batch_size: int | None = None

    embedding_cache_max_size: int = 10000
    embedding_cache_dir: str | None = None

    bare_postgrest: bool = False

    enable_http_health: bool = False
    health_port: int = 8080

    @field_validator("database_backend")
    @classmethod
    def check_database_backend(cls, v: str) -> str:
        allowed = {"supabase", "postgres"}
        if v not in allowed:
            raise ValueError(f"database_backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def check_provider(cls, v: str) -> str:
        allowed = {"ollama", "openai", "mistral", "voyage"}
        if v not in allowed:
            raise ValueError(f"embedding_provider must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        """Set provider-aware defaults and validate backend config."""
        if self.embedding_dim is None:
            self.embedding_dim = PROVIDER_DEFAULT_DIMS.get(
                self.embedding_provider, 1024
            )
        if self.embedding_batch_size is None:
            self.embedding_batch_size = PROVIDER_BATCH_DEFAULTS.get(
                self.embedding_provider, 50
            )
        if self.database_backend == "supabase" and not self.supabase_url:
            raise ValueError("SUPABASE_URL is required when DATABASE_BACKEND=supabase")
        if self.database_backend == "postgres" and not self.database_url:
            raise ValueError("DATABASE_URL is required when DATABASE_BACKEND=postgres")
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
