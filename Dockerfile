# --- Stage 1: Build ---
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

# --- Stage 2: Runtime ---
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://host.docker.internal:11434

ENTRYPOINT ["ogham"]
