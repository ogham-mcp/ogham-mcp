# --- Stage 1: Build ---
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project --extra all

COPY src/ src/
# Phase B (v0.13): canonical SQL lives at sql/ only. The wheel ships
# migrations + the 3 schema files at ogham/sql/* via hatchling
# force-include, which means uv sync needs sql/ in the build context.
COPY sql/ sql/
RUN uv sync --frozen --no-dev --no-editable --extra all

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
