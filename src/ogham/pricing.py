"""Embedding pricing helpers used by audit logging."""

from __future__ import annotations

from typing import Any

# Rates in USD per 1K input tokens for repo-enabled remote embedding models.
# Verified against official pricing pages on 2026-04-14. Re-check upstream
# pricing pages when repo defaults change or provider pricing is updated:
# The contributor brief showed a broader sample table; this module keeps the
# initial scope narrower to repo-enabled models and conservative Gemini handling.
# - OpenAI: https://platform.openai.com/docs/pricing/
# - Voyage: https://docs.voyageai.com/docs/pricing
# - Google Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/pricing?hl=en
EMBEDDING_PRICING_PER_1K_TOKENS: dict[str, float] = {
    "openai:text-embedding-3-small": 0.00002,
    "voyage:voyage-4-lite": 0.00002,
}


def calculate_embedding_cost(usage: dict[str, Any] | None) -> float | None:
    """Return estimated USD cost for an embedding request."""
    if not usage:
        return None

    model = usage.get("model")
    if not model:
        return None

    if model.startswith("ollama:") or model.startswith("onnx:"):
        return 0.0

    if model.startswith("gemini:"):
        return None

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        return None

    rate = EMBEDDING_PRICING_PER_1K_TOKENS.get(model)
    if rate is None:
        return None

    return round((float(input_tokens) / 1000.0) * rate, 8)
