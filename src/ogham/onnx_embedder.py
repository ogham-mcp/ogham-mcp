"""ONNX BGE-M3 embedding provider.

Produces dense + sparse vectors in a single model pass using the
yuniko-software/bge-m3-onnx model with HuggingFace's `tokenizers` library.

All heavy imports (onnxruntime, tokenizers, numpy) are lazy — this module
is safe to import even when the ONNX deps aren't installed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# XLM-RoBERTa special tokens: [PAD]=1, [UNK]=3, [CLS]=0, [SEP]=2
_SPECIAL_TOKEN_IDS = frozenset({0, 1, 2, 3})

# BGE-M3 vocabulary size (XLM-RoBERTa)
VOCAB_SIZE = 250002


@dataclass
class OnnxResult:
    """Result from a single ONNX BGE-M3 forward pass."""

    dense: list[float]
    sparse: dict[int, float]


# ── Singleton model holder ────────────────────────────────────────────

_tokenizer = None
_session = None
_model_lock = threading.Lock()


def _get_model(model_path: str | None = None):
    """Lazy-load the ONNX session and tokenizer (singleton, thread-safe)."""
    global _tokenizer, _session
    if _session is not None:
        return _tokenizer, _session

    with _model_lock:
        # Double-check after acquiring lock
        if _session is not None:
            return _tokenizer, _session

        if model_path is None:
            model_path = str(Path.home() / ".cache" / "ogham" / "bge-m3-onnx" / "bge_m3_model.onnx")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run 'ogham download-model bge-m3' to download it."
            )

        import onnxruntime as ort
        from tokenizers import Tokenizer

        logger.info("Loading tokenizer for BAAI/bge-m3...")
        _tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")
        _tokenizer.enable_truncation(max_length=8192)
        _tokenizer.no_padding()

        logger.info("Loading ONNX model from %s...", model_path)
        options = ort.SessionOptions()
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = False  # release memory between inferences
        options.log_severity_level = 2  # WARNING
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        _session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX BGE-M3 model loaded.")
        return _tokenizer, _session


# ── Encoding ──────────────────────────────────────────────────────────


def encode(text: str, model_path: str | None = None) -> OnnxResult:
    """Encode a single text, returning dense + sparse vectors."""
    import numpy as np

    tokenizer, session = _get_model(model_path)

    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    dense_embeddings, sparse_weights = session.run(
        ["dense_embeddings", "sparse_weights"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )

    # Dense: already L2-normalized by the model export
    dense = dense_embeddings[0].tolist()

    # Sparse: per-token max weight, skip specials
    sparse: dict[int, float] = {}
    for i, token_id in enumerate(encoded.ids):
        if encoded.attention_mask[i] == 1 and token_id not in _SPECIAL_TOKEN_IDS:
            weight = float(np.max(sparse_weights[0, i]))
            if weight > 0:
                sparse[token_id] = max(sparse.get(token_id, 0), weight)

    return OnnxResult(dense=dense, sparse=sparse)


def encode_batch(texts: list[str], model_path: str | None = None) -> list[OnnxResult]:
    """Encode multiple texts sequentially.

    We disable padding and loop instead of batching because sparse weight
    extraction needs per-token attention masks without padding noise.
    Batching with padding would inflate sparse weights for pad positions.
    """
    return [encode(text, model_path) for text in texts]


# ── Sparse format conversion ─────────────────────────────────────────


def sparse_to_sparsevec(sparse: dict[int, float], dim: int = VOCAB_SIZE) -> str:
    """Convert sparse dict {token_id: weight} to pgvector sparsevec format.

    Format: '{idx1:val1, idx2:val2, ...}/dim'
    Indices are 1-based for pgvector.
    """
    if not sparse:
        return f"{{}}/{dim}"
    pairs = sorted(sparse.items())
    for tid, _ in pairs:
        if not (0 <= tid < dim):
            raise ValueError(f"Token ID {tid} out of bounds for vocab size {dim}")
    entries = ",".join(f"{tid + 1}:{weight:.6f}" for tid, weight in pairs)
    return f"{{{entries}}}/{dim}"
