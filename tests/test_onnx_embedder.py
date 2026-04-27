"""Tests for the ONNX BGE-M3 embedding provider."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ogham.onnx_embedder import (
    VOCAB_SIZE,
    OnnxResult,
    encode,
    sparse_to_sparsevec,
)

# ── sparse_to_sparsevec ─────────────────────────────────────────────


class TestSparseToSparsevec:
    def test_basic_format(self):
        sparse = {10: 0.5, 200: 0.3}
        result = sparse_to_sparsevec(sparse)
        # Indices are 1-based for pgvector
        assert result == "{11:0.500000,201:0.300000}/250002"

    def test_empty_sparse(self):
        assert sparse_to_sparsevec({}) == "{}/250002"

    def test_sorted_by_token_id(self):
        sparse = {300: 0.1, 50: 0.9, 150: 0.5}
        result = sparse_to_sparsevec(sparse)
        assert result.startswith("{51:")
        assert "151:" in result
        assert result.endswith("/250002")

    def test_custom_dim(self):
        sparse = {5: 0.1}
        result = sparse_to_sparsevec(sparse, dim=100)
        assert result == "{6:0.100000}/100"

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            sparse_to_sparsevec({300000: 0.5}, dim=VOCAB_SIZE)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            sparse_to_sparsevec({-1: 0.5})


# ── encode (mocked ONNX session) ────────────────────────────────────


def _make_mock_session(dense_output, sparse_output):
    """Create a mock ONNX session returning given outputs."""
    session = MagicMock()
    session.run.return_value = [dense_output, sparse_output]
    return session


def _make_mock_encoding(ids, attention_mask=None):
    """Create a mock tokenizer encoding."""
    enc = MagicMock()
    enc.ids = ids
    enc.attention_mask = attention_mask or [1] * len(ids)
    return enc


@pytest.fixture(autouse=True)
def _reset_onnx_singleton():
    """Reset the singleton model between tests."""
    import ogham.onnx_embedder as mod

    old_session, old_tokenizer = mod._session, mod._tokenizer
    mod._session = None
    mod._tokenizer = None
    yield
    mod._session = old_session
    mod._tokenizer = old_tokenizer


class TestEncode:
    def _run_encode(self, token_ids, sparse_weights_2d, dense_vec=None, attention_mask=None):
        """Helper: run encode() with mocked session and tokenizer."""
        import ogham.onnx_embedder as mod

        n_tokens = len(token_ids)
        if dense_vec is None:
            # L2-normalized 1024-dim vector
            dense_vec = np.zeros((1, 1024), dtype=np.float32)
            dense_vec[0, 0] = 1.0

        sparse_arr = np.array([[sparse_weights_2d[i] for i in range(n_tokens)]], dtype=np.float32)
        # sparse_weights shape: (1, n_tokens, 1) — per-token single weight
        # but the code does np.max(sparse_weights[0, i]) so shape (1, n_tokens) works too
        # Actually looking at the code: sparse_weights[0, i] then np.max — so it handles any
        # trailing dims. Let's use shape (1, n_tokens, 1) to match the model output.
        sparse_arr = sparse_arr.reshape(1, n_tokens, 1)

        mock_session = _make_mock_session(dense_vec, sparse_arr)
        mock_encoding = _make_mock_encoding(token_ids, attention_mask)

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = mock_encoding

        mod._session = mock_session
        mod._tokenizer = mock_tokenizer

        return encode("test text")

    def test_filters_special_tokens(self):
        # Token IDs 0,1,2,3 are special — should be excluded from sparse
        token_ids = [0, 100, 200, 2]  # CLS, real, real, SEP
        weights = [0.9, 0.5, 0.3, 0.8]  # high weights on specials to verify filtering

        result = self._run_encode(token_ids, weights)

        assert 0 not in result.sparse
        assert 2 not in result.sparse
        assert 100 in result.sparse
        assert 200 in result.sparse

    def test_filters_zero_weights(self):
        token_ids = [0, 100, 200, 2]
        weights = [0.0, 0.5, 0.0, 0.0]

        result = self._run_encode(token_ids, weights)

        assert 200 not in result.sparse
        assert result.sparse == {100: pytest.approx(0.5)}

    def test_duplicate_token_keeps_max(self):
        # Same token ID appears twice — should keep the max weight
        token_ids = [0, 100, 100, 2]
        weights = [0.0, 0.3, 0.7, 0.0]

        result = self._run_encode(token_ids, weights)

        assert result.sparse[100] == pytest.approx(0.7)

    def test_dense_output(self):
        dense = np.zeros((1, 1024), dtype=np.float32)
        dense[0, :3] = [0.6, 0.8, 0.0]  # L2 norm = 1.0
        token_ids = [0, 100, 2]
        weights = [0.0, 0.5, 0.0]

        result = self._run_encode(token_ids, weights, dense_vec=dense)

        assert len(result.dense) == 1024
        assert result.dense[0] == pytest.approx(0.6)
        assert result.dense[1] == pytest.approx(0.8)

    def test_respects_attention_mask(self):
        # Tokens with attention_mask=0 should be ignored
        token_ids = [0, 100, 200, 2]
        weights = [0.0, 0.5, 0.9, 0.0]
        attention_mask = [1, 1, 0, 1]  # token 200 masked out

        result = self._run_encode(token_ids, weights, attention_mask=attention_mask)

        assert 200 not in result.sparse
        assert 100 in result.sparse

    def test_returns_onnx_result(self):
        token_ids = [0, 100, 2]
        weights = [0.0, 0.5, 0.0]

        result = self._run_encode(token_ids, weights)

        assert isinstance(result, OnnxResult)
        assert isinstance(result.dense, list)
        assert isinstance(result.sparse, dict)

    def test_session_run_requests_specific_outputs(self):
        """Verify we request only dense+sparse, not ColBERT."""
        import ogham.onnx_embedder as mod

        token_ids = [0, 100, 2]
        weights = [0.0, 0.5, 0.0]

        self._run_encode(token_ids, weights)

        session = cast(Any, mod._session)
        session.run.assert_called_once()
        output_names = session.run.call_args[0][0]
        assert output_names == ["dense_embeddings", "sparse_weights"]


# ── Singleton / model loading ────────────────────────────────────────


class TestModelLoading:
    def test_missing_model_raises_helpful_error(self):
        """FileNotFoundError should mention download-model command."""
        import ogham.onnx_embedder as mod

        mod._session = None
        mod._tokenizer = None

        with patch("ogham.onnx_embedder.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False

            with pytest.raises(FileNotFoundError, match="download-model"):
                mod._get_model("/nonexistent/model.onnx")

        # Tokenizer should NOT have been loaded (check runs before tokenizer init)
        assert mod._tokenizer is None

    def test_singleton_returns_same_session(self):
        """Second call to _get_model should return cached session."""
        import ogham.onnx_embedder as mod

        fake_session = MagicMock()
        fake_tokenizer = MagicMock()
        mod._session = fake_session
        mod._tokenizer = fake_tokenizer

        tok, sess = mod._get_model()
        assert sess is fake_session
        assert tok is fake_tokenizer
