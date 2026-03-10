from unittest.mock import MagicMock, patch

import pytest

from ogham.retry import with_retry


def test_retry_succeeds_on_second_attempt():
    """Should retry and succeed if second attempt works"""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("First attempt failed")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 2


def test_retry_fails_after_max_attempts():
    """Should raise after max_attempts exhausted"""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01)
    def always_fails():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Always fails")

    with pytest.raises(ConnectionError, match="Always fails"):
        always_fails()

    assert call_count == 3


def test_retry_only_catches_specified_exceptions():
    """Should not retry on non-specified exceptions"""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
    def raises_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("Not a ValueError")

    with pytest.raises(TypeError):
        raises_type_error()

    assert call_count == 1  # No retry


def test_embedding_generation_retries_on_connection_error():
    """generate_embedding should retry on Ollama connection errors"""
    from ogham.embeddings import generate_embedding

    call_count = 0

    def mock_ollama_embed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Ollama not responding")
        return {"embeddings": [[0.1] * 1024]}

    with patch("ogham.embeddings.settings") as mock_settings:
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_dim = 1024
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_embed_model = "mxbai-embed-large"
        mock_settings.embedding_cache_max_size = 1000

        mock_client = MagicMock()
        mock_client.embed = mock_ollama_embed
        import uuid
        unique_text = f"retry_test_{uuid.uuid4()}"
        with patch("ogham.embeddings._ollama_client", mock_client):
            result = generate_embedding(unique_text)

            assert len(result) == 1024
            assert call_count == 2  # Failed once, succeeded on retry
