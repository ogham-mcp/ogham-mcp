"""Unit tests for PostgresBackend query construction."""

from ogham.backends.postgres import _embedding_literal


def test_embedding_literal_format():
    """Embedding list is formatted as Postgres vector literal."""
    result = _embedding_literal([0.1, 0.2, 0.3])
    assert result == "[0.1,0.2,0.3]"


def test_embedding_literal_empty():
    """Empty embedding produces empty brackets."""
    result = _embedding_literal([])
    assert result == "[]"


def test_embedding_literal_precision():
    """Float precision is preserved."""
    result = _embedding_literal([0.123456789])
    assert "0.123456789" in result
