import pytest


@pytest.fixture(autouse=True)
def _reset_db_backend():
    """Reset the database backend singleton between tests."""
    from ogham.database import _reset_backend

    _reset_backend()
    yield
    _reset_backend()
