import json

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")


def test_generate_openapi_spec():
    """Should generate valid OpenAPI spec with all tools"""
    from ogham.openapi import generate_openapi_spec

    spec = generate_openapi_spec()

    assert spec["openapi"] == "3.1.0"
    assert "Ogham" in spec["info"]["title"]
    assert len(spec["paths"]) > 0
    # Check that known tools are present
    tool_names = [p.split("/")[-1] for p in spec["paths"]]
    assert "store_memory" in tool_names
    assert "hybrid_search" in tool_names
    assert "health_check" in tool_names


def test_openapi_tool_has_description():
    """Each tool path should have a description"""
    from ogham.openapi import generate_openapi_spec

    spec = generate_openapi_spec()

    for path, methods in spec["paths"].items():
        post = methods.get("post", {})
        assert "summary" in post, f"Missing summary for {path}"
        assert "operationId" in post, f"Missing operationId for {path}"


def test_write_openapi_spec_json(tmp_path):
    """Should write spec to JSON file"""
    from ogham.openapi import write_openapi_spec

    output = tmp_path / "openapi.json"
    write_openapi_spec(str(output))

    assert output.exists()
    data = json.loads(output.read_text())
    assert data["openapi"] == "3.1.0"
