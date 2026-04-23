"""Tests for structured store wrappers (v0.10.1 item 7.1).

The wrappers are thin conveniences over store_memory that format content
into a standard prose shape and set the right tag + metadata. The tests
verify formatting + tagging + metadata without requiring a live backend.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.config import settings

    settings._reset()
    yield
    settings._reset()


# --- store_preference ---


def test_store_preference_basic():
    from ogham.tools.memory import store_preference

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}

        store_preference(
            preference="dark mode in the editor UI",
            subject="editor UI",
            alternatives=["light mode", "system default"],
            strength="strong",
        )

    kwargs = mock_store.call_args.kwargs
    assert "Preference: dark mode in the editor UI" in kwargs["content"]
    assert "Subject: editor UI" in kwargs["content"]
    assert "Rejected alternatives: light mode, system default" in kwargs["content"]
    assert "Strength: strong" in kwargs["content"]

    assert "type:preference" in kwargs["tags"]
    assert kwargs["metadata"]["type"] == "preference"
    assert kwargs["metadata"]["strength"] == "strong"
    assert kwargs["metadata"]["alternatives"] == ["light mode", "system default"]


def test_store_preference_rejects_invalid_strength():
    from ogham.tools.memory import store_preference

    with pytest.raises(ValueError, match="strength must be one of"):
        store_preference(preference="strong coffee with oat milk", strength="fanatical")


def test_store_preference_preserves_extra_tags():
    from ogham.tools.memory import store_preference

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}
        store_preference(
            preference="PostgreSQL",
            tags=["project:ogham", "db"],
        )

    tags = mock_store.call_args.kwargs["tags"]
    assert "project:ogham" in tags
    assert "db" in tags
    assert "type:preference" in tags


# --- store_fact ---


def test_store_fact_basic():
    from ogham.tools.memory import store_fact

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}

        store_fact(
            fact="BGE-M3 has 568M parameters",
            subject="BGE-M3",
            confidence=0.95,
            source_citation="HuggingFace model card",
        )

    kwargs = mock_store.call_args.kwargs
    assert "Fact: BGE-M3 has 568M parameters" in kwargs["content"]
    assert "Subject: BGE-M3" in kwargs["content"]
    assert "Source: HuggingFace model card" in kwargs["content"]

    assert "type:fact" in kwargs["tags"]
    assert kwargs["metadata"]["type"] == "fact"
    assert kwargs["metadata"]["confidence"] == 0.95
    assert kwargs["metadata"]["source_citation"] == "HuggingFace model card"


def test_store_fact_rejects_invalid_confidence():
    from ogham.tools.memory import store_fact

    with pytest.raises(ValueError, match="confidence must be between"):
        store_fact(fact="something important about the codebase", confidence=1.5)

    with pytest.raises(ValueError, match="confidence must be between"):
        store_fact(fact="something important about the codebase", confidence=-0.1)


def test_store_fact_defaults_confidence_to_one():
    from ogham.tools.memory import store_fact

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}
        store_fact(fact="The Earth orbits the Sun")

    assert mock_store.call_args.kwargs["metadata"]["confidence"] == 1.0


# --- store_event ---


def test_store_event_basic():
    from ogham.tools.memory import store_event

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}

        store_event(
            event="Acme internal demo",
            when="2026-04-21 15:00",
            participants=["Kevin", "Iain", "Cemre"],
            location="Munich office",
        )

    kwargs = mock_store.call_args.kwargs
    assert "Event: Acme internal demo" in kwargs["content"]
    assert "When: 2026-04-21 15:00" in kwargs["content"]
    assert "Participants: Kevin, Iain, Cemre" in kwargs["content"]
    assert "Location: Munich office" in kwargs["content"]

    assert "type:event" in kwargs["tags"]
    assert kwargs["metadata"]["type"] == "event"
    assert kwargs["metadata"]["when"] == "2026-04-21 15:00"
    assert kwargs["metadata"]["participants"] == ["Kevin", "Iain", "Cemre"]
    assert kwargs["metadata"]["location"] == "Munich office"


def test_store_event_extracts_dates_into_metadata():
    """Dates in the content should land in metadata.dates via extract_dates."""
    from ogham.tools.memory import store_event

    with (
        patch("ogham.tools.memory.store_memory") as mock_store,
        patch(
            "ogham.tools.memory.extract_dates",
            return_value=["2026-04-21"],
        ),
    ):
        mock_store.return_value = {"id": "mem-1", "status": "stored"}
        store_event(event="Acme Q2 pitch demo", when="2026-04-21")

    assert mock_store.call_args.kwargs["metadata"]["dates"] == ["2026-04-21"]


def test_store_event_works_with_minimal_args():
    from ogham.tools.memory import store_event

    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}
        store_event(event="Server deployment")

    content = mock_store.call_args.kwargs["content"]
    # Just "Event: ..." on a single line when optional args are absent
    assert content == "Event: Server deployment"
    assert "type:event" in mock_store.call_args.kwargs["tags"]


# --- All wrappers preserve tags + call through to store_memory ---


@pytest.mark.parametrize(
    "wrapper_name,args",
    [
        ("store_preference", {"preference": "strong coffee with oat milk"}),
        ("store_fact", {"fact": "water boils at 100 celsius at sea level"}),
        ("store_event", {"event": "team stand-up meeting"}),
    ],
)
def test_wrapper_calls_store_memory(wrapper_name, args):
    """Each wrapper is a thin conveniennce over store_memory. Verify it dispatches."""
    from ogham.tools import memory as memory_module

    wrapper = getattr(memory_module, wrapper_name)
    with patch("ogham.tools.memory.store_memory") as mock_store:
        mock_store.return_value = {"id": "mem-1", "status": "stored"}
        wrapper(**args)

    assert mock_store.called
    assert "content" in mock_store.call_args.kwargs
    assert "metadata" in mock_store.call_args.kwargs
    assert "tags" in mock_store.call_args.kwargs
