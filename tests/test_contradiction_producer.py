"""Tests for the contradiction producer (v0.10.1 item 1.3).

Negation-marker polarity detection + contradicts-edge creation wiring
inside store_memory_enriched.
"""

from unittest.mock import patch

import pytest

FAKE_EMBEDDING = [0.1] * 1024


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


# --- Polarity detection unit tests ---


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("I use Python for data science", 1),
        ("I no longer use Python", -1),
        ("We deprecated the old API last month", -1),
        ("I was at the store yesterday", 1),
        ("Changed to PostgreSQL from MySQL", -1),
        ("Replaced by the new service", -1),
        ("This is my favourite memory", 1),
        ("The meeting is cancelled", -1),
        ("Previously we tried option A", -1),
        ("", 1),
        ("Deployed the API successfully", 1),
    ],
)
def test_detect_polarity_english(content, expected):
    """English negation markers flip polarity; positive content stays positive."""
    from ogham.extraction import detect_negation_polarity

    assert detect_negation_polarity(content) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Ich benutze nicht mehr Python", -1),  # German "nicht mehr"
        ("Ya no uso Python", -1),  # Spanish "ya no"
        ("non più uso Python", -1),  # Italian "non più" (contiguous)
        ("Больше не использую Python", -1),  # Russian "больше не"
        ("不再使用 Python", -1),  # Chinese "不再"
        ("Python artık değil kullanılıyor", -1),  # Turkish "artık değil" (contiguous)
        ("Python 대체됨 Supabase", -1),  # Korean "대체됨"
        ("Python ist veraltet", -1),  # German "veraltet"
    ],
)
def test_detect_polarity_multilingual(content, expected):
    """Negation markers from non-English language YAMLs also trigger.

    Note: markers match as contiguous substrings. Languages with split
    constructions (Italian ne... più, Turkish artık... değil, Irish ní... níos
    mó) need the marker words adjacent or the sentence phrased differently.
    This is a documented limitation; more sophisticated grammar-aware
    detection would require per-language tokenisers and is out of scope for
    the v0.10.1 producer.
    """
    from ogham.extraction import detect_negation_polarity

    assert detect_negation_polarity(content) == expected


# --- Service integration: contradicts-edge creation on store ---


def test_store_creates_contradicts_edge_on_opposite_polarity():
    """Opposite-polarity conflict (>75% similar, one negated) -> contradicts edge."""
    from ogham.service import store_memory_enriched

    existing = {
        "id": "mem-old",
        "content": "I use PostgreSQL for production databases",
        "similarity": 0.88,
    }

    with (
        patch(
            "ogham.service.generate_embedding",
            return_value=FAKE_EMBEDDING,
        ),
        patch("ogham.service.hybrid_search_memories", return_value=[existing]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.create_relationship") as create_rel,
        patch("ogham.service.emit_audit_event"),
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=["entity:PostgreSQL"]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        response = store_memory_enriched(
            content="I no longer use PostgreSQL, replaced by Supabase",
            profile="default",
            source="test",
        )

    assert "contradicts" in response
    assert response["contradicts"][0]["id"] == "mem-old"
    create_rel.assert_called_once()
    kwargs = create_rel.call_args.kwargs
    assert kwargs["source_id"] == "mem-new"
    assert kwargs["target_id"] == "mem-old"
    assert kwargs["relationship"] == "contradicts"
    assert kwargs["metadata"]["reason"] == "opposite_polarity"
    assert kwargs["metadata"]["new_polarity"] == -1
    assert kwargs["metadata"]["existing_polarity"] == 1


def test_store_no_contradicts_when_both_positive():
    """Similar memories with same polarity -> no contradicts edge."""
    from ogham.service import store_memory_enriched

    existing = {
        "id": "mem-old",
        "content": "I use PostgreSQL for production databases",
        "similarity": 0.88,
    }

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.hybrid_search_memories", return_value=[existing]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.create_relationship") as create_rel,
        patch("ogham.service.emit_audit_event"),
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=["entity:PostgreSQL"]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        response = store_memory_enriched(
            content="I also use PostgreSQL heavily in staging",
            profile="default",
            source="test",
        )

    assert "contradicts" not in response
    create_rel.assert_not_called()


def test_store_no_contradicts_when_no_conflict():
    """No similar memories in conflict range -> no contradicts edge even with negation."""
    from ogham.service import store_memory_enriched

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.hybrid_search_memories", return_value=[]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.create_relationship") as create_rel,
        patch("ogham.service.emit_audit_event"),
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        response = store_memory_enriched(
            content="I no longer use something entirely new and unrelated",
            profile="default",
            source="test",
        )

    assert "contradicts" not in response
    create_rel.assert_not_called()


def test_store_emits_contradict_audit_event():
    """When contradicts edge created, an audit event with operation=contradict_detected fires."""
    from ogham.service import store_memory_enriched

    existing = {
        "id": "mem-old",
        "content": "We use voyage-3 as our embedding model",
        "similarity": 0.91,
    }

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.hybrid_search_memories", return_value=[existing]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.create_relationship"),
        patch("ogham.service.emit_audit_event") as audit,
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=["entity:voyage-3"]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        store_memory_enriched(
            content="Switched to voyage-3-large, voyage-3 deprecated",
            profile="default",
            source="test",
        )

    operations = [call.kwargs.get("operation") for call in audit.call_args_list]
    assert "contradict_detected" in operations
    assert "store" in operations


def test_store_graceful_when_create_relationship_fails():
    """A create_relationship failure should not prevent the store from succeeding."""
    from ogham.service import store_memory_enriched

    existing = {
        "id": "mem-old",
        "content": "Running Redis as cache",
        "similarity": 0.82,
    }

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.hybrid_search_memories", return_value=[existing]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch(
            "ogham.service.create_relationship",
            side_effect=RuntimeError("db down"),
        ),
        patch("ogham.service.emit_audit_event"),
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        response = store_memory_enriched(
            content="Redis no longer used, replaced by Valkey",
            profile="default",
            source="test",
        )

    # Store succeeds even if contradicts-edge creation fails
    assert response["status"] == "stored"
    assert "contradicts" not in response


def _store_against(existing_content: str, new_content: str):
    """Store new_content with one >75%-similar existing memory; return (response, create_rel)."""
    from ogham.service import store_memory_enriched

    existing = {"id": "mem-old", "content": existing_content, "similarity": 0.84}
    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.hybrid_search_memories", return_value=[existing]),
        patch("ogham.service.db_get_profile_ttl", return_value=None),
        patch(
            "ogham.service.db_store",
            return_value={"id": "mem-new", "created_at": "2026-01-01T00:00:00Z"},
        ),
        patch("ogham.service.db_auto_link", return_value=0),
        patch("ogham.service.create_relationship") as create_rel,
        patch("ogham.service.emit_audit_event"),
        patch("ogham.service.extract_dates", return_value=[]),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_recurrence", return_value=[]),
        patch("ogham.service.compute_importance", return_value=0.7),
        patch("ogham.hooks._mask_secrets", side_effect=lambda t: t),
    ):
        response = store_memory_enriched(content=new_content, profile="default", source="test")
    return response, create_rel


def test_polarity_reads_the_whole_existing_memory_not_a_preview():
    """TBU-304: polarity must be compared on the same population on both sides.

    The existing memory's negation marker sits after character 200. Reading only
    a 200-char preview made it look positive, so a long new memory that also
    contains a marker was recorded as contradicting it -- and since v0.20 a
    contradicts edge hides the target from recall.
    """
    head = "Checkpoint: shipped the CLI release, merged the queue, and tagged it. " * 3
    assert 200 < len(head) < 260
    existing_content = head + "Drain is no longer synchronous."
    assert "no longer" not in existing_content[:200]
    assert len(existing_content) <= 300  # inside the length gate, so polarity decides

    response, create_rel = _store_against(
        existing_content,
        "Handoff note: the outbox is no longer drained synchronously.",
    )

    assert "contradicts" not in response
    create_rel.assert_not_called()


def test_short_true_contradiction_still_detected():
    """The fix must not blind the producer to a real short supersession."""
    response, create_rel = _store_against(
        "We deploy the API on Railway",
        "We no longer deploy the API on Railway, replaced by Fly",
    )

    assert response["contradicts"][0]["id"] == "mem-old"
    create_rel.assert_called_once()


def test_long_memories_never_get_an_auto_contradicts_edge():
    """TBU-304: polarity on multi-paragraph notes is noise, so no edge -- but the
    similar memory is still reported in `conflicts`."""
    filler = " Notes on the release, the queue and the follow-up work." * 6
    response, create_rel = _store_against(
        "We deploy the API on Railway." + filler,
        "We no longer deploy the API on Railway, replaced by Fly." + filler,
    )

    assert "contradicts" not in response
    assert response["conflicts"][0]["id"] == "mem-old"
    create_rel.assert_not_called()
