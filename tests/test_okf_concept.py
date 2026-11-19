from datetime import datetime, timezone

import pytest

from ogham.okf.concept import (
    derive_okf_type,
    frontmatter_to_memory,
    generator_actor,
    memory_to_frontmatter,
    strip_type_tags,
)


def test_derive_okf_type_default_when_no_type_tag():
    assert derive_okf_type([]) == "Memory"
    assert derive_okf_type(["project:ogham", "council-override"]) == "Memory"


def test_derive_okf_type_picks_first_alphabetically_and_titlecases():
    tags = ["project:ogham", "type:gotcha", "type:decision"]
    # Alphabetical: type:decision wins
    assert derive_okf_type(tags) == "Decision"


def test_derive_okf_type_single_type_tag():
    assert derive_okf_type(["type:pattern"]) == "Pattern"


def test_derive_okf_type_handles_hyphenated_types():
    assert derive_okf_type(["type:code-change"]) == "Code-change"


def test_derive_okf_type_ignores_empty_type_tag():
    # 'type:' with no value is malformed; treat as if absent.
    assert derive_okf_type(["type:"]) == "Memory"
    assert derive_okf_type(["type:", "type:decision"]) == "Decision"


def test_strip_type_tags_removes_only_winning_type_tag():
    tags = ["type:decision", "type:gotcha", "project:ogham"]
    # Winner ("type:decision") removed; others remain
    assert strip_type_tags(tags) == ["type:gotcha", "project:ogham"]


def test_strip_type_tags_no_type_tags_returns_unchanged():
    tags = ["project:ogham", "council-override"]
    assert strip_type_tags(tags) == tags


def test_memory_to_frontmatter_required_and_recommended_fields():
    memory = {
        "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
        "content": "Use UUID primary keys",
        "tags": ["project:ogham", "type:decision"],
        "source": "claude-code",
        "created_at": "2026-06-17T08:51:03.613750Z",
        "metadata": {},
    }
    fm = memory_to_frontmatter(memory)
    assert fm["type"] == "Decision"
    assert fm["x-ogham-id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
    assert fm["tags"] == ["project:ogham"]
    assert fm["x-ogham-source"] == "claude-code"
    # OKF v0.3 §5.2: generated {by, at} replaces the v0.1 `timestamp`.
    assert fm["generated"]["at"] == "2026-06-17T08:51:03.613750Z"
    assert fm["generated"]["by"].startswith("ogham/")
    for legacy in ("id", "source", "timestamp"):
        assert legacy not in fm


def test_memory_to_frontmatter_timestamp_is_str_for_datetime_or_iso_string_created_at():
    """TBU-162 audit: the Postgres backend returns created_at as a real
    datetime object; the Supabase REST backend returns an ISO string.
    Uncoerced, yaml.safe_dump would write these differently (unquoted YAML
    timestamp scalar vs quoted string), making the exported OKF bundle
    byte-dependent on which backend produced it. Both shapes must produce
    the same str timestamp."""
    base = {
        "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
        "content": "x",
        "tags": [],
        "source": None,
        "metadata": {},
    }
    iso_str = "2026-06-17T08:51:03.613750+00:00"
    dt = datetime(2026, 6, 17, 8, 51, 3, 613750, tzinfo=timezone.utc)

    fm_from_str = memory_to_frontmatter({**base, "created_at": iso_str})
    fm_from_dt = memory_to_frontmatter({**base, "created_at": dt})

    assert isinstance(fm_from_str["generated"]["at"], str)
    assert isinstance(fm_from_dt["generated"]["at"], str)
    assert fm_from_dt["generated"]["at"] == fm_from_str["generated"]["at"] == iso_str


def test_a_naive_datetime_is_written_with_a_utc_offset():
    """OKF §5 rejects a date-time with no offset. created_at is timestamptz,
    so a naive value can only be UTC that lost its tzinfo."""
    fm = memory_to_frontmatter(
        {"id": "x", "content": "x", "tags": [], "created_at": datetime(2026, 6, 17, 8, 0, 0)}
    )
    assert fm["generated"]["at"] == "2026-06-17T08:00:00+00:00"


def test_memory_to_frontmatter_missing_created_at_is_none():
    memory = {
        "id": "11111111-2222-3333-4444-555555555555",
        "content": "x",
        "tags": [],
        "source": None,
        "metadata": {},
    }
    fm = memory_to_frontmatter(memory)
    # `by` is required in `generated`; `at` is only written when known.
    assert fm["generated"] == {"by": generator_actor()}


def test_memory_to_frontmatter_title_from_first_line():
    memory = {
        "id": "11111111-2222-3333-4444-555555555555",
        "content": "First line is the title\n\nRest is the body.",
        "tags": [],
        "source": None,
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": {},
    }
    fm = memory_to_frontmatter(memory)
    assert fm["title"] == "First line is the title"


def test_memory_to_frontmatter_omits_resource_for_abstract_concepts():
    # Memories without a clear URI resource: field MUST be absent per spec §4.1
    memory = {
        "id": "11111111-2222-3333-4444-555555555555",
        "content": "x",
        "tags": [],
        "source": None,
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": {},
    }
    fm = memory_to_frontmatter(memory)
    assert "resource" not in fm


def test_memory_to_frontmatter_passes_metadata_through_flat():
    # Spec §4.1: unknown keys MAY be added flat. We pass metadata keys directly
    # under top-level frontmatter so consumers can reason about them without
    # knowing our nesting convention.
    memory = {
        "id": "11111111-2222-3333-4444-555555555555",
        "content": "x",
        "tags": [],
        "source": None,
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": {"language": "en", "session_id": "abc123"},
    }
    fm = memory_to_frontmatter(memory)
    assert fm["language"] == "en"
    assert fm["session_id"] == "abc123"


def test_memory_to_frontmatter_omits_optional_fields_when_empty():
    memory = {
        "id": "11111111-2222-3333-4444-555555555555",
        "content": "x",
        "tags": [],
        "source": None,
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": {},
    }
    fm = memory_to_frontmatter(memory)
    assert "x-ogham-source" not in fm
    assert fm["tags"] == []  # empty list is OK


def test_frontmatter_to_memory_reconstructs_type_tag():
    fm = {
        "type": "Decision",
        "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
        "tags": ["project:ogham"],
        "timestamp": "2026-06-17T08:51:03.613750Z",
        "source": "claude-code",
    }
    body = "Use UUID primary keys"
    memory = frontmatter_to_memory(fm, body)
    assert memory["id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
    assert memory["content"] == "Use UUID primary keys"
    # Type tag reconstructed from OKF type (lowercased)
    assert "type:decision" in memory["tags"]
    assert "project:ogham" in memory["tags"]
    assert memory["source"] == "claude-code"


def test_frontmatter_to_memory_default_type_does_not_emit_type_tag():
    # OKF type "Memory" is our default; round-trip MUST NOT emit type:memory
    # tag because that would pollute the user's namespace on every import.
    fm = {
        "type": "Memory",
        "id": "11111111-2222-3333-4444-555555555555",
        "tags": ["project:ogham"],
        "timestamp": "2026-06-17T00:00:00Z",
    }
    memory = frontmatter_to_memory(fm, "x")
    assert "type:memory" not in memory["tags"]
    assert memory["tags"] == ["project:ogham"]


def test_frontmatter_to_memory_preserves_unknown_keys_in_metadata():
    # Spec §4.1: consumers SHOULD preserve unknown keys when round-tripping.
    fm = {
        "type": "Memory",
        "id": "11111111-2222-3333-4444-555555555555",
        "tags": [],
        "timestamp": "2026-06-17T00:00:00Z",
        "custom_field": "custom_value",
        "another_one": 42,
    }
    memory = frontmatter_to_memory(fm, "x")
    assert memory["metadata"]["custom_field"] == "custom_value"
    assert memory["metadata"]["another_one"] == 42


def test_frontmatter_to_memory_missing_id_returns_none_for_id():
    fm = {"type": "Memory", "tags": [], "timestamp": "2026-06-17T00:00:00Z"}
    memory = frontmatter_to_memory(fm, "x")
    assert memory["id"] is None  # caller decides whether to mint a new UUID


def test_frontmatter_to_memory_preserves_spec_recommended_fields_in_metadata():
    fm = {
        "type": "BigQuery Table",
        "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99",
        "title": "Customer Orders",
        "description": "One row per completed order.",
        "resource": "https://console.cloud.google.com/...",
        "tags": ["sales"],
        "timestamp": "2026-05-28T14:30:00Z",
    }
    memory = frontmatter_to_memory(fm, "body")
    assert memory["metadata"]["title"] == "Customer Orders"
    assert memory["metadata"]["description"] == "One row per completed order."
    assert memory["metadata"]["resource"] == "https://console.cloud.google.com/..."
    # Verify round-trip: memory_to_frontmatter re-emits them at top level
    fm_out = memory_to_frontmatter({**memory, "created_at": "2026-05-28T14:30:00Z"})
    assert fm_out["title"] == "Customer Orders"
    assert fm_out["description"] == "One row per completed order."
    assert fm_out["resource"] == "https://console.cloud.google.com/..."


# ── D7: memory_entities -> MENTIONS ───────────────────────────────────────


def test_memory_to_frontmatter_emits_mentions_for_linked_entities():
    """memory_entities becomes MENTIONS -- one of the five verified Schema.org
    alignments, so schema:mentions lands on the memory concepts themselves."""
    memory = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "content": "Ogham ships supersession ranking",
        "tags": ["type:decision"],
        "created_at": "2026-08-03T09:00:00+00:00",
    }
    fm = memory_to_frontmatter(
        memory,
        entity_paths=["entities/ogham-e42", "entities/supabase-e7"],
    )
    # Relative to memories/, per OKF §6 rule 3 -- not v0.1 wiki links.
    assert fm["x-ogham-mentions"] == ["../entities/ogham-e42.md", "../entities/supabase-e7.md"]
    assert "MENTIONS" not in fm


def test_memory_to_frontmatter_omits_mentions_when_there_are_no_entities():
    memory = {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "content": "x", "tags": []}
    assert "x-ogham-mentions" not in memory_to_frontmatter(memory, entity_paths=[])
    assert "x-ogham-mentions" not in memory_to_frontmatter(memory)


def test_metadata_cannot_overwrite_the_mentions_edges():
    """Same guard the spec/extension fields already have: metadata is producer
    data and must not clobber a triple."""
    memory = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "content": "x",
        "tags": [],
        "metadata": {"x-ogham-mentions": "not-a-link"},
    }
    fm = memory_to_frontmatter(memory, entity_paths=["entities/ogham-e42"])
    assert fm["x-ogham-mentions"] == ["../entities/ogham-e42.md"]


def test_entity_paths_is_keyword_only():
    """The default keeps every existing caller working; keyword-only keeps a
    future positional argument from silently landing on it."""
    memory = {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "content": "x", "tags": []}
    with pytest.raises(TypeError):
        memory_to_frontmatter(memory, ["entities/ogham-e42"])  # type: ignore[misc]


def test_stale_mentions_in_metadata_never_reaches_the_bundle():
    """MENTIONS is derived from memory_entities, never carried as producer data.

    This is a real round-trip path, not a hypothetical. frontmatter_to_memory
    files every unrecognised frontmatter key into metadata, so a memory imported
    from a bundle that had MENTIONS carries it in metadata forever. On the next
    export, if that memory has no entity links, the metadata flatten would emit
    the OLD bundle's MENTIONS -- naming entity concepts the new bundle never
    wrote. That is precisely the dangling link the format flags everywhere else.

    The `k not in fm` guard cannot catch it: with no entities, MENTIONS is not
    in fm at all, so metadata wins uncontested.
    """
    memory = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "content": "x",
        "tags": [],
        "metadata": {
            "MENTIONS": "[[entities/ghost-e999]]",
            "x-ogham-mentions": ["../entities/ghost-e999.md"],
        },
    }
    for fm in (memory_to_frontmatter(memory, entity_paths=[]), memory_to_frontmatter(memory)):
        assert "MENTIONS" not in fm
        assert "x-ogham-mentions" not in fm
    # And a real derivation still wins over the stale value.
    fm = memory_to_frontmatter(memory, entity_paths=["entities/ogham-e42"])
    assert fm["x-ogham-mentions"] == ["../entities/ogham-e42.md"]
    assert "MENTIONS" not in fm


def test_frontmatter_id_is_serialisable_when_the_backend_returns_a_uuid():
    """yaml.safe_dump raises RepresenterError on a uuid.UUID, so passing the
    psycopg id straight through made write_concept explode. Sibling of the
    make_filename fix -- same root cause, different consumer."""
    import uuid

    import yaml

    memory_id = uuid.UUID("d2b3e6b9-1996-4e1d-97f9-53bb27a3ffe9")
    fm = memory_to_frontmatter({"id": memory_id, "content": "x", "tags": []})

    assert fm["x-ogham-id"] == str(memory_id)
    yaml.safe_dump(fm)  # must not raise
