"""Integration tests against real Supabase + Ollama.

Run with: uv run pytest tests/test_integration.py -v
Skip with: uv run pytest -m 'not integration'

Uses a dedicated '_test_integration' profile to avoid polluting real data.
All test memories are cleaned up in the teardown fixture.
"""

import pytest

TEST_PROFILE = "_test_integration"


def _can_connect() -> bool:
    """Check if Supabase and Ollama are reachable."""
    try:
        from ogham.database import get_client

        get_client().table("memories").select("id").limit(1).execute()

        from ogham.embeddings import generate_embedding

        generate_embedding("connection test")
        return True
    except Exception:
        return False


# Skip entire module if services are unreachable
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _can_connect(), reason="Supabase or Ollama not reachable"),
]


@pytest.fixture(autouse=True)
def cleanup_test_profile():
    """Delete all memories in the test profile after each test."""
    yield
    from ogham.database import get_client

    get_client().table("memories").delete().eq("profile", TEST_PROFILE).execute()


# ------------------------------------------------------------------
# Store & retrieve
# ------------------------------------------------------------------


def test_store_and_search():
    """Store a memory, then find it via semantic search and hybrid search."""
    from ogham.database import hybrid_search_memories, search_memories, store_memory
    from ogham.embeddings import generate_embedding

    content = "The deployment pipeline uses Terragrunt plan on PR, apply on merge to main"
    embedding = generate_embedding(content)

    stored = store_memory(
        content=content,
        embedding=embedding,
        profile=TEST_PROFILE,
        source="integration-test",
        tags=["type:decision", "project:infra"],
    )
    assert stored["id"]
    assert stored["profile"] == TEST_PROFILE

    query_embedding = generate_embedding("how does the deployment pipeline work")

    # Pure vector search (backward compat)
    results = search_memories(
        query_embedding=query_embedding,
        profile=TEST_PROFILE,
        threshold=0.5,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0]["content"] == content

    # Hybrid search
    hybrid_results = hybrid_search_memories(
        query_text="deployment pipeline Terragrunt",
        query_embedding=query_embedding,
        profile=TEST_PROFILE,
        limit=5,
    )
    assert len(hybrid_results) >= 1
    assert hybrid_results[0]["content"] == content
    assert "keyword_rank" in hybrid_results[0]


def test_store_and_list_recent():
    """Store memories, then list them in reverse chronological order."""
    from ogham.database import list_recent_memories, store_memory
    from ogham.embeddings import generate_embedding

    contents = [
        "First memory for listing test",
        "Second memory for listing test",
        "Third memory for listing test",
    ]
    for c in contents:
        store_memory(
            content=c,
            embedding=generate_embedding(c),
            profile=TEST_PROFILE,
            source="integration-test",
        )

    results = list_recent_memories(profile=TEST_PROFILE, limit=10)
    assert len(results) == 3
    # Most recent first
    assert results[0]["content"] == contents[2]
    assert results[2]["content"] == contents[0]
    # New columns present
    assert "access_count" in results[0]
    assert results[0]["access_count"] == 0


# ------------------------------------------------------------------
# Access tracking & ACT-R scoring
# ------------------------------------------------------------------


def test_record_access_increments_count():
    """record_access should bump access_count and set last_accessed_at."""
    from ogham.database import get_client, record_access, store_memory
    from ogham.embeddings import generate_embedding

    content = "Memory for access tracking test"
    embedding = generate_embedding(content)
    stored = store_memory(
        content=content,
        embedding=embedding,
        profile=TEST_PROFILE,
    )
    memory_id = stored["id"]

    # Initially access_count=0, last_accessed_at=None
    row = (
        get_client()
        .table("memories")
        .select("access_count, last_accessed_at")
        .eq("id", memory_id)
        .execute()
    ).data[0]
    assert row["access_count"] == 0
    assert row["last_accessed_at"] is None

    # Record access twice
    record_access([memory_id])
    record_access([memory_id])

    row = (
        get_client()
        .table("memories")
        .select("access_count, last_accessed_at")
        .eq("id", memory_id)
        .execute()
    ).data[0]
    assert row["access_count"] == 2
    assert row["last_accessed_at"] is not None


def test_actr_scoring_boosts_accessed_memories():
    """Frequently accessed memories should get higher relevance than untouched ones."""
    from ogham.database import record_access, search_memories, store_memory
    from ogham.embeddings import generate_embedding

    # Store two nearly identical memories
    content_a = "AWS naming convention is org-env-service"
    content_b = "AWS naming pattern follows org-env-service format"
    emb_a = generate_embedding(content_a)
    emb_b = generate_embedding(content_b)

    stored_a = store_memory(
        content=content_a, embedding=emb_a, profile=TEST_PROFILE, tags=["boosted"]
    )
    store_memory(
        content=content_b, embedding=emb_b, profile=TEST_PROFILE, tags=["not-boosted"]
    )

    # Boost memory A with repeated access
    for _ in range(10):
        record_access([stored_a["id"]])

    # Search — both should match, but A should rank higher by relevance
    query_emb = generate_embedding("AWS naming convention")
    results = search_memories(
        query_embedding=query_emb,
        profile=TEST_PROFILE,
        threshold=0.5,
        limit=10,
    )

    assert len(results) >= 2
    # Results ordered by relevance (desc), so the boosted one should be first
    assert results[0]["content"] == content_a
    assert results[0]["relevance"] > results[1]["relevance"]
    assert results[0]["access_count"] == 10


def test_graph_centrality_boosts_search_ranking():
    """Memories with more graph relationships should rank higher than isolated ones."""
    from ogham.database import get_client, search_memories, store_memory
    from ogham.embeddings import generate_embedding

    # Store three memories: A (will be linked), B (isolated), C (link target)
    content_a = "Kubernetes pod scaling policy is HPA with 70% CPU target"
    content_b = "Kubernetes pod scaling uses HPA with CPU target of 70%"
    content_c = "Container orchestration uses horizontal pod autoscaler"
    emb_a = generate_embedding(content_a)
    emb_b = generate_embedding(content_b)
    emb_c = generate_embedding(content_c)

    stored_a = store_memory(
        content=content_a, embedding=emb_a, profile=TEST_PROFILE, tags=["linked"]
    )
    stored_b = store_memory(
        content=content_b, embedding=emb_b, profile=TEST_PROFILE, tags=["isolated"]
    )
    stored_c = store_memory(
        content=content_c, embedding=emb_c, profile=TEST_PROFILE, tags=["target"]
    )

    # Create relationships for memory A (both directions to boost centrality)
    client = get_client()
    for target_id in [stored_b["id"], stored_c["id"]]:
        client.table("memory_relationships").insert(
            {
                "source_id": stored_a["id"],
                "target_id": target_id,
                "relationship": "related",
                "strength": 1.0,
                "created_by": "test",
            }
        ).execute()

    # Search — A and B have near-identical content, but A has graph connections
    query_emb = generate_embedding("kubernetes pod scaling HPA")
    results = search_memories(
        query_embedding=query_emb,
        profile=TEST_PROFILE,
        threshold=0.3,
        limit=10,
    )

    assert len(results) >= 2
    # Memory A (linked) should rank higher than B (isolated)
    result_ids = [r["id"] for r in results]
    assert stored_a["id"] in result_ids
    assert stored_b["id"] in result_ids
    idx_a = result_ids.index(stored_a["id"])
    idx_b = result_ids.index(stored_b["id"])
    assert idx_a < idx_b, "Linked memory should rank higher than isolated one"
    assert results[idx_a]["relevance"] > results[idx_b]["relevance"]


# ------------------------------------------------------------------
# Tag and source filtering
# ------------------------------------------------------------------


def test_search_filters_by_tags():
    """Search should filter results by tag overlap."""
    from ogham.database import search_memories, store_memory
    from ogham.embeddings import generate_embedding

    store_memory(
        content="Tagged memory about databases",
        embedding=generate_embedding("Tagged memory about databases"),
        profile=TEST_PROFILE,
        tags=["topic:database"],
    )
    store_memory(
        content="Tagged memory about networking",
        embedding=generate_embedding("Tagged memory about networking"),
        profile=TEST_PROFILE,
        tags=["topic:network"],
    )

    query_emb = generate_embedding("database or networking")
    results = search_memories(
        query_embedding=query_emb,
        profile=TEST_PROFILE,
        threshold=0.3,
        tags=["topic:database"],
    )

    assert all("topic:database" in r["tags"] for r in results)


def test_search_filters_by_source():
    """Search should filter results by source."""
    from ogham.database import search_memories, store_memory
    from ogham.embeddings import generate_embedding

    store_memory(
        content="Memory from cursor about testing",
        embedding=generate_embedding("Memory from cursor about testing"),
        profile=TEST_PROFILE,
        source="cursor",
    )
    store_memory(
        content="Memory from claude-code about testing",
        embedding=generate_embedding("Memory from claude-code about testing"),
        profile=TEST_PROFILE,
        source="claude-code",
    )

    query_emb = generate_embedding("testing")
    results = search_memories(
        query_embedding=query_emb,
        profile=TEST_PROFILE,
        threshold=0.3,
        source="cursor",
    )

    assert all(r["source"] == "cursor" for r in results)


# ------------------------------------------------------------------
# Update & delete
# ------------------------------------------------------------------


def test_update_memory():
    """Update a memory's content and verify re-embedding works."""
    from ogham.database import search_memories, store_memory, update_memory
    from ogham.embeddings import generate_embedding

    original = "The API uses REST"
    stored = store_memory(
        content=original,
        embedding=generate_embedding(original),
        profile=TEST_PROFILE,
    )

    updated_content = "The API uses GraphQL, not REST"
    updated = update_memory(
        memory_id=stored["id"],
        updates={
            "content": updated_content,
            "embedding": str(generate_embedding(updated_content)),
        },
        profile=TEST_PROFILE,
    )
    assert updated["content"] == updated_content

    # Search for the updated content
    results = search_memories(
        query_embedding=generate_embedding("GraphQL API"),
        profile=TEST_PROFILE,
        threshold=0.3,
    )
    assert any(r["content"] == updated_content for r in results)


def test_delete_memory():
    """Delete a memory and verify it's gone."""
    from ogham.database import delete_memory, list_recent_memories, store_memory
    from ogham.embeddings import generate_embedding

    stored = store_memory(
        content="Memory to delete",
        embedding=generate_embedding("Memory to delete"),
        profile=TEST_PROFILE,
    )

    deleted = delete_memory(stored["id"], profile=TEST_PROFILE)
    assert deleted is True

    remaining = list_recent_memories(profile=TEST_PROFILE)
    assert all(r["id"] != stored["id"] for r in remaining)


def test_delete_wrong_profile():
    """Delete should fail when profile doesn't match."""
    from ogham.database import delete_memory, store_memory
    from ogham.embeddings import generate_embedding

    stored = store_memory(
        content="Profile-isolated memory",
        embedding=generate_embedding("Profile-isolated memory"),
        profile=TEST_PROFILE,
    )

    deleted = delete_memory(stored["id"], profile="nonexistent-profile")
    assert deleted is False


# ------------------------------------------------------------------
# Profile isolation
# ------------------------------------------------------------------


def test_profile_isolation():
    """Memories in one profile should not appear in another."""
    from ogham.database import get_client, search_memories, store_memory
    from ogham.embeddings import generate_embedding

    other_profile = "_test_integration_other"
    content = "This memory is profile-isolated"
    embedding = generate_embedding(content)

    try:
        store_memory(content=content, embedding=embedding, profile=TEST_PROFILE)
        store_memory(
            content="Different profile memory",
            embedding=generate_embedding("Different profile memory"),
            profile=other_profile,
        )

        # Search in test profile should not return the other profile's memory
        results = search_memories(
            query_embedding=embedding, profile=TEST_PROFILE, threshold=0.3
        )
        assert all(r["profile"] == TEST_PROFILE for r in results)

        # Search in other profile should not return test profile's memory
        results = search_memories(
            query_embedding=embedding, profile=other_profile, threshold=0.3
        )
        assert all(r["profile"] == other_profile for r in results)
    finally:
        get_client().table("memories").delete().eq("profile", other_profile).execute()


# ------------------------------------------------------------------
# Stats & profiles RPCs
# ------------------------------------------------------------------


def test_get_memory_stats():
    """get_memory_stats should return correct counts."""
    from ogham.database import get_memory_stats, store_memory
    from ogham.embeddings import generate_embedding

    store_memory(
        content="Stats test memory one",
        embedding=generate_embedding("Stats test memory one"),
        profile=TEST_PROFILE,
        source="test-source",
        tags=["tag:alpha", "tag:beta"],
    )
    store_memory(
        content="Stats test memory two",
        embedding=generate_embedding("Stats test memory two"),
        profile=TEST_PROFILE,
        source="test-source",
        tags=["tag:alpha"],
    )

    stats = get_memory_stats(TEST_PROFILE)
    assert stats["total"] == 2
    assert stats["sources"]["test-source"] == 2
    # tag:alpha appears in both
    alpha_tag = next(t for t in stats["top_tags"] if t["tag"] == "tag:alpha")
    assert alpha_tag["count"] == 2


def test_list_profiles():
    """list_profiles should include the test profile."""
    from ogham.database import list_profiles, store_memory
    from ogham.embeddings import generate_embedding

    store_memory(
        content="Profile listing test",
        embedding=generate_embedding("Profile listing test"),
        profile=TEST_PROFILE,
    )

    profiles = list_profiles()
    profile_names = [p["profile"] for p in profiles]
    assert TEST_PROFILE in profile_names

    test_entry = next(p for p in profiles if p["profile"] == TEST_PROFILE)
    assert test_entry["count"] >= 1


# ------------------------------------------------------------------
# Expiration
# ------------------------------------------------------------------


def test_expired_memories_hidden_from_search():
    """Expired memories should not appear in search or list results."""
    from datetime import datetime, timedelta, timezone

    from ogham.database import (
        list_recent_memories,
        search_memories,
        store_memory,
    )
    from ogham.embeddings import generate_embedding

    content = "This memory will expire immediately"
    embedding = generate_embedding(content)

    # Expire 1 second in the past
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store_memory(
        content=content,
        embedding=embedding,
        profile=TEST_PROFILE,
        expires_at=expired_at,
    )

    results = search_memories(
        query_embedding=embedding, profile=TEST_PROFILE, threshold=0.3
    )
    assert all(r["content"] != content for r in results)

    recent = list_recent_memories(profile=TEST_PROFILE)
    assert all(r["content"] != content for r in recent)


def test_cleanup_expired():
    """cleanup_expired should delete expired memories."""
    from datetime import datetime, timedelta, timezone

    from ogham.database import cleanup_expired, count_expired, store_memory
    from ogham.embeddings import generate_embedding

    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store_memory(
        content="Expired memory for cleanup test",
        embedding=generate_embedding("Expired memory for cleanup test"),
        profile=TEST_PROFILE,
        expires_at=expired_at,
    )

    count = count_expired(TEST_PROFILE)
    assert count >= 1

    deleted = cleanup_expired(TEST_PROFILE)
    assert deleted >= 1

    count_after = count_expired(TEST_PROFILE)
    assert count_after == 0


# ------------------------------------------------------------------
# Confidence scoring
# ------------------------------------------------------------------


def test_reinforce_increases_confidence():
    """Reinforcing a memory should increase its confidence."""
    from ogham.database import get_client, store_memory, update_confidence
    from ogham.embeddings import generate_embedding

    stored = store_memory(
        content="Reinforcement test memory",
        embedding=generate_embedding("Reinforcement test memory"),
        profile=TEST_PROFILE,
    )

    # Default confidence is 0.5
    row = (
        get_client()
        .table("memories")
        .select("confidence")
        .eq("id", stored["id"])
        .execute()
    ).data[0]
    assert row["confidence"] == 0.5

    # Reinforce
    new_conf = update_confidence(stored["id"], 0.85, TEST_PROFILE)
    assert new_conf > 0.5

    # Reinforce again — should go higher
    new_conf2 = update_confidence(stored["id"], 0.85, TEST_PROFILE)
    assert new_conf2 > new_conf


def test_contradict_decreases_confidence():
    """Contradicting a memory should decrease its confidence."""
    from ogham.database import store_memory, update_confidence
    from ogham.embeddings import generate_embedding

    stored = store_memory(
        content="Contradiction test memory",
        embedding=generate_embedding("Contradiction test memory"),
        profile=TEST_PROFILE,
    )

    new_conf = update_confidence(stored["id"], 0.15, TEST_PROFILE)
    assert new_conf < 0.5


def test_confidence_affects_search_ranking():
    """High-confidence memories should rank above low-confidence ones."""
    from ogham.database import search_memories, store_memory, update_confidence
    from ogham.embeddings import generate_embedding

    content_high = "Database backups run at 3am UTC"
    content_low = "Database backups run at 3am UTC daily"
    emb_high = generate_embedding(content_high)
    emb_low = generate_embedding(content_low)

    stored_high = store_memory(
        content=content_high, embedding=emb_high, profile=TEST_PROFILE
    )
    stored_low = store_memory(
        content=content_low, embedding=emb_low, profile=TEST_PROFILE
    )

    # Boost one, lower the other
    for _ in range(3):
        update_confidence(stored_high["id"], 0.85, TEST_PROFILE)
        update_confidence(stored_low["id"], 0.15, TEST_PROFILE)

    results = search_memories(
        query_embedding=generate_embedding("database backup schedule"),
        profile=TEST_PROFILE,
        threshold=0.3,
    )

    assert len(results) >= 2
    assert results[0]["id"] == stored_high["id"]
    assert results[0]["confidence"] > results[1]["confidence"]


# ------------------------------------------------------------------
# Hybrid search
# ------------------------------------------------------------------


def test_hybrid_search_finds_exact_keywords():
    """Hybrid search should find exact keyword matches that pure vector search might miss."""
    from ogham.database import hybrid_search_memories, store_memory
    from ogham.embeddings import generate_embedding

    content = "The production database connection string uses SUPABASE_KEY=sbp_abc123def456"
    embedding = generate_embedding(content)
    store_memory(
        content=content,
        embedding=embedding,
        profile=TEST_PROFILE,
        source="integration-test",
        tags=["type:config"],
    )

    query = "SUPABASE_KEY"
    query_embedding = generate_embedding(query)
    results = hybrid_search_memories(
        query_text=query,
        query_embedding=query_embedding,
        profile=TEST_PROFILE,
        limit=5,
    )

    assert len(results) >= 1
    assert "SUPABASE_KEY" in results[0]["content"]
    assert results[0]["keyword_rank"] > 0.0


def test_hybrid_search_boosts_dual_matches():
    """Memories matching both semantically and by keyword should rank highest."""
    from ogham.database import hybrid_search_memories, store_memory
    from ogham.embeddings import generate_embedding

    content_a = "Deploy to us-east-1 using the Terragrunt pipeline"
    content_b = "The deployment process uses infrastructure-as-code in AWS"

    emb_a = generate_embedding(content_a)
    emb_b = generate_embedding(content_b)

    store_memory(content=content_a, embedding=emb_a, profile=TEST_PROFILE)
    store_memory(content=content_b, embedding=emb_b, profile=TEST_PROFILE)

    query = "us-east-1 deployment"
    query_embedding = generate_embedding(query)
    results = hybrid_search_memories(
        query_text=query,
        query_embedding=query_embedding,
        profile=TEST_PROFILE,
        limit=5,
    )

    assert len(results) >= 2
    assert "us-east-1" in results[0]["content"]


def test_hybrid_search_respects_filters():
    """Hybrid search should respect tag and source filters."""
    from ogham.database import hybrid_search_memories, store_memory
    from ogham.embeddings import generate_embedding

    content = "Redis cache configuration for session storage"
    embedding = generate_embedding(content)

    store_memory(
        content=content,
        embedding=embedding,
        profile=TEST_PROFILE,
        source="cursor",
        tags=["type:config"],
    )
    store_memory(
        content="Redis cache settings for the API layer",
        embedding=generate_embedding("Redis cache settings for the API layer"),
        profile=TEST_PROFILE,
        source="claude-code",
        tags=["type:decision"],
    )

    results = hybrid_search_memories(
        query_text="Redis cache",
        query_embedding=embedding,
        profile=TEST_PROFILE,
        source="cursor",
        limit=5,
    )

    assert all(r["source"] == "cursor" for r in results)


# ------------------------------------------------------------------
# Batch import
# ------------------------------------------------------------------


def test_batch_import():
    """Batch import should store multiple memories and dedup correctly."""
    import json

    from ogham.export_import import import_memories

    memories = [
        {
            "content": (
                f"Integration test batch memory {i}: "
                f"unique content about topic {i} that is long enough to be meaningful"
            ),
            "source": "test",
            "tags": ["batch-test"],
            "metadata": {},
        }
        for i in range(5)
    ]
    export_data = json.dumps({"memories": memories})

    # First import: all 5 should be stored
    result = import_memories(export_data, TEST_PROFILE, dedup_threshold=0.0)
    assert result["imported"] == 5
    assert result["skipped"] == 0

    # Second import with dedup: all 5 should be skipped
    result = import_memories(export_data, TEST_PROFILE, dedup_threshold=0.8)
    assert result["skipped"] == 5
    assert result["imported"] == 0


def test_batch_check_duplicates():
    """batch_check_duplicates should find existing memories and reject non-matches."""
    from ogham.database import batch_check_duplicates, store_memory
    from ogham.embeddings import generate_embedding

    # Store a memory to check against
    embedding = generate_embedding("integration test batch dedup")
    store_memory(
        content="integration test batch dedup",
        embedding=embedding,
        profile=TEST_PROFILE,
        source="test",
    )

    # Check: the same embedding should be a dup, a dissimilar one should not
    unrelated_embedding = generate_embedding(
        "quantum physics entanglement superposition wavelength"
    )
    results = batch_check_duplicates(
        query_embeddings=[embedding, unrelated_embedding],
        profile=TEST_PROFILE,
        threshold=0.8,
    )

    assert len(results) == 2
    assert results[0] is True   # same embedding = duplicate
    assert results[1] is False  # unrelated embedding = not a match


# ------------------------------------------------------------------
# Memory relationships & knowledge graph
# ------------------------------------------------------------------


def test_auto_link_creates_edges():
    """auto_link_memory should create 'similar' edges between similar memories."""
    from ogham.database import auto_link_memory, get_related_memories, store_memory
    from ogham.embeddings import generate_embedding

    # Store two similar memories
    content_a = "PostgreSQL uses MVCC for concurrent transaction isolation"
    content_b = "Postgres implements multi-version concurrency control for transaction isolation"
    emb_a = generate_embedding(content_a)
    emb_b = generate_embedding(content_b)

    stored_a = store_memory(
        content=content_a, embedding=emb_a, profile=TEST_PROFILE, source="test"
    )
    stored_b = store_memory(
        content=content_b, embedding=emb_b, profile=TEST_PROFILE, source="test"
    )

    # Auto-link memory B — should find A as similar
    links_created = auto_link_memory(
        memory_id=stored_b["id"],
        embedding=emb_b,
        profile=TEST_PROFILE,
        threshold=0.5,  # lower threshold for test reliability
        max_links=5,
    )
    assert links_created >= 1

    # Verify the edge exists via get_related_memories
    related = get_related_memories(
        memory_id=stored_b["id"],
        depth=1,
        min_strength=0.3,
    )
    assert len(related) >= 1
    related_ids = [r["id"] for r in related]
    assert stored_a["id"] in related_ids
    assert related[0]["relationship"] == "similar"


def test_explore_knowledge_returns_graph():
    """explore_memory_graph should return seeds at depth 0 and linked memories at depth 1."""
    from ogham.database import (
        auto_link_memory,
        explore_memory_graph,
        store_memory,
    )
    from ogham.embeddings import generate_embedding

    # Memory A: will be a seed match
    content_a = "Kubernetes pods restart automatically when health checks fail"
    emb_a = generate_embedding(content_a)
    stored_a = store_memory(
        content=content_a, embedding=emb_a, profile=TEST_PROFILE, source="test"
    )

    # Memory B: similar to A (should be linked)
    content_b = "K8s container health probes trigger automatic pod restarts on failure"
    emb_b = generate_embedding(content_b)
    stored_b = store_memory(
        content=content_b, embedding=emb_b, profile=TEST_PROFILE, source="test"
    )

    # Memory C: unrelated
    content_c = "Python virtualenvs isolate package dependencies per project"
    emb_c = generate_embedding(content_c)
    store_memory(
        content=content_c, embedding=emb_c, profile=TEST_PROFILE, source="test"
    )

    # Create auto-links between A and B (both directions for reliability)
    auto_link_memory(
        memory_id=stored_a["id"],
        embedding=emb_a,
        profile=TEST_PROFILE,
        threshold=0.5,
        max_links=5,
    )
    auto_link_memory(
        memory_id=stored_b["id"],
        embedding=emb_b,
        profile=TEST_PROFILE,
        threshold=0.5,
        max_links=5,
    )

    # Explore: query for Kubernetes health checks (limit=2 to exclude unrelated C)
    query = "Kubernetes health check restart"
    query_emb = generate_embedding(query)
    results = explore_memory_graph(
        query_text=query,
        query_embedding=query_emb,
        profile=TEST_PROFILE,
        limit=2,
        depth=1,
        min_strength=0.3,
    )

    assert len(results) >= 2

    # Should have at least one depth-0 seed and one depth-1 traversed result
    depths = {r["depth"] for r in results}
    assert 0 in depths

    # Both Kubernetes memories should be present, unrelated C should not
    result_contents = [r["content"] for r in results]
    assert content_a in result_contents
    assert content_b in result_contents
    assert content_c not in result_contents


def test_store_decision_creates_supports_edges():
    """store_decision with related_memories should create 'supports' relationship edges."""
    from ogham.database import (
        create_relationship,
        get_related_memories,
        store_memory,
    )
    from ogham.embeddings import generate_embedding

    # Store a context memory first
    context_content = "The API currently handles 500 requests per second under load"
    context_emb = generate_embedding(context_content)
    context_mem = store_memory(
        content=context_content,
        embedding=context_emb,
        profile=TEST_PROFILE,
        source="test",
    )

    # Store a decision memory
    decision_content = (
        "Decision: Scale horizontally with read replicas\n"
        "Rationale: Current single instance hits 500 rps ceiling"
    )
    decision_emb = generate_embedding(decision_content)
    decision_mem = store_memory(
        content=decision_content,
        embedding=decision_emb,
        profile=TEST_PROFILE,
        source="test",
        tags=["type:decision"],
        metadata={"type": "decision", "alternatives": ["vertical scaling", "caching"]},
    )

    # Create the supports edge (simulating what store_decision tool does)
    create_relationship(
        source_id=decision_mem["id"],
        target_id=context_mem["id"],
        relationship="supports",
        strength=1.0,
        created_by="user",
    )

    # Verify: traverse from the decision should find the context memory
    related = get_related_memories(
        memory_id=decision_mem["id"],
        depth=1,
        min_strength=0.5,
        relationship_types=["supports"],
    )
    assert len(related) >= 1
    related_ids = [r["id"] for r in related]
    assert context_mem["id"] in related_ids
    assert related[0]["relationship"] == "supports"
    assert related[0]["edge_strength"] == 1.0

    # Verify: traverse from context memory should also find the decision (bidirectional)
    related_back = get_related_memories(
        memory_id=context_mem["id"],
        depth=1,
        min_strength=0.5,
        relationship_types=["supports"],
    )
    assert len(related_back) >= 1
    back_ids = [r["id"] for r in related_back]
    assert decision_mem["id"] in back_ids
