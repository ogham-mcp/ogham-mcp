"""MCP prompt templates for common memory workflows."""

from ogham.app import mcp
from ogham.database import (
    count_expired as db_count_expired,
)
from ogham.database import (
    get_memory_stats,
    hybrid_search_memories,
    list_recent_memories,
)
from ogham.embeddings import generate_embedding
from ogham.tools.memory import get_active_profile


@mcp.prompt()
def summarize_recent(limit: int = 10) -> str:
    """Summarize recent memories in the active profile."""
    profile = get_active_profile()
    memories = list_recent_memories(profile=profile, limit=limit)

    if not memories:
        return f"No memories found in profile '{profile}'."

    lines = [f"Here are the {len(memories)} most recent memories in the '{profile}' profile:\n"]
    for i, mem in enumerate(memories, 1):
        tags = ", ".join(mem.get("tags", []))
        lines.append(f"{i}. [{mem['created_at']}] {mem['content']}")
        if tags:
            lines.append(f"   Tags: {tags}")
    lines.append("\nPlease provide a concise summary of the key themes and information.")
    return "\n".join(lines)


@mcp.prompt()
def find_decisions(topic: str) -> str:
    """Find decisions made about a specific topic."""
    profile = get_active_profile()
    query = f"decision about {topic}"
    embedding = generate_embedding(query)
    results = hybrid_search_memories(
        query_text=query,
        query_embedding=embedding,
        profile=profile,
        tags=["type:decision"],
        limit=10,
    )

    if not results:
        return f"No decisions found about '{topic}' in profile '{profile}'."

    lines = [f"Found {len(results)} decisions related to '{topic}':\n"]
    for r in results:
        relevance = r.get('relevance', r.get('similarity', 0))
        lines.append(f"- (relevance: {relevance:.3f}) {r['content']}")
    lines.append("\nPlease summarize these decisions and any patterns you notice.")
    return "\n".join(lines)


@mcp.prompt()
def profile_overview() -> str:
    """Show an overview of the active memory profile."""
    profile = get_active_profile()
    stats = get_memory_stats(profile=profile)
    recent = list_recent_memories(profile=profile, limit=5)

    lines = [f"Profile Overview: '{profile}'\n"]
    lines.append(f"Total memories: {stats.get('total', 0)}")

    sources = stats.get("sources") or {}
    if sources:
        lines.append(
            f"Sources: {', '.join(f'{k}: {v}' for k, v in sources.items())}"
        )

    top_tags = stats.get("top_tags") or []
    if top_tags:
        tag_str = ", ".join(f"{t['tag']} ({t['count']})" for t in top_tags[:5])
        lines.append(f"Top tags: {tag_str}")

    if recent:
        lines.append(f"\nMost recent {len(recent)} memories:")
        for mem in recent:
            lines.append(f"  - {mem['content'][:100]}")

    lines.append("\nPlease provide insights about this profile's memory collection.")
    return "\n".join(lines)


@mcp.prompt()
def cleanup_check() -> str:
    """Preview what expired memory cleanup would do."""
    profile = get_active_profile()
    expired_count = db_count_expired(profile)

    if expired_count == 0:
        return f"No expired memories in profile '{profile}'. Nothing to clean up."

    return (
        f"Profile '{profile}' has {expired_count} expired memories.\n\n"
        f"These memories are already hidden from searches and listings. "
        f"Running cleanup_expired() will permanently delete them.\n\n"
        f"Would you like to proceed with cleanup?"
    )
