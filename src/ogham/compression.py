"""Memory compression -- gradual degradation from full text to gist to tags.

Memories compress over time based on age, access count, and importance.
Original content is always preserved for decompression.
"""

import logging
import re

from ogham.data.loader import get_compression_decision_words

logger = logging.getLogger(__name__)

# Sentence scoring patterns for gist extraction
_FILE_PATH_RE = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception|Traceback)\b")

_DECISION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in get_compression_decision_words("en")) + r")\b",
    re.IGNORECASE,
)
_NUMBER_VERSION_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")


def _score_sentence(sentence: str) -> float:
    """Score a sentence by information density for gist extraction."""
    score = 0.0
    if _FILE_PATH_RE.search(sentence):
        score += 3.0
    if _ERROR_RE.search(sentence):
        score += 4.0
    if _DECISION_RE.search(sentence):
        score += 3.0
    if _NUMBER_VERSION_RE.search(sentence):
        score += 2.0
    if "`" in sentence:
        score += 2.0
    return score


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if s.strip()]


def compress_to_gist(content: str) -> str:
    """Compress content to key sentences (~30% of original).

    Preserves code blocks verbatim. Keeps first and last sentence
    (primacy-recency effect). Scores middle sentences by information
    density and keeps the highest-scoring ones.
    """
    # Extract and preserve code blocks
    code_blocks = _CODE_BLOCK_RE.findall(content)
    text_without_code = _CODE_BLOCK_RE.sub("", content)

    sentences = _split_sentences(text_without_code)
    if not sentences:
        return "\n\n".join(code_blocks) if code_blocks else content

    if len(sentences) <= 3:
        parts = list(sentences)
        if code_blocks:
            parts.extend(code_blocks)
        return "\n".join(parts)

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        score = _score_sentence(sent)
        if i == 0:
            score += 10.0  # primacy
        if i == len(sentences) - 1:
            score += 8.0  # recency
        scored.append((i, sent, score))

    # Target ~30% of original length
    target_length = max(len(content) * 0.3, 50)
    scored.sort(key=lambda x: x[2], reverse=True)

    selected = set()
    current_length = sum(len(cb) for cb in code_blocks)
    for idx, sent, _score in scored:
        if current_length >= target_length:
            break
        selected.add(idx)
        current_length += len(sent)

    # Always include first and last
    selected.add(0)
    selected.add(len(sentences) - 1)

    # Reconstruct in original order
    gist_sentences = [sentences[i] for i in sorted(selected)]
    parts = gist_sentences
    if code_blocks:
        parts.extend(code_blocks)

    return "\n".join(parts)


def compress_to_tags(content: str, tags: list[str] | None = None) -> str:
    """Compress to a one-line summary + tags. Target: < 200 chars."""
    sentences = _split_sentences(content)
    summary = sentences[0] if sentences else content[:80]

    tag_part = ", ".join(tags[:5]) if tags else "general"
    result = f"{summary} | Tags: {tag_part}"

    if len(result) > 200:
        available = 200 - len(f" | Tags: {tag_part}")
        if available > 10:
            summary = summary[: available - 3] + "..."
        result = f"{summary} | Tags: {tag_part}"

    return result[:200]


def get_compression_target(memory: dict) -> int:
    """Determine target compression level based on age and importance.

    Returns: 0 = full, 1 = gist, 2 = tags
    """
    from datetime import datetime, timezone

    created = memory.get("created_at", "")
    if not created:
        return 0

    try:
        if isinstance(created, str):
            created_dt = datetime.fromisoformat(created)
        else:
            created_dt = created
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0

    now = datetime.now(timezone.utc)
    hours = (now - created_dt).total_seconds() / 3600.0

    # Resistance multiplier: important/active memories resist compression
    resistance = 1.0
    if memory.get("importance", 0.5) > 0.7:
        resistance *= 2.0
    if memory.get("confidence", 0.5) > 0.8:
        resistance *= 1.3
    if memory.get("access_count", 0) > 10:
        resistance *= 1.5

    gist_threshold = 168 * resistance  # 7 days * resistance
    tag_threshold = 720 * resistance  # 30 days * resistance

    current = memory.get("compression_level", 0)

    if hours >= tag_threshold and current < 2:
        return 2
    elif hours >= gist_threshold and current < 1:
        return 1
    return current
