"""Importer for Anthropic's Claude.ai conversation export.

Anthropic offers a first-party data export at Settings -> Privacy ->
Request your data. Within ~24-48h you receive a ZIP containing
``conversations.json``, ``users.json``, and (optionally)
``projects.json``. This module turns that ZIP into Ogham memories.

DESIGN: per-turn-pair, assistant-as-content
============================================
A naive importer treats every chat message as one memory. A year of
heavy Claude.ai usage produces 10K+ messages -- 80% of which are
"ok"/"thanks"/"rewrite this" pleasantries that pollute search.

Instead, we walk consecutive ``human -> assistant`` pairs and emit one
memory per pair. The memory ``content`` is the **assistant turn** (the
actual signal), and the **human prompt** lives in
``metadata.user_prompt``. This flips the retrieval profile so
``hybrid_search`` matches on the answer's content while the original
question is still recoverable for context.

Dedup is UUID-based via ``metadata.claude_message_uuid`` -- re-importing
the same export six months later only adds new messages.

GRANULARITY MODES
=================
* ``turn-pairs`` (default) -- one memory per human/assistant pair,
  with smart-filter to drop noise turns.
* ``summarize`` -- stub for v0.15.x. One LLM-distilled memory per
  conversation; richest signal but adds LLM cost.
* ``raw`` -- one memory per message. Mirrors the chrome-extension
  output for completeness; not recommended.

PRIVACY
=======
A year of Claude.ai history can include medical, legal, financial,
relationship content. The default filter is conservative; users
worried about specific topics should pre-prune the export ZIP or wait
for ``--review`` mode (v0.15.x).
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ogham.export_import import import_memories as _import_memories

logger = logging.getLogger(__name__)

# Smart-filter thresholds. Empirically tuned to drop pleasantries while
# keeping anything resembling actual content. Override in CLI if you
# want every "thanks" preserved.
_MIN_HUMAN_CHARS = 10
_MIN_ASSISTANT_CHARS = 50

# Common short pleasantry stems. Lowercase comparison after strip().
_HUMAN_NOISE = frozenset(
    {
        "ok",
        "okay",
        "thanks",
        "thank you",
        "ty",
        "yes",
        "no",
        "great",
        "perfect",
        "got it",
        "sounds good",
        "cool",
        "nice",
        "wow",
        "lol",
        "k",
        "👍",
        "👋",
        "🙏",
    }
)

# Modes accepted by import_claude_ai_export. ``raw`` and ``summarize``
# are placeholders for future granularity choices; the parser produces
# turn-pairs for both today, with a TODO marker on the summarize path.
ImportMode = Literal["turn-pairs", "raw", "summarize"]


def _is_noise(human_text: str, assistant_text: str) -> bool:
    """Return True if the turn-pair looks like a pleasantry exchange.

    Conservative -- we'd rather keep a borderline pair than drop a real
    one. Two AND'd checks: (a) human turn under threshold, (b) human
    turn is a known noise stem. Either alone isn't enough -- a 5-char
    project name in the human turn shouldn't drop a long assistant
    response.
    """
    if len(assistant_text.strip()) < _MIN_ASSISTANT_CHARS:
        return True
    h = human_text.strip().lower().rstrip(".!?")
    if len(h) < _MIN_HUMAN_CHARS and h in _HUMAN_NOISE:
        return True
    return False


def _slugify(text: str, max_len: int = 60) -> str:
    """Produce a tag-safe slug from a conversation title."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:max_len] or "untitled"


def _open_export(path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw conversation dicts from a ZIP, file, or directory.

    Anthropic's export ships as a ZIP whose root contains
    ``conversations.json``. Users sometimes unzip first and point at the
    extracted directory or directly at the JSON file. We accept all three.
    """
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            try:
                with zf.open("conversations.json") as fh:
                    data = json.load(fh)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{path} does not contain conversations.json -- "
                    "check this is an Anthropic data export ZIP"
                ) from exc
        if isinstance(data, list):
            yield from data
        return
    if path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        return
    if path.is_dir():
        target = path / "conversations.json"
        if not target.is_file():
            raise FileNotFoundError(
                f"no conversations.json under {path} -- expected an "
                "Anthropic data export directory or its conversations.json"
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        return
    raise FileNotFoundError(f"not a Claude.ai export: {path}")


def _message_text(msg: dict[str, Any]) -> str:
    """Pull plain text out of a Claude.ai message.

    Older exports use a flat ``text`` field. Newer ones (post structured-
    content rollout) use ``content: [{"type": "text", "text": "..."}]``.
    Concatenate all text blocks for compatibility; non-text blocks
    (tool_use, tool_result) are skipped because they rarely carry
    standalone meaning out of context.
    """
    flat = msg.get("text")
    if isinstance(flat, str) and flat:
        return flat
    blocks = msg.get("content")
    if isinstance(blocks, list):
        parts = [
            b.get("text", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        return "\n\n".join(parts)
    return ""


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_turn_pairs(
    conversation: dict[str, Any], *, smart_filter: bool = True
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Walk ``conversation.chat_messages`` and yield (human, assistant) pairs.

    A pair is two consecutive messages with the human sender first and
    the assistant sender immediately after. Orphan turns (system
    messages, two humans in a row, an assistant with no preceding human)
    are dropped silently -- in practice they're metadata artifacts, not
    user-actionable content.

    With ``smart_filter=True`` (default), pairs that look like
    pleasantries are skipped. Disable for archival imports where every
    word should land.
    """
    messages = conversation.get("chat_messages") or []
    if not isinstance(messages, list):
        return
    pending: dict[str, Any] | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        sender = msg.get("sender")
        if sender == "human":
            pending = msg
            continue
        if sender == "assistant":
            if pending is None:
                continue
            h_text = _message_text(pending)
            a_text = _message_text(msg)
            human, pending = pending, None
            if not h_text or not a_text:
                continue
            if smart_filter and _is_noise(h_text, a_text):
                continue
            yield human, msg


def turn_pair_to_memory(
    human: dict[str, Any],
    assistant: dict[str, Any],
    conversation: dict[str, Any],
    *,
    project_tag: str | None = None,
) -> dict[str, Any]:
    """Convert a (human, assistant) pair into an Ogham memory dict.

    Memory shape:
      content   = assistant turn text (the signal)
      tags      = source:claude-ai
                + claude-conversation:<title-slug>
                + project:<project_tag> (when given)
      metadata  = {
          user_prompt: <human turn text>,
          claude_message_uuid: <assistant uuid>,  # for dedup
          claude_human_uuid:   <human uuid>,
          claude_conversation_uuid: <conversation uuid>,
          claude_conversation_title: <name>,
          claude_created_at: <assistant.created_at iso>,
          turn_index:        <0-based pair number in conversation>,
      }
      source    = claude-ai

    The conversation title is preserved both as a tag (for filtering)
    and in metadata (for display). If the user later renames the
    conversation in Claude.ai and re-exports, the tag tracks the new
    title -- previous memories keep the old slug.
    """
    title = conversation.get("name") or "Untitled"
    title_slug = _slugify(title)
    h_text = _message_text(human)
    a_text = _message_text(assistant)

    tags = ["source:claude-ai", f"claude-conversation:{title_slug}"]
    if project_tag:
        tags.append(f"project:{project_tag}")

    metadata: dict[str, Any] = {
        "user_prompt": h_text,
        "claude_message_uuid": assistant.get("uuid"),
        "claude_human_uuid": human.get("uuid"),
        "claude_conversation_uuid": conversation.get("uuid"),
        "claude_conversation_title": title,
        "claude_created_at": assistant.get("created_at"),
    }
    return {
        "content": a_text,
        "tags": tags,
        "source": "claude-ai",
        "metadata": metadata,
    }


def _passes_since_filter(conv: dict[str, Any], since: datetime | None) -> bool:
    if since is None:
        return True
    updated = _parse_iso(conv.get("updated_at")) or _parse_iso(conv.get("created_at"))
    return updated is None or updated >= since


def parse_export(
    path: Path | str,
    *,
    mode: ImportMode = "turn-pairs",
    smart_filter: bool = True,
    project_tag: str | None = None,
    since: datetime | str | None = None,
) -> list[dict[str, Any]]:
    """Walk an export and return memory dicts ready for import_memories.

    Args:
        path: ZIP, conversations.json, or extracted directory.
        mode: ``turn-pairs`` (default), ``raw``, or ``summarize``.
            ``summarize`` is currently aliased to ``turn-pairs`` until
            the LLM-distillation path lands; mode is preserved in
            metadata so re-runs can upgrade them.
        smart_filter: drop noise pairs (default True). ``raw`` mode
            disables this regardless of the parameter.
        project_tag: explicit override for the ``project:<tag>`` tag.
        since: only include conversations updated on/after this date.
            Accepts datetime or ISO 8601 string.
    """
    from datetime import timezone

    path_obj = Path(path).expanduser()
    since_dt: datetime | None
    if isinstance(since, str):
        since_dt = _parse_iso(since) or _parse_iso(f"{since}T00:00:00+00:00")
    else:
        since_dt = since
    # Coerce naive datetimes to UTC so comparisons with the export's
    # tz-aware timestamps don't trip TypeError. Date-only `since=`
    # strings ("2026-01-01") parse naive; treat them as UTC midnight.
    if since_dt is not None and since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)

    out: list[dict[str, Any]] = []
    use_filter = smart_filter and mode != "raw"

    for idx_in_export, conv in enumerate(_open_export(path_obj)):
        if not _passes_since_filter(conv, since_dt):
            continue
        for turn_idx, (human, assistant) in enumerate(
            extract_turn_pairs(conv, smart_filter=use_filter)
        ):
            mem = turn_pair_to_memory(human, assistant, conv, project_tag=project_tag)
            mem["metadata"]["turn_index"] = turn_idx
            mem["metadata"]["import_mode"] = mode
            out.append(mem)
        # Light progress signal -- a 10K-conversation export is not
        # unusual. Logging at INFO would spam; DEBUG is right for the
        # rare deep-debug session.
        if (idx_in_export + 1) % 100 == 0:
            logger.debug("parsed %d conversations from %s", idx_in_export + 1, path_obj)
    return out


def import_claude_ai_export(
    path: Path | str,
    profile: str,
    *,
    mode: ImportMode = "turn-pairs",
    smart_filter: bool = True,
    project_tag: str | None = None,
    since: datetime | str | None = None,
    dedup_threshold: float = 0.8,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_embed_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Parse a Claude.ai export and import the resulting memories.

    Stitches parse_export -> JSON envelope -> import_memories. Returns
    the import_memories result dict augmented with the source path and
    the parsed-memory count for diagnostic visibility.
    """
    parsed = parse_export(
        path,
        mode=mode,
        smart_filter=smart_filter,
        project_tag=project_tag,
        since=since,
    )

    if not parsed:
        return {
            "status": "complete",
            "profile": profile,
            "imported": 0,
            "skipped": 0,
            "total": 0,
            "path": str(path),
            "warning": (
                "no turn-pairs extracted (empty export, all filtered, "
                "or no conversations after --since)"
            ),
        }

    envelope = json.dumps({"memories": parsed})
    result = _import_memories(
        envelope,
        profile=profile,
        dedup_threshold=dedup_threshold,
        on_progress=on_progress,
        on_embed_progress=on_embed_progress,
    )
    result["path"] = str(path)
    result["mode"] = mode
    return result
