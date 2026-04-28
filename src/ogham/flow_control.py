"""Runtime controls for Ogham recall and inscribe flows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_recall_override: bool | None = None
_inscribe_override: bool | None = None


def set_flow_overrides(
    *,
    recall: bool | None = None,
    inscribe: bool | None = None,
) -> None:
    """Set process-local overrides for CLI/server entrypoints."""
    global _recall_override, _inscribe_override
    if recall is not None:
        _recall_override = recall
    if inscribe is not None:
        _inscribe_override = inscribe


def clear_flow_overrides() -> None:
    """Clear process-local overrides. Used by tests."""
    global _recall_override, _inscribe_override
    _recall_override = None
    _inscribe_override = None


@contextmanager
def temporary_flow_overrides(
    *,
    recall: bool | None = None,
    inscribe: bool | None = None,
) -> Iterator[None]:
    """Apply overrides for one command and restore the previous process state."""
    global _recall_override, _inscribe_override
    old_recall = _recall_override
    old_inscribe = _inscribe_override
    set_flow_overrides(recall=recall, inscribe=inscribe)
    try:
        yield
    finally:
        _recall_override = old_recall
        _inscribe_override = old_inscribe


def recall_enabled() -> bool:
    if _recall_override is not None:
        return _recall_override
    from ogham.config import settings

    return bool(getattr(settings, "recall_enabled", True))


def inscribe_enabled() -> bool:
    if _inscribe_override is not None:
        return _inscribe_override
    from ogham.config import settings

    return bool(getattr(settings, "inscribe_enabled", True))


def disabled_message(flow: str) -> str:
    if flow == "recall":
        return "Recall is disabled for this Ogham process."
    if flow == "inscribe":
        return "Inscribe is disabled for this Ogham process."
    return f"{flow} is disabled for this Ogham process."


def disabled_payload(flow: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "disabled",
        "flow": flow,
        "message": disabled_message(flow),
        **extra,
    }


def flow_status() -> dict[str, bool]:
    return {
        "recall_enabled": recall_enabled(),
        "inscribe_enabled": inscribe_enabled(),
    }
