"""Deterministic gap signals for retrieval results (task #262, v1).

SQL/Python supply the facts; the LLM (prose level) only narrates them. The
agent gets a grounded sense of what a result set is missing or can't be
trusted on. In-result signals are computed here over rows already fetched
by hybrid_search; out-of-result signals (the `deep` level) come from a
result-ID-scoped backend lookup (see database.gap_out_of_result_contradictions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ogham.database import gap_out_of_result_contradictions
from ogham.llm import synthesize

logger = logging.getLogger(__name__)


def _parse_dt(value: str | datetime | None) -> datetime | None:
    """Normalize a DB-row date value to a tz-aware datetime.

    The Postgres backend (psycopg) returns `created_at` as a `datetime`
    object; the Supabase REST backend returns an ISO string. Accept both
    (and reject anything else) so this never crashes hybrid_search.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class GapReport:
    """Structured gap facts for one scope. Every field is grounded in data."""

    scope_size: int = 0
    staleness: dict[str, Any] = field(default_factory=dict)
    low_confidence: dict[str, Any] = field(default_factory=dict)
    contradictions: dict[str, Any] = field(default_factory=lambda: {"count": 0, "pairs": []})
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_size": self.scope_size,
            "staleness": self.staleness,
            "low_confidence": self.low_confidence,
            "contradictions": self.contradictions,
            "coverage": self.coverage,
        }


def compute_in_result_signals(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    stale_days: int,
    confidence_floor: float,
) -> GapReport:
    """Compute staleness + low-confidence over the in-hand result rows. No DB."""
    report = GapReport(scope_size=len(rows))

    ages_days: list[int] = []
    for row in rows:
        dt = _parse_dt(row.get("created_at", ""))
        if dt is not None:
            ages_days.append(max(0, (now - dt).days))

    if ages_days:
        newest = min(ages_days)
        report.staleness = {
            "newest_write_days": newest,
            "older_than_stale_days": sum(1 for a in ages_days if a > stale_days),
            "stale_days_threshold": stale_days,
        }
    else:
        report.staleness = {
            "newest_write_days": None,
            "older_than_stale_days": 0,
            "stale_days_threshold": stale_days,
        }

    confidences = [
        row["confidence"] for row in rows if isinstance(row.get("confidence"), (int, float))
    ]
    below = [c for c in confidences if c < confidence_floor]
    report.low_confidence = {
        "below_floor": len(below),
        "confidence_floor": confidence_floor,
        "max_confidence": max(confidences)
        if confidences
        else None,  # max across the whole result set, not just below-floor
    }
    # coverage left empty here; the deep level fills it
    return report


def is_material(report: GapReport) -> bool:
    """True when the report contains something worth surfacing to the agent."""
    if report.contradictions.get("count", 0) > 0:
        return True
    if report.low_confidence.get("below_floor", 0) > 0:
        return True
    newest = report.staleness.get("newest_write_days")
    threshold = report.staleness.get("stale_days_threshold")
    if newest is not None and threshold is not None and newest > threshold:
        return True
    return False


def render_template(report: GapReport) -> str:
    """Deterministic Markdown 'heads up' from grounded facts. Empty when healthy."""
    if not is_material(report):
        return ""
    parts: list[str] = []
    n = report.staleness.get("newest_write_days")
    if n is not None and n > report.staleness.get("stale_days_threshold", 0):
        parts.append(f"newest write in this scope is {n} days old, so it may be out of date")
    c = report.contradictions.get("count", 0)
    if c:
        parts.append(f"{c} unresolved contradiction(s) among these memories")
    bf = report.low_confidence.get("below_floor", 0)
    if bf:
        parts.append(f"{bf} of {report.scope_size} memories are low-confidence")
    cov = report.coverage.get("note")
    if cov:
        parts.append(cov)
    return "Heads up before relying on this: " + "; ".join(parts) + "."


_NARRATE_SYSTEM = (
    "You write a one-sentence 'heads up' for an AI agent about gaps in its memory. "
    "Use ONLY the facts in the JSON. Never invent a gap. Be specific and under 40 words. "
    "Frame as 'no recent activity / unresolved contradiction', not 'you don't know'."
)


def narrate(report: GapReport, *, provider: str, model: str) -> str:
    """LLM-narrated Markdown from grounded facts; template fallback on any error."""
    if not is_material(report):
        return ""
    import json

    prompt = (
        "Render a short heads-up for the agent from these grounded gap facts.\n\n"
        + json.dumps(report.to_dict(), default=str)
    )
    try:
        # max_tokens must leave room for "thinking" models (e.g. Gemini 2.5
        # Flash): at 160 the reasoning tokens consume the budget and the visible
        # answer is truncated to a fragment. The heads-up itself is one short
        # sentence; 1024 is headroom, not output length.
        out = synthesize(
            prompt, provider=provider, model=model, system=_NARRATE_SYSTEM, max_tokens=1024
        )
        return out.strip() or render_template(report)
    except Exception as exc:  # noqa: BLE001 — narration must never break the read path
        logger.debug("gap narration fell back to template: %s", exc)
        return render_template(report)


def _coverage_lookup(profile: str, tags: list[str] | None) -> dict[str, Any]:
    """Best-effort topic-summary coverage. v1 stub seam: returns {} by default.

    Kept as a separate function so compute_deep_signals can degrade to
    "wiki not enabled" if a future implementation queries a topic_summaries
    table that a vanilla-Postgres self-hoster never migrated in.
    """
    return {}


def compute_deep_signals(
    report: GapReport,
    *,
    profile: str,
    result_ids: list[str],
    tags: list[str] | None = None,
) -> GapReport:
    """Augment an in-result report with out-of-result contradictions + coverage."""
    if result_ids:
        report.contradictions = gap_out_of_result_contradictions(profile, result_ids)
    try:
        report.coverage = _coverage_lookup(profile, tags)
    except Exception as exc:  # noqa: BLE001 — coverage is best-effort, never breaks the path
        if "topic_summaries" in str(exc) or "does not exist" in str(exc):
            report.coverage = {"note": "wiki not enabled"}
        else:
            logger.debug("gap coverage lookup skipped: %s", exc)
            report.coverage = {}
    return report
