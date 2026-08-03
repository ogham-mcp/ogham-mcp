"""Demote memories that a newer memory contradicts (TBU-207).

Ogham already detects contradictions, records them as `contradicts` edges at
full strength, and reports them in `gap_note`. It then ranked as if none of
that existed. Measured on the live store, 7 of 8 queries returned a correction
below the memories it explicitly supersedes -- ask "is the managed gateway
still running" and you are told, confidently, about three regions that were
shut down.

That is the one retrieval failure that costs correctness rather than tokens.
Dilution wastes context; this returns a wrong answer with no signal it is wrong.

WHY RECENCY, NOT EDGE DIRECTION
-------------------------------
A `contradicts` edge has a source and a target, and the source is the
corrector. But `gap_contradictions_for_ids` normalises both endpoints to
(in_result, other) and discards which was which, so direction is not available
without new SQL.

It is also the weaker signal. A correction is necessarily written after the
thing it corrects, so recency carries the same information without depending on
who called `contradict_memory` in which order. Checked against every
`contradicts` edge in the live store on 2026-07-30: 148 of 148 have the source
strictly newer than the target. No exceptions.

So: when two memories contradict each other, the older one is the superseded
one. That holds whether or not the edge was recorded in the conventional
direction.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# A demoted row keeps its place in the list rather than being dropped: the
# caller may still want it, and silently removing a memory somebody stored is
# a bigger surprise than ranking it last.
_DEMOTION_FLOOR = -1.0


def find_superseded(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map in-result id -> the newer memory that supersedes it.

    `pairs` entries need `in_result_id`, `other_id` and `other_created_at`.
    A pair is ignored when either timestamp is missing or the other memory is
    not strictly newer -- an equal or older counterpart is a disagreement, not
    a correction, and guessing between two peers would be worse than leaving
    the ranking alone.
    """
    by_id = {str(r.get("id")): r for r in rows}
    superseded: dict[str, dict[str, Any]] = {}

    for pair in pairs:
        in_id = str(pair.get("in_result_id") or "")
        row = by_id.get(in_id)
        if row is None:
            continue
        mine = _as_datetime(row.get("created_at"))
        theirs = _as_datetime(pair.get("other_created_at"))
        if mine is None or theirs is None or theirs <= mine:
            continue
        # Keep the newest corrector when several contradict the same memory.
        current = superseded.get(in_id)
        if current is None or theirs > current["_when"]:
            superseded[in_id] = {
                "superseded_by": str(pair.get("other_id") or ""),
                "strength": pair.get("strength"),
                "_when": theirs,
            }

    return superseded


def apply_supersession(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Push superseded rows to the bottom and mark them, in place-safe order.

    The mark is the point as much as the reorder. A caller that only reads the
    top result gets the correction; a caller that reads the whole set can see
    which rows are stale and why, without opting into `gap="deep"` and parsing
    a separate note.
    """
    superseded = find_superseded(rows, pairs)
    if not superseded:
        return rows

    out = []
    for row in rows:
        rid = str(row.get("id"))
        hit = superseded.get(rid)
        if hit is None:
            out.append(row)
            continue
        marked = dict(row)
        marked["superseded_by"] = hit["superseded_by"]
        marked["relevance"] = _DEMOTION_FLOOR
        out.append(marked)

    logger.info(
        "supersession: demoted %d of %d results contradicted by a newer memory",
        len(superseded),
        len(rows),
    )

    # Stable sort: untouched rows keep their relative order, demoted ones sink.
    def _rank(row: dict[str, Any]) -> float:
        value = row.get("relevance")
        return float(value) if value is not None else 0.0

    out.sort(key=_rank, reverse=True)
    return out


def _as_datetime(value: Any) -> datetime | None:
    """Accept psycopg datetimes and Supabase ISO strings; reject anything else."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None
