"""Supabase backend implementing the EntityGraph Protocol.

Uses ``postgrest.SyncPostgrestClient`` directly (same convention as
``ogham.backends.supabase.SupabaseBackend`` -- avoids the heavy
``supabase`` SDK transitive deps: storage3, pyiceberg, pyroaring). All
queries go through PostgREST table calls.

Atomicity caveat: unlike ``PostgresEntityGraph.store_triple`` (one DB
transaction), PostgREST has no cross-request transaction here. The
supersession UPDATE and the new-row INSERT are two separate HTTP calls;
a crash between them could leave a superseded row with no successor
pointed at it via ``superseded_by`` (though ``valid_to`` is already
stamped, so the row is no longer "current" either way -- the partial
unique index on ``valid_to IS NULL`` is never violated). Acceptable for
v0.16 per the plan; TBU-122 integration tests should include a
partial-failure scenario if this needs hardening later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from postgrest import SyncPostgrestClient

from ogham.entity_graph import Entity, EntityEdge, JoinResult, Predicate, split_entity_ref


def _rows(data: Any) -> list[dict[str, Any]]:
    """Normalise a PostgREST list response to plain dicts.

    Mirrors ``ogham.backends.supabase._rows`` -- the PostgREST client
    types responses as a broad JSON union, which explodes pyright with
    hundreds of overload errors on every ``row["key"]`` access unless
    the shape is narrowed once, here, at the boundary.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError(f"Expected PostgREST list response, got {type(data).__name__}")
    return [_row(item) for item in data]


def _row(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected PostgREST row response, got {type(data).__name__}")
    return {str(key): value for key, value in data.items()}


logger = logging.getLogger(__name__)


class SupabaseEntityGraph:
    """Concrete EntityGraph backed by a Supabase project via PostgREST."""

    def __init__(self, client: SyncPostgrestClient, allowed_predicates: Iterable[str]):
        self._client = client
        self._allowed = set(allowed_predicates)

    # -- store_triple ------------------------------------------------

    def store_triple(
        self,
        subject: str | int,
        predicate: Predicate,
        object_: str | int,
        source_memory_id: UUID | None,
        profile: str,
        metadata: dict | None = None,
        derived_from: list[dict] | None = None,
    ) -> int:
        subj_id = self._resolve_to_id(subject, profile)
        obj_id = self._resolve_to_id(object_, profile)
        if subj_id is None or obj_id is None:
            raise ValueError(f"cannot resolve subject={subject!r} or object={object_!r}")
        if subj_id == obj_id:
            raise ValueError("self-referential edges are not allowed")

        md = metadata or {}
        df = derived_from or []
        # Supersede current row -- update valid_to on existing match.
        #
        # NOTE: value is computed client-side (not the SQL literal "now()").
        # PostgREST sends JSON payload values as-is; Postgres accepts the
        # special string 'now' (no parens) when casting text->timestamptz,
        # but NOT 'now()' -- that's function-call syntax, only valid inside
        # a SQL expression, and errors with "invalid input syntax for type
        # timestamp with time zone" when cast from a literal string. An
        # explicit ISO-8601 UTC timestamp avoids relying on that parser
        # quirk and keeps this call deterministic/testable.
        now_iso = datetime.now(UTC).isoformat()
        (
            self._client.table("entity_edges")
            .update({"valid_to": now_iso})
            .eq("subject_id", subj_id)
            .eq("predicate", str(predicate))
            .eq("object_id", obj_id)
            .eq("profile", profile)
            .is_("valid_to", "null")
            .execute()
        )
        # Insert new current row.
        result = (
            self._client.table("entity_edges")
            .insert(
                {
                    "subject_id": subj_id,
                    "predicate": str(predicate),
                    "object_id": obj_id,
                    "profile": profile,
                    "fact_id": str(source_memory_id) if source_memory_id else None,
                    "strength": 1.0,
                    "metadata": md,
                    "derived_from": df,
                    # valid_from defaults to now(); valid_to null
                }
            )
            .execute()
        )
        inserted = _rows(result.data)
        if not inserted:
            raise RuntimeError("INSERT into entity_edges returned no row")
        new_id = int(inserted[0]["id"])

        # Wire superseded_by on the just-updated row.
        (
            self._client.table("entity_edges")
            .update({"superseded_by": new_id})
            .eq("subject_id", subj_id)
            .eq("predicate", str(predicate))
            .eq("object_id", obj_id)
            .eq("profile", profile)
            .not_.is_("valid_to", "null")
            .is_("superseded_by", "null")
            .execute()
        )
        return new_id

    # -- query_join --------------------------------------------------

    def query_join(
        self,
        start_entity: str | int,
        predicate_path: list[Predicate],
        profile: str,
        hop_limit: int,
        direction: str = "outgoing",
    ) -> JoinResult | None:
        if hop_limit < len(predicate_path):
            raise ValueError(
                f"hop_limit={hop_limit} smaller than predicate_path length {len(predicate_path)}"
            )

        start_id = self._resolve_to_id(start_entity, profile)
        if start_id is None:
            return None

        current_ids: list[int] = [start_id]
        entities_by_id: dict[int, Entity] = {start_id: self._fetch_entity(start_id)}
        edges: list[EntityEdge] = []
        citations: list[UUID] = []
        visited: set[int] = {start_id}

        for step_pred in predicate_path:
            next_ids: list[int] = []
            for cur_id in current_ids:
                q = self._client.table("entity_edges").select(
                    "id,subject_id,predicate,object_id,profile,fact_id,strength,metadata,"
                    "derived_from,valid_from,valid_to"
                )
                if direction == "outgoing":
                    q = q.eq("subject_id", cur_id)
                else:
                    q = q.eq("object_id", cur_id)
                q = q.eq("predicate", str(step_pred)).eq("profile", profile).is_("valid_to", "null")
                for row in _rows(q.execute().data):
                    edge = EntityEdge(
                        id=int(row["id"]),
                        subject_id=int(row["subject_id"]),
                        predicate=Predicate(row["predicate"]),
                        object_id=int(row["object_id"]),
                        profile=row["profile"],
                        fact_id=UUID(row["fact_id"]) if row["fact_id"] else None,
                        strength=float(row["strength"]),
                        metadata=row["metadata"] or {},
                        valid_from=row["valid_from"],
                        valid_to=row["valid_to"],
                        derived_from=row.get("derived_from") or [],
                    )
                    edges.append(edge)
                    if edge.fact_id is not None:
                        citations.append(edge.fact_id)
                    neighbour = (
                        int(row["object_id"]) if direction == "outgoing" else int(row["subject_id"])
                    )
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    next_ids.append(neighbour)
                    if neighbour not in entities_by_id:
                        entities_by_id[neighbour] = self._fetch_entity(neighbour)
            if not next_ids:
                return None
            current_ids = next_ids

        # entities_by_id is a regular dict; Python guarantees insertion-order
        # iteration (3.7+), and start_id/neighbours are only ever inserted
        # the first time they're encountered above -- so .values() already
        # yields BFS insertion order. Do NOT sort by id (TBU-150): the
        # entities list is a path signal for MCP-tool consumers, not an
        # id-sorted set. See JoinResult docstring in ogham.entity_graph.
        entity_list = list(entities_by_id.values())
        return JoinResult(entities=entity_list, edges=edges, citations=citations)

    # -- aliases -----------------------------------------------------

    def add_alias(self, entity_id: int, alias: str, profile: str) -> None:
        # ignore_duplicates=True -> Prefer: resolution=ignore-duplicates,
        # i.e. INSERT ... ON CONFLICT DO NOTHING. Without it, postgrest-py's
        # default (ignore_duplicates=False) sends resolution=merge-duplicates
        # (ON CONFLICT DO UPDATE), which would silently repoint an existing
        # alias to a new entity_id on a duplicate (alias, profile) -- last-
        # write-wins. That diverges from the Postgres backend's
        # `ON CONFLICT (alias, profile) DO NOTHING` (first-write-wins) and
        # breaks the EntityGraph Protocol's cross-backend behavioural
        # contract. First-write-wins is the safer default here; callers who
        # actually want to repoint an alias should delete + re-add it
        # explicitly rather than relying on upsert to silently overwrite.
        (
            self._client.table("entity_aliases")
            .upsert(
                {"entity_id": entity_id, "alias": alias, "profile": profile},
                on_conflict="alias,profile",
                ignore_duplicates=True,
            )
            .execute()
        )

    def resolve_alias(self, name_or_id: str | int, profile: str) -> Entity | None:
        entity_id = self._resolve_to_id(name_or_id, profile)
        if entity_id is None:
            return None
        return self._fetch_entity(entity_id)

    # -- provenance (TBU-124/125/126) ---------------------------------

    def _row_to_edge(self, row: dict[str, Any]) -> EntityEdge:
        return EntityEdge(
            id=int(row["id"]),
            subject_id=int(row["subject_id"]),
            predicate=Predicate(row["predicate"]),
            object_id=int(row["object_id"]),
            profile=row["profile"],
            fact_id=UUID(row["fact_id"]) if row["fact_id"] else None,
            strength=float(row["strength"]),
            metadata=row["metadata"] or {},
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            derived_from=row.get("derived_from") or [],
        )

    def fetch_edge(self, edge_id: int, profile: str) -> EntityEdge | None:
        result = (
            self._client.table("entity_edges")
            .select("*")
            .eq("id", edge_id)
            .eq("profile", profile)
            .execute()
        )
        rows = _rows(result.data)
        return self._row_to_edge(rows[0]) if rows else None

    def find_citing_edges(
        self, *, source_edge_id: int | None, source_memory_id: str | None, profile: str
    ) -> list[EntityEdge]:
        if source_edge_id is not None:
            needle: list[dict[str, Any]] = [{"source_edge_id": source_edge_id}]
        elif source_memory_id is not None:
            needle = [{"source_memory_id": source_memory_id}]
        else:
            return []
        # JSON-encode the needle before .contains(): postgrest-py's
        # .contains() only json.dumps() a dict value; for a list it assumes
        # a native Postgres array column and does ",".join(value), which
        # raises TypeError on a list of dicts. Pre-encoding to a string
        # takes the string branch instead, sending `cs.<value>` verbatim --
        # matching the spec's intended `derived_from=cs.[...]` filter.
        result = (
            self._client.table("entity_edges")
            .select("*")
            .eq("profile", profile)
            .contains("derived_from", json.dumps(needle))
            .execute()
        )
        return [self._row_to_edge(row) for row in _rows(result.data)]

    # -- enumeration (TBU-130) ---------------------------------------

    def list_entities(self, profile: str) -> list[Entity]:
        """Every entity reachable in ``profile``, ordered by id.

        PostgREST has no IN-subquery, so the three-way union the Postgres
        backend does in SQL happens here in Python instead: memory links plus
        both endpoints of every edge. Including both endpoints is what
        guarantees an OKF export never writes an edge whose object is missing.
        """
        ids: set[int] = set()

        linked = (
            self._client.table("memory_entities")
            .select("entity_id")
            .eq("profile", profile)
            .execute()
        )
        for row in _rows(linked.data):
            ids.add(int(row["entity_id"]))

        edge_rows = (
            self._client.table("entity_edges")
            .select("subject_id,object_id")
            .eq("profile", profile)
            .execute()
        )
        for row in _rows(edge_rows.data):
            ids.add(int(row["subject_id"]))
            ids.add(int(row["object_id"]))

        # `.in_("id", [])` is a PostgREST error, not an empty result -- so a
        # profile with no entities has to short-circuit before the query is
        # built rather than relying on the filter to return nothing.
        if not ids:
            return []

        result = (
            self._client.table("entities")
            .select("id,canonical_name,entity_type")
            .in_("id", sorted(ids))
            .order("id")
            .execute()
        )
        return [
            Entity(
                id=int(row["id"]),
                canonical_name=row["canonical_name"],
                entity_type=row["entity_type"],
            )
            for row in _rows(result.data)
        ]

    def list_edges(self, profile: str, *, current_only: bool = True) -> list[EntityEdge]:
        query = (
            self._client.table("entity_edges")
            .select(
                "id,subject_id,predicate,object_id,profile,fact_id,strength,metadata,"
                "derived_from,valid_from,valid_to"
            )
            .eq("profile", profile)
        )
        if current_only:
            query = query.is_("valid_to", "null")
        return [self._row_to_edge(row) for row in _rows(query.order("id").execute().data)]

    def list_aliases(self, profile: str) -> dict[int, list[str]]:
        result = (
            self._client.table("entity_aliases")
            .select("entity_id,alias")
            .eq("profile", profile)
            .order("entity_id")
            .execute()
        )
        out: dict[int, list[str]] = {}
        for row in _rows(result.data):
            out.setdefault(int(row["entity_id"]), []).append(row["alias"])
        return out

    # -- helpers -----------------------------------------------------

    def find_entity(self, canonical_name: str, entity_type: str) -> int | None:
        """Read-only lookup on the exact natural key. Mirrors the Postgres backend."""
        result = (
            self._client.table("entities")
            .select("id")
            .eq("canonical_name", canonical_name)
            .eq("entity_type", entity_type)
            .limit(1)
            .execute()
        )
        rows = _rows(result.data)
        return int(rows[0]["id"]) if rows else None

    def upsert_entity(self, canonical_name: str, entity_type: str) -> int:
        """Get or create by natural key. Mirrors the Postgres backend.

        ``mention_count`` is untouched: an import is not a mention.
        """
        existing = (
            self._client.table("entities")
            .select("id")
            .eq("canonical_name", canonical_name)
            .eq("entity_type", entity_type)
            .limit(1)
            .execute()
        )
        rows = _rows(existing.data)
        if rows:
            return int(rows[0]["id"])
        created = (
            self._client.table("entities")
            .insert({"canonical_name": canonical_name, "entity_type": entity_type})
            .execute()
        )
        new_rows = _rows(created.data)
        if not new_rows:
            raise RuntimeError(f"could not create entity {entity_type}:{canonical_name}")
        return int(new_rows[0]["id"])

    def _resolve_to_id(self, name_or_id: str | int, profile: str) -> int | None:
        """Resolve a name, a qualified ``type:name`` ref, or an id to an entity id.

        Mirrors the Postgres backend exactly. ``entities`` is
        UNIQUE (canonical_name, entity_type), so a BARE NAME IS NOT A KEY --
        every ``*Error`` lands under both ``entity:`` and ``error:``. The
        previous name-only ``limit(1)`` resolved between them arbitrarily.
        """
        if isinstance(name_or_id, int):
            return name_or_id
        entity_type, name = split_entity_ref(name_or_id)
        # Exact natural key when the caller qualified the reference.
        if entity_type is not None:
            result = (
                self._client.table("entities")
                .select("id")
                .eq("canonical_name", name)
                .eq("entity_type", entity_type)
                .limit(1)
                .execute()
            )
            rows = _rows(result.data)
            if rows:
                return int(rows[0]["id"])
        # Unqualified: deterministic, and say so when it was ambiguous.
        result = (
            self._client.table("entities")
            .select("id,entity_type")
            .eq("canonical_name", name_or_id)
            .order("id")
            .execute()
        )
        rows = _rows(result.data)
        if rows:
            if len(rows) > 1:
                logger.warning(
                    "entity reference %r is ambiguous across types %s -- resolving to "
                    "id=%s. Qualify it as '<type>:%s' to be explicit.",
                    name_or_id,
                    [r.get("entity_type") for r in rows],
                    rows[0]["id"],
                    name_or_id,
                )
            return int(rows[0]["id"])
        # Fall back to alias
        result = (
            self._client.table("entity_aliases")
            .select("entity_id")
            .eq("alias", name_or_id)
            .eq("profile", profile)
            .limit(1)
            .execute()
        )
        alias_rows = _rows(result.data)
        return int(alias_rows[0]["entity_id"]) if alias_rows else None

    def _fetch_entity(self, entity_id: int) -> Entity:
        # Deliberately NOT .single() -- PostgREST returns HTTP 406 (which
        # postgrest-py raises as APIError, not a clean empty/None result)
        # when .single() gets 0 or >1 rows. That would make the
        # missing-entity case surface as an uncaught APIError instead of
        # the LookupError this method promises, diverging from the
        # Postgres backend's clean `row is None -> raise LookupError`.
        # Plain select + empty-list check keeps the same LookupError
        # contract across both backends.
        result = (
            self._client.table("entities")
            .select("id,canonical_name,entity_type")
            .eq("id", entity_id)
            .execute()
        )
        rows = _rows(result.data)
        if not rows:
            raise LookupError(f"entity id {entity_id} not found")
        row = rows[0]
        return Entity(
            id=int(row["id"]), canonical_name=row["canonical_name"], entity_type=row["entity_type"]
        )
