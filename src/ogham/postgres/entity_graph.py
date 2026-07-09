"""Postgres backend implementing the EntityGraph Protocol.

Uses psycopg (sync) with the existing project connection pool. All SQL
lives here; the domain module never sees it. See
``docs/superpowers/plans/2026-07-02-typed-edge-v0.16-alpha.md`` Task 3
(TBU-112) for the design.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ogham.entity_graph import Entity, EntityEdge, JoinResult, Predicate


class PostgresEntityGraph:
    """Concrete EntityGraph backed by Postgres tables 041-043."""

    def __init__(self, pool: ConnectionPool[Connection[Any]], allowed_predicates: Iterable[str]):
        self._pool = pool
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
        with self._pool.connection() as conn, conn.cursor() as cur:
            # Supersede any current row with same (subject, predicate, object, profile).
            cur.execute(
                """
                UPDATE entity_edges
                   SET valid_to = now()
                 WHERE subject_id = %s AND predicate = %s AND object_id = %s
                   AND profile = %s AND valid_to IS NULL
                """,
                (subj_id, str(predicate), obj_id, profile),
            )
            # Insert new current row.
            cur.execute(
                """
                INSERT INTO entity_edges(
                    subject_id, predicate, object_id, profile,
                    fact_id, strength, metadata, derived_from, valid_from, valid_to
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now(), NULL)
                RETURNING id
                """,
                (
                    subj_id,
                    str(predicate),
                    obj_id,
                    profile,
                    source_memory_id,
                    1.0,
                    Jsonb(md),
                    Jsonb(df),
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT into entity_edges returned no id")
            new_id = row["id"]

            # If we superseded a prior row, wire superseded_by.
            cur.execute(
                """
                UPDATE entity_edges
                   SET superseded_by = %s
                 WHERE subject_id = %s AND predicate = %s AND object_id = %s
                   AND profile = %s AND valid_to IS NOT NULL AND superseded_by IS NULL
                """,
                (new_id, subj_id, str(predicate), obj_id, profile),
            )
            conn.commit()
            return int(new_id)

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
        entities_by_id: dict[int, Entity] = {}
        entities_by_id[start_id] = self._fetch_entity(start_id)
        edges: list[EntityEdge] = []
        citations: list[UUID] = []
        visited: set[int] = {start_id}

        with self._pool.connection() as conn, conn.cursor() as cur:
            for step_pred in predicate_path:
                next_ids: list[int] = []
                for cur_id in current_ids:
                    if direction == "outgoing":
                        cur.execute(
                            """
                            SELECT id, subject_id, predicate, object_id, profile,
                                   fact_id, strength, metadata, derived_from, valid_from, valid_to
                              FROM entity_edges
                             WHERE subject_id = %s AND predicate = %s
                               AND profile = %s AND valid_to IS NULL
                            """,
                            (cur_id, str(step_pred), profile),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, subject_id, predicate, object_id, profile,
                                   fact_id, strength, metadata, derived_from, valid_from, valid_to
                              FROM entity_edges
                             WHERE object_id = %s AND predicate = %s
                               AND profile = %s AND valid_to IS NULL
                            """,
                            (cur_id, str(step_pred), profile),
                        )
                    for row in cur.fetchall():
                        edge = EntityEdge(
                            id=row["id"],
                            subject_id=row["subject_id"],
                            predicate=Predicate(row["predicate"]),
                            object_id=row["object_id"],
                            profile=row["profile"],
                            fact_id=row["fact_id"],
                            strength=row["strength"],
                            metadata=row["metadata"] or {},
                            valid_from=row["valid_from"],
                            valid_to=row["valid_to"],
                            derived_from=row["derived_from"] or [],
                        )
                        edges.append(edge)
                        if edge.fact_id is not None:
                            citations.append(edge.fact_id)
                        neighbour = (
                            row["object_id"] if direction == "outgoing" else row["subject_id"]
                        )
                        if neighbour in visited:
                            continue  # cycle -- skip
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
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_aliases(entity_id, alias, profile)
                     VALUES (%s, %s, %s)
                ON CONFLICT (alias, profile) DO NOTHING
                """,
                (entity_id, alias, profile),
            )
            conn.commit()

    def resolve_alias(self, name_or_id: str | int, profile: str) -> Entity | None:
        entity_id = self._resolve_to_id(name_or_id, profile)
        if entity_id is None:
            return None
        return self._fetch_entity(entity_id)

    # -- provenance (TBU-124/125/126) ---------------------------------

    def _row_to_edge(self, row: Any) -> EntityEdge:
        return EntityEdge(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=Predicate(row["predicate"]),
            object_id=row["object_id"],
            profile=row["profile"],
            fact_id=row["fact_id"],
            strength=row["strength"],
            metadata=row["metadata"] or {},
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            derived_from=row["derived_from"] or [],
        )

    def fetch_edge(self, edge_id: int, profile: str) -> EntityEdge | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, subject_id, predicate, object_id, profile, fact_id, strength,
                          metadata, derived_from, valid_from, valid_to
                     FROM entity_edges WHERE id = %s AND profile = %s""",
                (edge_id, profile),
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row is not None else None

    def find_citing_edges(
        self, *, source_edge_id: int | None, source_memory_id: str | None, profile: str
    ) -> list[EntityEdge]:
        if source_edge_id is not None:
            needle: Any = Jsonb([{"source_edge_id": source_edge_id}])
        elif source_memory_id is not None:
            needle = Jsonb([{"source_memory_id": source_memory_id}])
        else:
            return []
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, subject_id, predicate, object_id, profile, fact_id, strength,
                          metadata, derived_from, valid_from, valid_to
                     FROM entity_edges WHERE profile = %s AND derived_from @> %s""",
                (profile, needle),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    # -- helpers -----------------------------------------------------

    def _resolve_to_id(self, name_or_id: str | int, profile: str) -> int | None:
        if isinstance(name_or_id, int):
            return name_or_id
        with self._pool.connection() as conn, conn.cursor() as cur:
            # Try canonical name first
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = %s LIMIT 1",
                (name_or_id,),
            )
            row = cur.fetchone()
            if row:
                return int(row["id"])
            # Fall back to alias
            cur.execute(
                """
                SELECT entity_id FROM entity_aliases
                 WHERE alias = %s AND profile = %s LIMIT 1
                """,
                (name_or_id, profile),
            )
            row = cur.fetchone()
            return int(row["entity_id"]) if row else None

    def _fetch_entity(self, entity_id: int) -> Entity:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_name, entity_type FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise LookupError(f"entity id {entity_id} not found")
            return Entity(
                id=int(row["id"]),
                canonical_name=row["canonical_name"],
                entity_type=row["entity_type"],
            )
