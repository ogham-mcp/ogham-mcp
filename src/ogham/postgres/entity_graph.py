"""Postgres backend implementing the EntityGraph Protocol.

Uses psycopg (sync) with the existing project connection pool. All SQL
lives here; the domain module never sees it. See
``docs/superpowers/plans/2026-07-02-typed-edge-v0.16-alpha.md`` Task 3
(TBU-112) for the design.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ogham.entity_graph import Entity, EntityEdge, JoinResult, Predicate, split_entity_ref

logger = logging.getLogger(__name__)


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

    # -- enumeration (TBU-130) ---------------------------------------

    def list_entities(self, profile: str) -> list[Entity]:
        """Every entity reachable in ``profile``, ordered by id.

        NOT a table scan: `entities` is global (no profile column), so the
        profile scope comes from `memory_entities` plus both endpoints of
        `entity_edges`. Including both endpoints is what guarantees every edge
        an OKF export writes has its object present as a concept.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_name, entity_type
                  FROM entities
                 WHERE id IN (
                        SELECT entity_id FROM memory_entities WHERE profile = %(p)s
                        UNION
                        SELECT subject_id FROM entity_edges WHERE profile = %(p)s
                        UNION
                        SELECT object_id  FROM entity_edges WHERE profile = %(p)s
                       )
                 ORDER BY id
                """,
                {"p": profile},
            )
            return [
                Entity(
                    id=int(row["id"]),
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                )
                for row in cur.fetchall()
            ]

    def list_edges(self, profile: str, *, current_only: bool = True) -> list[EntityEdge]:
        # The interpolated fragment is one of two literals chosen by a bool --
        # no caller input reaches the SQL string. Values stay parameterised.
        current_clause = "AND valid_to IS NULL" if current_only else ""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, subject_id, predicate, object_id, profile, fact_id, strength,
                       metadata, derived_from, valid_from, valid_to
                  FROM entity_edges
                 WHERE profile = %(p)s {current_clause}
                 ORDER BY id
                """,
                {"p": profile},
            )
            return [self._row_to_edge(row) for row in cur.fetchall()]

    def list_aliases(self, profile: str) -> dict[int, list[str]]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_id, alias
                  FROM entity_aliases
                 WHERE profile = %(p)s
                 ORDER BY entity_id, alias
                """,
                {"p": profile},
            )
            out: dict[int, list[str]] = {}
            for row in cur.fetchall():
                out.setdefault(int(row["entity_id"]), []).append(row["alias"])
            return out

    # -- helpers -----------------------------------------------------

    def find_entity(self, canonical_name: str, entity_type: str) -> int | None:
        """Read-only lookup on the exact natural key. See the protocol docstring."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = %s AND entity_type = %s",
                (canonical_name, entity_type),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else None

    def upsert_entity(self, canonical_name: str, entity_type: str) -> int:
        """Get or create by natural key. See the protocol docstring.

        ``mention_count`` is untouched on both branches: an import is not a
        mention, and it feeds ranking in every profile.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type)
                VALUES (%s, %s)
                ON CONFLICT (canonical_name, entity_type) DO UPDATE
                    SET canonical_name = EXCLUDED.canonical_name
                RETURNING id
                """,
                (canonical_name, entity_type),
            )
            row = cur.fetchone()
            conn.commit()
            assert row is not None, "INSERT ... RETURNING always yields a row"
            return int(row["id"])

    def _resolve_to_id(self, name_or_id: str | int, profile: str) -> int | None:
        """Resolve a name, a qualified ``type:name`` ref, or an id to an entity id.

        ``entities`` is UNIQUE (canonical_name, entity_type), so a BARE NAME IS
        NOT A KEY. Ordinary extraction produces the same name under two types
        routinely -- every ``*Error`` matches both the CamelCase rule and the
        Error-suffix rule, giving ``entity:KeyError`` and ``error:KeyError``
        (18 such names on the live store, 2026-08-20). The previous
        ``LIMIT 1`` with no ORDER BY resolved between them arbitrarily, so a
        ``store_triple`` edge could attach to whichever row the planner
        happened to return.
        """
        if isinstance(name_or_id, int):
            return name_or_id
        entity_type, name = split_entity_ref(name_or_id)
        with self._pool.connection() as conn, conn.cursor() as cur:
            # Exact natural key when the caller qualified the reference.
            if entity_type is not None:
                cur.execute(
                    "SELECT id FROM entities WHERE canonical_name = %s AND entity_type = %s",
                    (name, entity_type),
                )
                row = cur.fetchone()
                if row:
                    return int(row["id"])
            # Unqualified: deterministic, and say so when it was ambiguous
            # rather than picking silently.
            cur.execute(
                "SELECT id, entity_type FROM entities WHERE canonical_name = %s ORDER BY id",
                (name_or_id,),
            )
            rows = cur.fetchall() or []
            if rows:
                if len(rows) > 1:
                    logger.warning(
                        "entity reference %r is ambiguous across types %s -- resolving to "
                        "id=%s. Qualify it as '<type>:%s' to be explicit.",
                        name_or_id,
                        [r["entity_type"] for r in rows],
                        rows[0]["id"],
                        name_or_id,
                    )
                return int(rows[0]["id"])
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
