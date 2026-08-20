"""Entity <-> OKF concept frontmatter marshalling, plus the wiki-link grammar.

Vault-LD SPEC §4.4 makes wiki links the object IRIs, and §4.5 mints identity
from the file name alone -- folders never enter the IRI. Two consequences shape
this module:

* a link resolves on its FINAL path segment, so the path in `[[entities/x-e42]]`
  disambiguates for a human and asserts nothing for the graph;
* file names must be unique across the whole bundle, so entity concepts carry
  `-e{entity_id}` the way memory concepts carry `-{uuid8}`.

Pure marshalling: no filesystem, no SQL. `bundle.py` owns IO.
"""

import re

from ogham.entity_graph import Entity, EntityEdge
from ogham.okf.identity import slugify

#: OKF requires exactly one field, `type`. This is entity concepts' value for it
#: and the reason an OKF-only consumer can read them without knowing anything
#: about Ogham.
ENTITY_OKF_TYPE = "Entity"

#: Bundle-relative directory holding entity concepts. Lives here rather than in
#: bundle.py because three modules now need to agree on it -- the writer, the
#: memory importer's skip, and the viewer's skip -- and a fourth hardcoded
#: "entities" is how one of them silently stops agreeing.
ENTITIES_DIR = "entities"

_WIKI_LINK = re.compile(r"^\[\[(.+?)\]\]$")


def make_entity_filename(entity: Entity) -> str:
    """`{slug}-e{entity_id}.md`.

    The `e` prefix is load-bearing rather than decorative: without it an entity
    whose canonical name is "1a2b3c4d" would mint the same IRI as a memory whose
    uuid8 is "1a2b3c4d", and §4.5 says two notes sharing a file name share one
    identity. It also makes the two namespaces obvious to someone reading the
    bundle by eye.
    """
    return f"{slugify(entity.canonical_name)}-e{entity.id}.md"


def wiki_link(rel_path: str) -> str:
    """Write a path-qualified wiki link.

    §4.4.1 SHOULDs path-qualification on generation even though resolution
    ignores everything but the last segment -- the path is a navigation
    affordance for humans and a hedge against future name collisions.
    """
    return f"[[{rel_path}]]"


def resolve_wiki_link(value: object) -> str | None:
    """Resolve a wiki link to the target note name, or None if it is not one.

    Per §4.4.1: the alias (`|display`) is display-only and MUST be ignored; a
    fragment (`#Heading`) addresses a location inside a note rather than a
    resource, so it is discarded and the edge resolves to the note itself; a
    path disambiguates only, so resolution uses the final segment.
    """
    if not isinstance(value, str):
        return None
    match = _WIKI_LINK.match(value.strip())
    if not match:
        return None
    target = match.group(1)
    target = target.split("|", 1)[0]  # alias is display-only
    target = target.split("#", 1)[0]  # fragment addresses a location, not a resource
    target = target.rsplit("/", 1)[-1]  # path disambiguates only
    return target.strip() or None


def entity_to_frontmatter(
    entity: Entity,
    edges: list[EntityEdge],
    aliases: list[str],
    path_by_entity_id: dict[int, str],
) -> dict:
    """Build an entity concept's frontmatter.

    ``edges`` are edges whose SUBJECT is this entity. Per D4 the exporter emits
    exactly the rows that exist and synthesises no inverses -- `inverse` is
    metadata on `entity_edge_predicates` and neither backend's `store_triple`
    writes the reciprocal row, so inventing one here would fabricate a fact.

    ``path_by_entity_id`` maps an entity id to its bundle-relative path without
    `.md`. An object missing from it is dangling: §4.4.1 and §5.6 both require
    flagging over silent loss, so it is recorded under `ogham_dangling` rather
    than dropped.

    Deliberately absent (D8, reversed by design council 2026-08-04): edge
    qualifiers. `store_triple` hardcodes `strength = 1.0` and
    `valid_from = now()`, so a sidecar carrying them would have advertised
    fidelity the write path cannot deliver. Qualifier carriage is designed
    alongside graph import, against a write path that can accept the fields.
    """
    fm: dict = {
        "type": ENTITY_OKF_TYPE,
        "entity_id": entity.id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
    }
    if aliases:
        fm["aliases"] = list(aliases)

    dangling: list[dict] = []
    for edge in edges:
        if edge.subject_id != entity.id:
            continue
        target = path_by_entity_id.get(edge.object_id)
        if target is None:
            dangling.append({"predicate": str(edge.predicate), "object_id": edge.object_id})
            continue
        fm.setdefault(str(edge.predicate), []).append(wiki_link(target))

    if dangling:
        fm["ogham_dangling"] = dangling
    return fm


#: Entity-id suffix minted by ``make_entity_filename``. Parsing it back is how
#: an imported edge target resolves to a concept without a second index.
_ENTITY_ID_SUFFIX = re.compile(r"-e(\d+)$")


def entity_id_from_note_name(note_name: str) -> int | None:
    """Recover the SOURCE entity id from a note name, or None.

    Source, not local: the id identifies the concept within the bundle it came
    from. Import maps it to a local id by natural key, never by trusting it.
    """
    m = _ENTITY_ID_SUFFIX.search(note_name)
    return int(m.group(1)) if m else None


class ParsedEntityConcept:
    """One entity concept read back from a bundle. Marshalling only, no ids."""

    __slots__ = (
        "canonical_name",
        "entity_type",
        "source_entity_id",
        "aliases",
        "edges",
        "note_name",
    )

    def __init__(
        self,
        canonical_name: str,
        entity_type: str,
        source_entity_id: int | None,
        aliases: list[str],
        edges: list[tuple[str, str]],
        note_name: str = "",
    ):
        self.canonical_name = canonical_name
        self.entity_type = entity_type
        self.source_entity_id = source_entity_id
        self.aliases = aliases
        #: (predicate, target NOTE NAME) -- resolution to an id happens later
        self.edges = edges
        #: this concept's own note name, which is what other concepts' edges
        #: point at. Resolution is by note name, never by source id.
        self.note_name = note_name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ParsedEntityConcept({self.entity_type}:{self.canonical_name!r}, "
            f"src={self.source_entity_id}, edges={len(self.edges)})"
        )


def frontmatter_to_entity(fm: dict, note_name: str = "") -> ParsedEntityConcept | None:
    """Parse an entity concept's frontmatter. Returns None if it is not one.

    **Allowlist, not denylist.** Only the 16 predicates in ``allowed_predicates``
    are read as edges. The drafted alternative enumerated what is *not* a triple,
    which meant any unknown list-valued key holding wiki-link-shaped values
    became an edge -- and vault-ld SPEC 4.3 names three host-tool keys a
    conforming tool MUST NOT emit as triples (`tags`, `aliases`, `cssclasses`),
    of which the draft covered one. An allowlist is both more correct and
    shorter.

    ``ogham_dangling`` is deliberately NOT read back as edges: the exporter
    writes it precisely because those objects were absent from the bundle, so
    there is nothing to point at.
    """
    from ogham.entity_graph import V1_PREDICATES

    if not isinstance(fm, dict) or fm.get("type") != ENTITY_OKF_TYPE:
        return None
    canonical_name = fm.get("canonical_name")
    entity_type = fm.get("entity_type")
    if not isinstance(canonical_name, str) or not canonical_name:
        return None
    if not isinstance(entity_type, str) or not entity_type:
        return None

    source_id = fm.get("entity_id")
    if not isinstance(source_id, int):
        source_id = entity_id_from_note_name(note_name)

    raw_aliases = fm.get("aliases")
    aliases = (
        [a for a in raw_aliases if isinstance(a, str) and a]
        if isinstance(raw_aliases, list)
        else []
    )

    edges: list[tuple[str, str]] = []
    for predicate in sorted(V1_PREDICATES):
        values = fm.get(predicate)
        if not isinstance(values, list):
            continue
        for value in values:
            target = resolve_wiki_link(value)
            if target:
                edges.append((predicate, target))

    return ParsedEntityConcept(canonical_name, entity_type, source_id, aliases, edges, note_name)
