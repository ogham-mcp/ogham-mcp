"""Entity <-> OKF concept frontmatter marshalling.

Two consequences of how bundles mint identity shape this module:

* a link resolves on its FINAL path segment (the note name), so the directory
  in `../entities/x-e42.md` disambiguates for a human and asserts nothing for
  the graph;
* file names must be unique across the whole bundle, so entity concepts carry
  `-e{entity_id}` the way memory concepts carry `-{uuid8}`.

OKF v0.3 (TBU-297): every Ogham-specific key is `x-ogham-*`, and each edge is
carried twice -- a structured `x-ogham-<predicate>` key holding relative paths,
and a marked body section of Markdown links that a v0.3 consumer counts as
graph edges (§6.1, §6.4). `aliases` is the one exception to the prefix: it
stays bare so Obsidian keeps reading it. The importer still reads the v0.1
shape: bare `entity_id`/`canonical_name`/`entity_type`/`ogham_dangling`,
upper-case predicate keys, and `[[wiki link]]` values.

Pure marshalling: no filesystem, no SQL. `bundle.py` owns IO.
"""

import re

from ogham.entity_graph import Entity, EntityEdge
from ogham.okf.identity import slugify
from ogham.okf.links import (
    RELATIONS_SECTION,
    build_section,
    relative_target,
    resolve_link_note_name,
    resolve_wiki_link,
    wiki_link,
    x_key,
)

#: OKF requires exactly one field, `type`. This is entity concepts' value for it
#: and the reason an OKF-only consumer can read them without knowing anything
#: about Ogham.
ENTITY_OKF_TYPE = "Entity"

#: Bundle-relative directory holding entity concepts. Lives here rather than in
#: bundle.py because three modules now need to agree on it -- the writer, the
#: memory importer's skip, and the viewer's skip -- and a fourth hardcoded
#: "entities" is how one of them silently stops agreeing.
ENTITIES_DIR = "entities"

#: Frontmatter keys: the v0.3 name first, then the v0.1 name the importer still
#: reads. The new name wins when a file carries both.
ENTITY_ID_KEYS = (x_key("entity-id"), "entity_id")
CANONICAL_NAME_KEYS = (x_key("canonical-name"), "canonical_name")
ENTITY_TYPE_KEYS = (x_key("entity-type"), "entity_type")
#: `aliases` stays UNPREFIXED on purpose (Kevin, 2026-11-19): it is the key
#: Obsidian reads for its native note aliases, and Obsidian is a supported
#: source. The validator's I203 info for it is accepted. `x-ogham-aliases` is
#: still read, for bundles written by the first v0.20 development commit.
ALIASES_KEYS = ("aliases", x_key("aliases"))
DANGLING_KEY = x_key("dangling")

__all__ = [
    "ENTITIES_DIR",
    "ENTITY_OKF_TYPE",
    "ParsedEntityConcept",
    "entity_id_from_note_name",
    "entity_to_body",
    "entity_to_frontmatter",
    "frontmatter_to_entity",
    "make_entity_filename",
    "predicate_key",
    # Re-exported from okf.links: callers and tests have always imported these
    # from here.
    "resolve_wiki_link",
    "wiki_link",
]


def predicate_key(predicate: str) -> str:
    """Frontmatter key for an edge predicate: `OWNS` -> `x-ogham-owns`.

    `MENTIONS` maps to `x-ogham-mentions`, the same key memory concepts use for
    their entity links, so one JSON-LD term covers both.
    """
    return x_key(str(predicate).lower().replace("_", "-"))


def _first(fm: dict, keys: tuple[str, ...]):
    """The value under the first key present -- new name wins over legacy."""
    for key in keys:
        if key in fm:
            return fm[key]
    return None


def make_entity_filename(entity: Entity) -> str:
    """`{slug}-e{entity_id}.md`.

    The `e` prefix is load-bearing rather than decorative: without it an entity
    whose canonical name is "1a2b3c4d" would mint the same note name as a memory
    whose uuid8 is "1a2b3c4d", and two notes sharing a file name share one
    identity. It also makes the two namespaces obvious to someone reading the
    bundle by eye.
    """
    return f"{slugify(entity.canonical_name)}-e{entity.id}.md"


def _subject_edges(
    entity: Entity, edges: list[EntityEdge], path_by_entity_id: dict[int, str]
) -> tuple[list[tuple[EntityEdge, str]], list[dict]]:
    """Split this entity's edges into (edge, link target) pairs and dangling ones.

    Stable-sorted by predicate so frontmatter key order does not depend on the
    order the database returned rows in: a re-export of an imported bundle
    reads edges back predicate by predicate, and has to write the same bytes.
    """
    linked: list[tuple[EntityEdge, str]] = []
    dangling: list[dict] = []
    for edge in sorted(edges, key=lambda e: str(e.predicate)):
        if edge.subject_id != entity.id:
            continue
        target = path_by_entity_id.get(edge.object_id)
        if target is None:
            dangling.append({"predicate": str(edge.predicate), "object_id": edge.object_id})
            continue
        linked.append((edge, relative_target(ENTITIES_DIR, f"{target}.md")))
    return linked, dangling


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
    `.md`. An object missing from it is dangling, and is recorded under
    `x-ogham-dangling` rather than dropped -- flagging beats silent loss.

    Each edge value is a path relative to this file (OKF §6 rule 3). The same
    edges also go into the body as Markdown links (``entity_to_body``).

    Deliberately absent (D8, reversed by design council 2026-08-04): edge
    qualifiers. `store_triple` hardcodes `strength = 1.0` and
    `valid_from = now()`, so a sidecar carrying them would have advertised
    fidelity the write path cannot deliver.
    """
    fm: dict = {
        "type": ENTITY_OKF_TYPE,
        ENTITY_ID_KEYS[0]: entity.id,
        CANONICAL_NAME_KEYS[0]: entity.canonical_name,
        ENTITY_TYPE_KEYS[0]: entity.entity_type,
    }
    if aliases:
        fm[ALIASES_KEYS[0]] = list(aliases)

    linked, dangling = _subject_edges(entity, edges, path_by_entity_id)
    for edge, target in linked:
        fm.setdefault(predicate_key(str(edge.predicate)), []).append(target)

    if dangling:
        fm[DANGLING_KEY] = dangling
    return fm


def entity_to_body(
    entity: Entity,
    edges: list[EntityEdge],
    path_by_entity_id: dict[int, str],
    name_by_entity_id: dict[int, str],
) -> str:
    """The marked Relations section, or "" for an entity with no linked edges.

    Entity concepts have no prose, so the section IS the body. Each line names
    the predicate before the link: OKF §6.1 makes a link an untyped edge and
    says the text near it tells the kind, so this is where the type goes for a
    v0.3 consumer. Dangling edges have no file to link to and stay
    frontmatter-only.
    """
    linked, _ = _subject_edges(entity, edges, path_by_entity_id)
    if not linked:
        return ""
    items = [
        (str(edge.predicate), name_by_entity_id.get(edge.object_id, target), target)
        for edge, target in linked
    ]
    return build_section(RELATIONS_SECTION, "Relations", items)


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

    **Allowlist, not denylist.** Only the 16 predicates in ``V1_PREDICATES``
    are read as edges, under either key spelling (`x-ogham-owns` or the v0.1
    `OWNS`). An allowlist means an unknown list-valued key holding link-shaped
    values never becomes an edge -- Vault-LD names three host-tool keys a
    conforming tool MUST NOT emit as triples (`tags`, `aliases`, `cssclasses`).

    Values may be relative paths, bundle-root paths, or legacy wiki links; all
    resolve to the target note name. The body Relations section is NOT read:
    it mirrors the frontmatter, and reading one carrier keeps edges from being
    counted twice.

    The dangling list is deliberately NOT read back as edges: the exporter
    writes it precisely because those objects were absent from the bundle, so
    there is nothing to point at.
    """
    from ogham.entity_graph import V1_PREDICATES

    if not isinstance(fm, dict) or fm.get("type") != ENTITY_OKF_TYPE:
        return None
    canonical_name = _first(fm, CANONICAL_NAME_KEYS)
    entity_type = _first(fm, ENTITY_TYPE_KEYS)
    if not isinstance(canonical_name, str) or not canonical_name:
        return None
    if not isinstance(entity_type, str) or not entity_type:
        return None

    source_id = _first(fm, ENTITY_ID_KEYS)
    if not isinstance(source_id, int) or isinstance(source_id, bool):
        source_id = entity_id_from_note_name(note_name)

    raw_aliases = _first(fm, ALIASES_KEYS)
    aliases = (
        [a for a in raw_aliases if isinstance(a, str) and a]
        if isinstance(raw_aliases, list)
        else []
    )

    edges: list[tuple[str, str]] = []
    for predicate in sorted(V1_PREDICATES):
        values = _first(fm, (predicate_key(predicate), predicate))
        if not isinstance(values, list):
            continue
        for value in values:
            target = resolve_link_note_name(value)
            if target:
                edges.append((predicate, target))

    return ParsedEntityConcept(canonical_name, entity_type, source_id, aliases, edges, note_name)
