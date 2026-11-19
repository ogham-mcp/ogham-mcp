"""Link grammar for OKF v0.3 bundles: Markdown body links plus legacy wiki links.

OKF v0.3 §6.1 is explicit: only ``[text](target)`` links (and reference-style
links) are links, and ``[[wikilinks]]`` are not. Up to v0.19 Ogham wrote every
edge as a wiki link in frontmatter, so a v0.3 consumer counting the bundle graph
(§6.4) saw zero edges. Two carriers now exist side by side:

* a structured ``x-ogham-*`` frontmatter key, which is what Ogham's own importer
  reads. Values are relative paths resolved per §6.
* a **marked body section** of standard Markdown links, so any v0.3 consumer
  sees the same edges. The importer strips exactly that section, so memory
  content round-trips byte for byte.

The marked section looks like this, and is always the LAST thing in the body::

    <!-- x-ogham-mentions -->
    ## Mentions

    - [Postgres](../entities/postgres-e1.md)
    <!-- /x-ogham-mentions -->

The HTML comment pair is the marker, not the heading. A memory whose content
merely contains ``## Mentions`` is never stripped, and stripping also requires
the structured key to be present (callers check that), so a hand-written block
in content without a matching key is left alone.

Pure functions: no filesystem, no SQL.
"""

from __future__ import annotations

import posixpath
import re
from urllib.parse import unquote

#: Prefix for every Ogham-specific frontmatter key. OKF v0.3 §4.1: future OKF
#: versions will not define `x-` keys, and producers SHOULD name theirs
#: `x-<producer>-<name>`.
X_PREFIX = "x-ogham-"

#: Marker names for the two body sections Ogham writes. Each doubles as the
#: structured frontmatter key that carries the same edges, so a reader of the
#: file can see which key the section mirrors.
MENTIONS_SECTION = "x-ogham-mentions"
RELATIONS_SECTION = "x-ogham-relations"

_WIKI_LINK = re.compile(r"^\[\[(.+?)\]\]$")
_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def x_key(name: str) -> str:
    """`x-ogham-<name>`, the one place the producer prefix is spelled."""
    return f"{X_PREFIX}{name}"


def wiki_link(rel_path: str) -> str:
    """Write a path-qualified wiki link. Legacy (OKF v0.1 bundles) only.

    Kept so tests and legacy fixtures can produce the v0.1 shape; the exporter
    no longer emits wiki links.
    """
    return f"[[{rel_path}]]"


def resolve_wiki_link(value: object) -> str | None:
    """Resolve a wiki link to the target note name, or None if it is not one.

    Per Vault-LD §4.4.1: the alias (`|display`) is display-only and MUST be
    ignored; a fragment (`#Heading`) addresses a location inside a note rather
    than a resource, so it is discarded and the edge resolves to the note
    itself; a path disambiguates only, so resolution uses the final segment.
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


def resolve_link_note_name(value: object) -> str | None:
    """Resolve any link value Ogham has ever written to the target note name.

    Accepts all three in-bundle forms: a legacy ``[[wiki link]]``, a path
    relative to the file (``../entities/x-e1.md``), and a bundle-root path
    (``/entities/x-e1.md``). OKF §11.2 requires consumers to resolve both path
    forms.

    Returns the final path segment without ``.md`` -- the note name, which is
    the identity the graph importer resolves on (file names are unique across
    the bundle by construction: ``-{uuid8}`` and ``-e{id}`` suffixes). A URL
    with a scheme is external and returns None.
    """
    legacy = resolve_wiki_link(value)
    if legacy is not None:
        return legacy
    if not isinstance(value, str):
        return None
    target = value.strip()
    if not target or _URI_SCHEME.match(target) or target.startswith("//"):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    name = target.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".md"):
        name = name[:-3]
    return name or None


def relative_target(from_dir: str, bundle_path: str) -> str:
    """``bundle_path`` as a link from a file in ``from_dir``, per OKF §6 rule 3.

    Both arguments are bundle-relative POSIX paths; ``from_dir`` is the
    directory of the file that will contain the link. Bundle file names are
    slugs (``[a-z0-9-]``), so nothing here needs percent-encoding.
    """
    return posixpath.relpath(bundle_path, from_dir or ".")


def _escape_link_text(text: str) -> str:
    """Escape what would end the link text early, and fold newlines."""
    text = " ".join(str(text).split())
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def build_section(name: str, heading: str, items: list[tuple[str, str, str]]) -> str:
    """Render a marked section of Markdown links.

    ``items`` are ``(prefix, text, target)``. ``prefix`` is written before the
    link (for example a predicate name, which §6.1 says is how the text near a
    link tells the kind of relationship); pass ``""`` for none.
    """
    lines = [f"<!-- {name} -->", f"## {heading}", ""]
    for prefix, text, target in items:
        lead = f"{prefix}: " if prefix else ""
        lines.append(f"- {lead}[{_escape_link_text(text)}]({target})")
    lines.append(f"<!-- /{name} -->")
    return "\n".join(lines)


def append_section(content: str, section: str) -> str:
    """Append ``section`` to ``content`` with an unambiguous separator.

    The separator is a blank line, and is omitted only when ``content`` is
    empty. ``strip_section`` inverts exactly this, so any content -- including
    content ending in newlines -- comes back unchanged.
    """
    if not section:
        return content
    return f"{content}\n\n{section}" if content else section


def _section_pattern(name: str) -> re.Pattern[str]:
    # Anchored at the end of the body, and every line between the markers must
    # be a list item holding a Markdown link. Content that contains a heading
    # called "Mentions", or even a complete marked block followed by more text,
    # does not match.
    return re.compile(
        rf"(?:\n\n)?<!-- {re.escape(name)} -->\n## [^\n]+\n\n"
        rf"(?:- [^\n]*\]\([^\n]*\)\n)+"
        rf"<!-- /{re.escape(name)} -->\n*\Z"
    )


def strip_section(body: str, name: str) -> str:
    """Remove the trailing marked section ``name`` from ``body``, if present.

    The inverse of ``append_section``. Returns ``body`` unchanged when it does
    not END with a well-formed section of that name.
    """
    match = _section_pattern(name).search(body)
    if match is None:
        return body
    return body[: match.start()]
