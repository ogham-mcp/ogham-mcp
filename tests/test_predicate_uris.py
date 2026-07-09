import re
from pathlib import Path

from typer.testing import CliRunner

from ogham.cli import app
from ogham.entity_graph import PREDICATE_URIS, V1_PREDICATES
from ogham.tools.entity_graph import describe_predicates_impl

_SCHEMA_ORG = {
    "OWNS": "https://schema.org/owns",
    "OWNED_BY": "https://schema.org/owner",
    "MENTIONS": "https://schema.org/mentions",
    "PART_OF": "https://schema.org/isPartOf",
    "CONTAINS": "https://schema.org/hasPart",
}

# Per-row seed-line regex (matches tests/test_schema_parity.py's guard):
# captures a predicate's ogham_uri and schema_org_uri (or None) from ITS
# OWN seed row, so a transposition between two rows (e.g. OWNS/OWNED_BY
# swapping owns/owner) fails instead of passing on a bare substring check.
_PREDICATE_ROW_URI_RE = (
    r"\('{pred}',.*?'(https://ogham-mcp\.dev/vocab#[^']*)',"
    r"\s*(?:'(https://schema\.org/[^']*)'|NULL),\s*NULL\)"
)


def test_predicate_uris_cover_vocab():
    assert set(PREDICATE_URIS) == V1_PREDICATES


def test_ogham_uri_for_every_predicate():
    for pred, uris in PREDICATE_URIS.items():
        assert uris["ogham"] == f"https://ogham-mcp.dev/vocab#{pred}"
        assert uris["iirds"] is None  # gated on TBU-128


def test_schema_org_uris_exactly_the_five():
    mapped = {p: u["schema_org"] for p, u in PREDICATE_URIS.items() if u["schema_org"]}
    assert mapped == _SCHEMA_ORG


def test_python_uris_match_sql_seed():
    """The Python mirror must equal the shipped schema seed (drift guard).

    Per-row check (not a file-wide substring check): a transposition of two
    predicates' URIs in either the Python constant or the SQL seed must fail
    this test, not pass because the string happens to appear elsewhere in
    the file.
    """
    sql = (Path(__file__).resolve().parents[1] / "sql" / "schema_postgres.sql").read_text()
    for pred, uris in PREDICATE_URIS.items():
        pattern = _PREDICATE_ROW_URI_RE.format(pred=re.escape(pred))
        match = re.search(pattern, sql)
        assert match, f"seed row for '{pred}' not found (or malformed) in schema_postgres.sql"
        sql_ogham_uri, sql_schema_org_uri = match.group(1), match.group(2)
        assert uris["ogham"] == sql_ogham_uri, (
            f"{pred} ogham_uri mismatch: Python={uris['ogham']!r} SQL={sql_ogham_uri!r}"
        )
        assert uris["schema_org"] == sql_schema_org_uri, (
            f"{pred} schema_org_uri mismatch: "
            f"Python={uris['schema_org']!r} SQL={sql_schema_org_uri!r}"
        )


def test_describe_predicates_shape():
    rows = describe_predicates_impl(uris=PREDICATE_URIS)
    assert len(rows) == len(V1_PREDICATES)
    assert [r["predicate"] for r in rows] == sorted(V1_PREDICATES)  # deterministic order
    keys = set(rows[0])
    assert keys == {"predicate", "ogham_uri", "schema_org_uri", "iirds_uri"}


def test_describe_predicates_values():
    rows = {r["predicate"]: r for r in describe_predicates_impl(uris=PREDICATE_URIS)}
    assert rows["PART_OF"]["schema_org_uri"] == "https://schema.org/isPartOf"
    assert rows["PART_OF"]["ogham_uri"] == "https://ogham-mcp.dev/vocab#PART_OF"
    assert rows["DEPENDS_ON"]["schema_org_uri"] is None  # unmapped -> None
    assert rows["DEPENDS_ON"]["iirds_uri"] is None


def test_cli_predicates_lists_uris():
    result = CliRunner().invoke(app, ["predicates"])
    assert result.exit_code == 0, result.output
    assert "https://ogham-mcp.dev/vocab#PART_OF" in result.output
    assert "https://schema.org/isPartOf" in result.output
    assert "DEPENDS_ON" in result.output
    assert "iirds=-" in result.output  # all 16 predicates are iirds_uri=None (TBU-128)
