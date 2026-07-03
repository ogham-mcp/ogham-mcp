"""TBU-159: dim-parameterized vector/halfvec columns across shipping schemas.

Covers the Python apply layer (`ogham.schema_apply.render_schema_sql`) --
the piece that substitutes the `:embedding_dim` psql placeholder for
callers that send SQL text via psycopg rather than the `psql` CLI.
"""

import subprocess
from pathlib import Path

import pytest

from ogham.schema_apply import render_schema_sql

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_FILES = [
    REPO_ROOT / "sql" / "schema.sql",
    REPO_ROOT / "sql" / "schema_postgres.sql",
    REPO_ROOT / "sql" / "schema_selfhost_supabase.sql",
]


# Last commit on main shipping literal vector(512)/halfvec(512) before the
# TBU-159 dim-parameterization landed (TBU-149, merged 2026-07-02). Pinned
# rather than "HEAD" so this characterization stays meaningful after the
# TBU-159 branch itself is committed and merged -- "HEAD" would then already
# be post-substitution and the comparison would be a no-op.
_PRE_TBU159_COMMIT = "775a6c2"


def _pre_tbu159_schema(schema_name: str) -> str:
    """Return the schema file's content at the pinned pre-TBU-159 commit."""
    result = subprocess.run(
        ["git", "show", f"{_PRE_TBU159_COMMIT}:sql/{schema_name}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"{_PRE_TBU159_COMMIT} not reachable in this checkout: {result.stderr.strip()}")
    return result.stdout


def _strip_comment_lines(sql: str) -> str:
    """Drop whole-line `--` comments and blank lines.

    TBU-159 added explanatory header comments (the psql dollar-quote gotcha,
    the sed-preprocessing instructions) that don't exist in the pre-refactor
    file and don't affect what gets executed. The characterization test below
    cares about executable-SQL equivalence, not comment-for-comment identity.
    """
    return "\n".join(
        line for line in sql.splitlines() if line.strip() and not line.strip().startswith("--")
    )


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
def test_render_at_512_matches_pre_tbu159_shipping_state(schema_path):
    """Characterization test: existing v0.15 installs (vector(512) hardcoded)
    must continue to work unchanged. Applying the new parameterized schema
    with embedding_dim=512 must produce the same *executable* SQL that
    shipped before TBU-159 (modulo added explanatory `--` comments) -- i.e.
    this refactor is a behavioral no-op at the default dim.
    """
    pre = _strip_comment_lines(_pre_tbu159_schema(schema_path.name))
    rendered = _strip_comment_lines(render_schema_sql(schema_path.read_text(), 512))
    assert rendered == pre


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("dim", [512, 1024, 3072])
def test_render_substitutes_every_placeholder(schema_path, dim):
    """No :embedding_dim placeholder should survive substitution, and the
    dim should appear in its place for every vector(...)/halfvec(...) site."""
    rendered = render_schema_sql(schema_path.read_text(), dim)
    assert ":embedding_dim" not in rendered
    assert f"vector({dim})" in rendered
    assert f"halfvec({dim})" in rendered


def test_render_schema_sql_rejects_non_positive_dim():
    with pytest.raises(ValueError, match="positive int"):
        render_schema_sql("vector(:embedding_dim)", 0)
    with pytest.raises(ValueError, match="positive int"):
        render_schema_sql("vector(:embedding_dim)", -5)


def test_render_schema_sql_rejects_none_with_value_error():
    """Code review follow-up: render_schema_sql(sql, None) used to raise a bare
    TypeError from int(None), contradicting the docstring's documented
    ValueError contract. None (and any other non-numeric value) must raise
    the same actionable ValueError as an out-of-range int."""
    with pytest.raises(ValueError, match="positive int"):
        render_schema_sql("vector(:embedding_dim)", None)  # type: ignore[arg-type]


# --- init_wizard._adjust_schema_dim wrapper ---


def test_adjust_schema_dim_coerces_string_dim():
    from ogham.init_wizard import _adjust_schema_dim

    assert "vector(1024)" in _adjust_schema_dim("vector(:embedding_dim)", "1024")


def test_adjust_schema_dim_falsy_string_falls_back_to_512():
    from ogham.init_wizard import _adjust_schema_dim

    assert "vector(512)" in _adjust_schema_dim("vector(:embedding_dim)", "")


def test_adjust_schema_dim_negative_dim_raises():
    """The wrapper itself still raises for a bad dim -- callers in
    init_wizard.py are responsible for catching this (see
    test_run_schema_negative_dim_does_not_crash in test_init_wizard.py)."""
    from ogham.init_wizard import _adjust_schema_dim

    with pytest.raises(ValueError, match="positive int"):
        _adjust_schema_dim("vector(:embedding_dim)", "-5")
