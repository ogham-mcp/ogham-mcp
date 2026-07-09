"""TBU-159: dim-parameterized vector/halfvec columns across shipping schemas.

Covers the Python apply layer (`ogham.schema_apply.render_schema_sql`) --
the piece that substitutes the `:embedding_dim` psql placeholder for
callers that send SQL text via psycopg rather than the `psql` CLI.

Note: a one-time characterization test (`test_render_at_512_matches_pre_tbu159_shipping_state`)
that diffed the rendered schema against a frozen pre-TBU-159 commit was
retired post-TBU-159 -- it proved the dim-parameterization refactor was a
behavioral no-op at the time, but a frozen baseline can never accommodate
legitimate schema growth after that point (e.g. TBU-129's predicate URI
columns). Dim-substitution correctness is still guarded going forward by
`test_render_substitutes_every_placeholder` below and by
`tests/test_schema_parity.py::test_no_hardcoded_vector_dim_in_shipping_schemas`.
"""

from pathlib import Path

import pytest

from ogham.schema_apply import render_schema_sql

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_FILES = [
    REPO_ROOT / "sql" / "schema.sql",
    REPO_ROOT / "sql" / "schema_postgres.sql",
    REPO_ROOT / "sql" / "schema_selfhost_supabase.sql",
]


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
