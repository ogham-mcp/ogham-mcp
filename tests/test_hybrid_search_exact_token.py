"""Can hybrid search return a memory by a token only that memory contains?

An exact rare token -- a ticket id, a binary name, a product name -- is the
highest-precision query a user can type, and disproportionately what an agent
issues when checking "did we already decide this?". TBU-244: the store held the
only row containing `TBU-243` and `ogham search "TBU-243"` did not return it in
the top 20; it appeared at rank 24 when the caller asked for 50.

There was no test asserting recall of a known ground truth, which is why it went
unnoticed. This is that test.

The setup is deliberately arithmetic rather than lifelike: embeddings are
constructed so the semantic ranking is exact and reproducible, with the needle
placed at the WORST possible semantic position. That isolates the fusion --
nothing here depends on an embedding provider, a model, or a corpus.

Why it fails today (RRF, rrf_k=10, semantic 0.7 / full-text 0.3, match_count=5):

    semantic rank 1, no keyword hit   0.7/(10+1)  + 0.3/(10+15) = 0.0756
    keyword rank 1, no semantic hit   0.7/(10+15) + 0.3/(10+1)  = 0.0553

The semantic leg is a nearest-neighbour scan, so it ALWAYS returns
`match_count * 3` rows -- everything has some cosine distance. The keyword leg
returns only genuine matches. Weighting a dense leg at 0.7 against a sparse one
at 0.3 means one exact lexical hit cannot outrank a full slate of merely-plausible
neighbours: the needle loses to semantic ranks 1-6 by construction.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCHEMA = REPO_ROOT / "sql" / "schema_postgres.sql"

EMBEDDING_DIM = 512
FILLER_ROWS = 39
# Long enough not to stem into anything, rare enough to appear exactly once.
NONCE = "ZQXJV7731"


def _unit(index: int) -> list[float]:
    """One-hot embedding. Distinct index => orthogonal => cosine 0 to each other."""
    vec = [0.0] * EMBEDDING_DIM
    vec[index] = 1.0
    return vec


def _query_vector() -> list[float]:
    """Weights 1/(i+1) over the filler positions, and ZERO on the needle's.

    Cosine to filler i is proportional to 1/(i+1), strictly decreasing, so the
    semantic ranking is total and deterministic: 0, 1, 2, ... The needle sits at
    an index the query never touches, so it is the worst semantic match in the
    corpus -- the hardest case, and the one users actually hit.
    """
    vec = [0.0] * EMBEDDING_DIM
    for i in range(FILLER_ROWS):
        vec[i] = 1.0 / (i + 1)
    return vec


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


@pytest.fixture
def seeded_db(pg_url):
    """Throwaway database with a corpus whose only NONCE-bearing row is the
    worst semantic match. Creates and drops its own database."""
    import psycopg

    from ogham.schema_apply import render_schema_sql

    name = f"ogham_recall_{uuid.uuid4().hex[:8]}"
    base = pg_url.rsplit("/", 1)[0]
    admin_url, db_url = f"{base}/postgres", f"{base}/{name}"

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')  # type: ignore[arg-type]
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute(render_schema_sql(SCHEMA.read_text(), EMBEDDING_DIM))  # type: ignore[arg-type]
            for i in range(FILLER_ROWS):
                conn.execute(
                    "INSERT INTO memories (content, profile, embedding) VALUES (%s, 't', %s)",
                    (
                        f"filler memory number {i} about deployment and indexing",
                        _vec_literal(_unit(i)),
                    ),
                )
            conn.execute(
                "INSERT INTO memories (content, profile, embedding) VALUES (%s, 't', %s)",
                (f"the decision recorded under {NONCE}", _vec_literal(_unit(500))),
            )
            yield conn, db_url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')  # type: ignore[arg-type]


def _search(conn, limit: int) -> list[str]:
    rows = conn.execute(
        "SELECT content FROM hybrid_search_memories(%s, %s::vector, %s, 't')",
        (NONCE, _vec_literal(_query_vector()), limit),
    ).fetchall()
    return [r[0] for r in rows]


@pytest.mark.postgres_integration
def test_the_corpus_is_set_up_as_intended(seeded_db):
    """Guard the instrument. If the needle were not unique, or the keyword leg
    could not see it, the real test below would fail for the wrong reason."""
    conn, _ = seeded_db
    unique = conn.execute(
        "SELECT count(*) FROM memories WHERE profile='t' AND content LIKE %s", (f"%{NONCE}%",)
    ).fetchone()
    assert unique is not None and unique[0] == 1, "needle must appear exactly once"

    lexical = conn.execute(
        "SELECT count(*) FROM memories WHERE profile='t' AND fts @@ websearch_to_tsquery(%s)",
        (NONCE,),
    ).fetchone()
    assert lexical is not None and lexical[0] == 1, (
        "the tsvector must match the needle -- if this fails the tokeniser, not "
        "the fusion, is the problem"
    )


def _needle_rank(conn, limit: int) -> int | None:
    for position, content in enumerate(_search(conn, limit), start=1):
        if NONCE in content:
            return position
    return None


# strict=True on purpose: when the fusion is fixed these turn XPASS and FAIL,
# which forces the marker off rather than letting a fixed defect sit here
# silently marked broken.
@pytest.mark.xfail(strict=True, reason="TBU-244: RRF buries the only exact match")
@pytest.mark.postgres_integration
def test_exact_rare_token_is_returned_first(seeded_db):
    """TBU-244. The only row containing the token must come back, ranked first.

    Rank 1 rather than merely 'present', because for a query that IS a unique
    identifier there is no plausible better answer.

    Measured 2026-08-17: absent from the top 5 entirely. All five returned rows
    are filler containing none of the query token.
    """
    conn, _ = seeded_db
    results = _search(conn, 5)
    assert results, "search returned nothing at all"
    assert NONCE in results[0], (
        f"the only memory containing {NONCE} did not rank first.\ngot: {[r[:60] for r in results]}"
    )


@pytest.mark.xfail(strict=True, reason="TBU-244: absent-leg penalty scales with match_count")
@pytest.mark.postgres_integration
def test_needle_rank_is_stable_across_requested_limits(seeded_db):
    """A row's rank should be a property of the query and the corpus, not of the
    page size. It is not: the absent-leg penalty is
    `coalesce(rank_ix, match_count * 3)`, so every score moves when the caller
    changes `--limit`.

    Measured on this 40-row corpus: absent at 5, rank 9 at 10 and 20, rank 8 at
    40 and 50. TBU-244 saw the same shape on the live store -- absent at 20,
    24th at 50.
    """
    conn, _ = seeded_db
    ranks = {limit: _needle_rank(conn, limit) for limit in (10, 20, 40)}
    assert len(set(ranks.values())) == 1, (
        f"the same row ranked differently depending only on the limit asked for: {ranks}"
    )
