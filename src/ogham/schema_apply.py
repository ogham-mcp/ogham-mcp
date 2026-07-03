"""Shared helper for applying shipping SQL schema files at a configured dim.

TBU-159: the three shipping schemas (``sql/schema.sql``,
``sql/schema_postgres.sql``, ``sql/schema_selfhost_supabase.sql``) declare
``vector``/``halfvec`` columns as ``vector(:embedding_dim)`` -- a psql ``-v``
variable placeholder (Design Council Option A, 2026-07-02). Running the
schema via the actual ``psql`` CLI (``psql -f schema.sql -v
embedding_dim=1024``) substitutes this natively. Callers that instead send
the SQL text straight to the server via psycopg (which has no psql variable
support) must pre-substitute the placeholder themselves -- that's what this
module does.

``embedding_dim`` is always an already-validated positive int by the time it
reaches here (``Settings.embedding_dim`` / the wizard's own prompt
validation), so a plain string substitution is safe: there is no untrusted
string ever interpolated into the SQL text, unlike an f-string built from
raw user input.
"""

_PLACEHOLDER = ":embedding_dim"


def render_schema_sql(sql_text: str, embedding_dim: int) -> str:
    """Return ``sql_text`` with every ``:embedding_dim`` placeholder replaced.

    Raises ValueError if ``embedding_dim`` isn't a positive int (including
    ``None`` or any other non-numeric value) -- catches a misconfigured
    caller before it sends malformed DDL to the database.
    """
    try:
        dim = int(embedding_dim)
    except (TypeError, ValueError) as e:
        raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}") from e
    if dim <= 0:
        raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}")
    return sql_text.replace(_PLACEHOLDER, str(dim))
