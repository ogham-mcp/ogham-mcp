"""Backend protocol-conformance: both backends must satisfy EntityGraph structurally."""


def test_postgres_backend_conforms_to_protocol():
    from ogham.postgres.entity_graph import PostgresEntityGraph

    # runtime_checkable Protocol: structural check on class methods only
    # Instance we don't have without a pool -- check class hasattr instead.
    for method in ("store_triple", "query_join", "add_alias", "resolve_alias"):
        assert callable(getattr(PostgresEntityGraph, method, None)), (
            f"PostgresEntityGraph missing method {method}"
        )


def test_supabase_backend_conforms_to_protocol():
    from ogham.supabase.entity_graph import SupabaseEntityGraph

    for method in ("store_triple", "query_join", "add_alias", "resolve_alias"):
        assert callable(getattr(SupabaseEntityGraph, method, None)), (
            f"SupabaseEntityGraph missing method {method}"
        )


def test_postgres_backend_accepts_expected_ctor():
    """Constructor signature is a real load-bearing contract."""
    import inspect

    from ogham.postgres.entity_graph import PostgresEntityGraph

    sig = inspect.signature(PostgresEntityGraph.__init__)
    params = list(sig.parameters.keys())
    assert "pool" in params
    assert "allowed_predicates" in params


def test_supabase_backend_accepts_expected_ctor():
    import inspect

    from ogham.supabase.entity_graph import SupabaseEntityGraph

    sig = inspect.signature(SupabaseEntityGraph.__init__)
    params = list(sig.parameters.keys())
    assert "client" in params
    assert "allowed_predicates" in params


# --- DatabaseBackend contradiction lookups (TBU-217) ---
# Both reach the backend via cast(Any, get_backend()) in database.py, so
# nothing breaks today -- but a new backend can satisfy the Protocol while
# missing methods the retrieval path calls at runtime.

_CONTRADICTION_LOOKUPS = ("in_result_contradictions", "gap_out_of_result_contradictions")


def test_protocol_declares_the_contradiction_lookups():
    """The Protocol must describe the whole backend surface, not part of it."""
    from ogham.backends.protocol import DatabaseBackend

    for method in _CONTRADICTION_LOOKUPS:
        assert callable(getattr(DatabaseBackend, method, None)), (
            f"DatabaseBackend Protocol does not declare {method}"
        )


def test_every_backend_implements_the_contradiction_lookups():
    """Postgres implements both; Supabase and Gateway degrade deliberately."""
    from ogham.backends.gateway import GatewayBackend
    from ogham.backends.postgres import PostgresBackend
    from ogham.backends.supabase import SupabaseBackend

    for backend in (PostgresBackend, SupabaseBackend, GatewayBackend):
        for method in _CONTRADICTION_LOOKUPS:
            assert callable(getattr(backend, method, None)), (
                f"{backend.__name__} missing method {method}"
            )
