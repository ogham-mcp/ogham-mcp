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
