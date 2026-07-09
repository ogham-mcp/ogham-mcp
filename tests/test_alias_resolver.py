"""AliasResolver tests -- uses a fake EntityGraph in place of a real backend."""

from ogham.entity_graph import Entity


class _FakeGraph:
    def __init__(self, entities_by_name: dict[str, Entity], aliases: dict[tuple[str, str], int]):
        self._by_name = entities_by_name
        self._aliases = aliases  # (alias, profile) -> entity_id

    def store_triple(self, *args, **kwargs):
        raise NotImplementedError

    def query_join(self, *args, **kwargs):
        raise NotImplementedError

    def fetch_edge(self, *args, **kwargs):
        raise NotImplementedError

    def find_citing_edges(self, *args, **kwargs):
        raise NotImplementedError

    def add_alias(self, entity_id, alias, profile):
        self._aliases[(alias, profile)] = entity_id

    def resolve_alias(self, name_or_id, profile):
        if isinstance(name_or_id, int):
            for e in self._by_name.values():
                if e.id == name_or_id:
                    return e
            return None
        if name_or_id in self._by_name:
            return self._by_name[name_or_id]
        entity_id = self._aliases.get((name_or_id, profile))
        if entity_id is None:
            return None
        for e in self._by_name.values():
            if e.id == entity_id:
                return e
        return None


def _make_graph():
    entities = {
        "AuthService": Entity(id=1, canonical_name="AuthService", entity_type="service"),
        "LoginModule": Entity(id=2, canonical_name="LoginModule", entity_type="module"),
    }
    aliases = {("auth", "work"): 1}
    return _FakeGraph(entities, aliases)


def test_resolves_canonical_name():
    from ogham.alias_resolver import AliasResolver

    r = AliasResolver(_make_graph())
    e = r.resolve("AuthService", "work")
    assert e is not None
    assert e.id == 1


def test_resolves_int_id_passthrough():
    from ogham.alias_resolver import AliasResolver

    r = AliasResolver(_make_graph())
    e = r.resolve(1, "work")
    assert e is not None
    assert e.canonical_name == "AuthService"


def test_resolves_alias():
    from ogham.alias_resolver import AliasResolver

    r = AliasResolver(_make_graph())
    e = r.resolve("auth", "work")
    assert e is not None
    assert e.id == 1


def test_alias_isolation_by_profile():
    from ogham.alias_resolver import AliasResolver

    r = AliasResolver(_make_graph())
    assert r.resolve("auth", "personal") is None


def test_unresolvable_returns_none():
    from ogham.alias_resolver import AliasResolver

    r = AliasResolver(_make_graph())
    assert r.resolve("Bogus", "work") is None
