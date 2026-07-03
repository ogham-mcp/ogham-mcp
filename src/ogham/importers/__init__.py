"""Issue-tracker importers -- fetch external tickets, map to Ogham memory shape.

Each importer exposes a client for its tracker's API and a pure
``map_*_to_memory`` function that returns ``{content, metadata, tags}``
suitable for ``store_memory``. Read-only against the tracker -- importers
never write back.
"""
