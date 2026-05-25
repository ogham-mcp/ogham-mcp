"""Anthropic Memory Tool 6-op contract (filesystem-backed).

Ogham's primary MCP surface is Postgres hybrid search. This package exposes a
spec-conformant filesystem memory layer for clients using context-editing memory
tools; CI validates it with `memory-tool-conformance`.
"""

from ogham.memory_tool.contract import make_memory_contract

__all__ = ["make_memory_contract"]
