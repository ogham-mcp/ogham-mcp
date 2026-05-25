"""Anthropic Memory Tool 6-op conformance (issue #52)."""

from __future__ import annotations

import tempfile

import pytest

memory_tool_conformance = pytest.importorskip("memory_tool_conformance")
from memory_tool_conformance.conformance import run_conformance  # noqa: E402

from ogham.memory_tool import make_memory_contract  # noqa: E402


def test_memory_tool_six_op_conformance():
    with tempfile.TemporaryDirectory() as tmp:
        impl = make_memory_contract(root=tmp)
        report = run_conformance(impl, server_name="ogham-mcp")
        assert report.all_pass, report.render()
