"""
Tests for Conv 2.41 / KE-028 upstream fix: Annotated[T|None, BeforeValidator(coerce)]
coercion wrappers on tool parameters in ogham/tools/memory.py.

Drop this into the upstream tests/ directory (or ogham/tests/ — check upstream layout).
Adjust imports once you've confirmed upstream's exact module path.

Run: pytest tests/test_list_coercion.py -v
"""

import json
import pytest
from pydantic import ValidationError, TypeAdapter

# Adjust this import after clone + reading upstream layout:
from ogham.tools.memory import (
    _coerce_list,
    _coerce_dict,
    ListStr,
    DictAny,
)


class TestCoerceList:
    def test_native_list_unchanged(self):
        assert _coerce_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_empty_list_unchanged(self):
        assert _coerce_list([]) == []

    def test_none_passes_through(self):
        assert _coerce_list(None) is None

    def test_json_string_list_coerced(self):
        assert _coerce_list('["a","b","c"]') == ["a", "b", "c"]

    def test_empty_json_array_coerced(self):
        assert _coerce_list("[]") == []

    def test_bare_string_wraps_as_single_element(self):
        # Fallback: a bare string that isn't JSON becomes a 1-element list.
        # This is the pragmatic behavior for tag-like fields.
        assert _coerce_list("foo") == ["foo"]

    def test_invalid_json_wraps_as_single_element(self):
        # Not valid JSON → fallback to [v]
        assert _coerce_list("not{valid]json") == ["not{valid]json"]

    def test_json_string_that_parses_to_dict_rejected(self):
        # A JSON-string that parses to dict should NOT be wrapped as list.
        # Current behavior: fallback wraps as [original_string]. Alternative: raise.
        # Document whichever the maintainer prefers; test reflects current behavior.
        result = _coerce_list('{"a":"b"}')
        assert result == ['{"a":"b"}']

    def test_int_input_raises(self):
        with pytest.raises(TypeError):
            _coerce_list(42)


class TestCoerceDict:
    def test_native_dict_unchanged(self):
        assert _coerce_dict({"a": 1}) == {"a": 1}

    def test_empty_dict_unchanged(self):
        assert _coerce_dict({}) == {}

    def test_none_passes_through(self):
        assert _coerce_dict(None) is None

    def test_json_string_dict_coerced(self):
        assert _coerce_dict('{"a":1,"b":"x"}') == {"a": 1, "b": "x"}

    def test_invalid_json_raises(self):
        # For dicts, we don't have a reasonable fallback — fail loudly.
        with pytest.raises((TypeError, ValueError)):
            _coerce_dict("not{valid]json")

    def test_list_as_string_raises(self):
        with pytest.raises((TypeError, ValueError)):
            _coerce_dict('["a","b"]')


class TestListStrAnnotated:
    """Via Pydantic TypeAdapter — mirrors how FastMCP validates tool params."""

    def setup_method(self):
        self.adapter = TypeAdapter(ListStr)

    def test_native_list_validates(self):
        assert self.adapter.validate_python(["a", "b"]) == ["a", "b"]

    def test_json_string_list_validates(self):
        assert self.adapter.validate_python('["a","b"]') == ["a", "b"]

    def test_none_validates(self):
        assert self.adapter.validate_python(None) is None

    def test_list_of_ints_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python([1, 2, 3])


class TestDictAnyAnnotated:
    def setup_method(self):
        self.adapter = TypeAdapter(DictAny)

    def test_native_dict_validates(self):
        assert self.adapter.validate_python({"k": "v"}) == {"k": "v"}

    def test_json_string_dict_validates(self):
        assert self.adapter.validate_python('{"k":"v"}') == {"k": "v"}

    def test_none_validates(self):
        assert self.adapter.validate_python(None) is None


class TestRegressionStoreMemorySignature:
    """
    Smoke test to ensure the store_memory tool accepts both input shapes.
    Requires a fixture for the ogham store backend. Skip if no test store is wired.
    """

    @pytest.mark.skip(reason="Requires ogham store fixture; enable once wired.")
    def test_store_memory_accepts_native_list(self, store):
        # Placeholder for integration-style test against the actual tool.
        pass

    @pytest.mark.skip(reason="Requires ogham store fixture; enable once wired.")
    def test_store_memory_accepts_stringified_list(self, store):
        pass
