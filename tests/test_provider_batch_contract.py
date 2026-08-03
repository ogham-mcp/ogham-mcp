"""Every provider batch path must return one embedding per input, in order.

0.17.2 added `_assert_full_batch` to all six providers, but only Gemini and
OpenAI had any test touching their batch function. `_embed_mistral_batch`,
`_embed_voyage_batch`, `_embed_ollama_batch` and `_embed_onnx_batch` had zero
references in the suite -- and `mistralai` and `voyageai` are not installed in
the dev environment, so those branches were never even imported, let alone
executed. Four of the six edits shipped unverified.

They turned out to be correct. That is luck, not evidence, and the next edit to
that file gets no warning at all. These tests supply the missing evidence with
stubbed clients, so no provider SDK needs installing.

The failure being guarded is not cosmetic: results are paired with their input
texts BY POSITION and then cached under each text's key, so a provider that
returns fewer embeddings than it was given would pair every later text with its
neighbour's vector -- silently, and cached.
"""

from unittest.mock import patch

import pytest

import ogham.embeddings as emb

VEC = [0.1] * 512


class _Item:
    """openai / mistral response element: a vector plus its index."""

    def __init__(self, values, index):
        self.embedding = values
        self.index = index


class _DataResp:
    def __init__(self, n):
        self.data = [_Item(VEC, i) for i in range(n)]
        self.usage = None


class _VoyageResp:
    def __init__(self, n):
        self.embeddings = [VEC] * n
        self.total_tokens = 1


def _client(attr, method, fn):
    """Build a stub client exposing `client.<attr>.<method>` or `client.<method>`."""
    if attr is None:
        return type("C", (), {method: staticmethod(fn)})()
    inner = type("Inner", (), {method: staticmethod(fn)})()
    return type("C", (), {attr: inner})()


def _settings():
    return patch.multiple(
        emb.settings,
        embedding_dim=512,
        openai_api_key="k",
        mistral_api_key="k",
        voyage_api_key="k",
        mistral_embed_model="m",
        voyage_embed_model="v",
        ollama_embed_model="o",
    )


# (label, function, patch target, stub factory taking a count)
PROVIDERS = [
    (
        "openai",
        "_embed_openai_batch",
        "_get_openai_client",
        lambda n: _client("embeddings", "create", lambda **k: _DataResp(n)),
    ),
    (
        "mistral",
        "_embed_mistral_batch",
        "_get_mistral_client",
        lambda n: _client("embeddings", "create", lambda **k: _DataResp(n)),
    ),
    (
        "voyage",
        "_embed_voyage_batch",
        "_get_voyage_client",
        lambda n: _client(None, "embed", lambda **k: _VoyageResp(n)),
    ),
    (
        "ollama",
        "_embed_ollama_batch",
        "_get_ollama_client",
        lambda n: _client(None, "embed", lambda **k: {"embeddings": [VEC] * n}),
    ),
]


@pytest.mark.parametrize("label,fn_name,target,stub", PROVIDERS, ids=[p[0] for p in PROVIDERS])
def test_batch_returns_one_embedding_per_input(label, fn_name, target, stub, monkeypatch):
    monkeypatch.setattr(emb, target, lambda: stub(3))
    with _settings():
        out = getattr(emb, fn_name)(["a", "b", "c"])
    assert len(out) == 3, f"{label} returned {len(out)} embeddings for 3 inputs"


@pytest.mark.parametrize("label,fn_name,target,stub", PROVIDERS, ids=[p[0] for p in PROVIDERS])
def test_short_batch_raises_rather_than_truncating(label, fn_name, target, stub, monkeypatch):
    """The whole point: refuse, do not silently pair texts with wrong vectors.

    Two layers can catch this and which one fires depends on the provider.
    openai and mistral carry a per-item index, so `_order_by_index` sees
    indices [0, 1] for 3 inputs and refuses first; voyage and ollama return
    bare lists, so `_assert_full_batch` is the one that fires. Either message
    is a pass -- what must never happen is a truncated list coming back.
    """
    monkeypatch.setattr(emb, target, lambda: stub(2))
    with _settings(), pytest.raises(RuntimeError, match="refusing to (pair|guess)"):
        getattr(emb, fn_name)(["a", "b", "c"])


def test_onnx_batch_is_one_call_per_text_so_cannot_go_short(monkeypatch):
    """onnx builds the list itself; the assertion is belt-and-braces."""
    monkeypatch.setattr(emb, "_embed_onnx", lambda t, usage_out=None: VEC)
    with _settings():
        assert len(emb._embed_onnx_batch(["a", "b", "c"])) == 3


def test_every_provider_batch_function_is_covered_here():
    """If a seventh provider is added, this test fails until it is listed above."""
    import inspect

    declared = {
        n
        for n, _ in inspect.getmembers(emb, inspect.isfunction)
        if n.startswith("_embed_") and n.endswith("_batch")
    }
    covered = {p[1] for p in PROVIDERS} | {"_embed_onnx_batch", "_embed_gemini_batch"}
    assert declared <= covered, f"untested provider batch paths: {sorted(declared - covered)}"
