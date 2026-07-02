"""Gemini L2-normalization parity tests.

`gemini-embedding-2` (GA) returns pre-normalized vectors at every output
dim -- verified empirically 2026-07-02 across 512/768/1536/3072 (||v||
within 2e-7 of 1.0). The client-side normalize is now redundant work
for this model.

Older / preview aliases (e.g. `gemini-embedding-001`) are treated as
not-pre-normalized -- we keep the defensive normalize path for them so
retrieval quality doesn't silently drift if someone pins an old model.

Mirror tests to internal/native/embedding_test.go in ogham-cli.
"""

from unittest.mock import patch

from ogham.embeddings import _embed_gemini, _l2_normalize


def test_l2_normalize_unit_length():
    """|{3, 4}| = 5, normalized = {0.6, 0.8}, sum(v^2) = 1."""
    got = _l2_normalize([3.0, 4.0])
    assert abs(got[0] - 0.6) < 1e-9
    assert abs(got[1] - 0.8) < 1e-9
    assert abs(sum(x * x for x in got) - 1.0) < 1e-9


def test_l2_normalize_zero_vector_passthrough():
    """Zero vector returns unchanged -- normalizing would divide by zero."""
    got = _l2_normalize([0.0, 0.0, 0.0])
    assert got == [0.0, 0.0, 0.0]


def test_l2_normalize_already_unit():
    """Already-unit vector comes back unchanged within float tolerance."""
    import math

    unit = [1.0 / math.sqrt(4)] * 4  # sum(v^2) = 1
    got = _l2_normalize(unit)
    for a, b in zip(got, unit):
        assert abs(a - b) < 1e-9


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]
        self.usage_metadata = None


def _fake_client(vectors):
    class _FakeModels:
        def embed_content(self, **kwargs):
            return _FakeResponse(vectors)

    class _FakeClient:
        def __init__(self):
            self.models = _FakeModels()

    return _FakeClient()


def test_embed_gemini_skips_normalize_for_ga_model_at_sub_3072(monkeypatch):
    """`gemini-embedding-2` returns pre-normalized vectors; the client
    must NOT re-normalize (would waste an O(n) pass and introduce fp error)."""
    from ogham import embeddings as emb_mod

    monkeypatch.setattr(emb_mod.settings, "gemini_api_key", "fake")
    monkeypatch.setattr(emb_mod.settings, "embedding_dim", 512)
    monkeypatch.setattr(emb_mod.settings, "gemini_embed_model", "gemini-embedding-2")

    # Non-unit magnitude on purpose: if we normalize, this test fails.
    raw = [float(i + 1) for i in range(512)]
    with patch.object(emb_mod, "_get_gemini_client", return_value=_fake_client([raw])):
        got = _embed_gemini("hello")

    # Server value passes through untouched (1.0 not 1.0/norm).
    assert got[0] == 1.0
    assert got == raw


def test_embed_gemini_normalizes_sub_3072_for_non_ga_model(monkeypatch):
    """For older / preview aliases we keep the defensive normalize -- we
    haven't verified their server-side normalization behavior."""
    from ogham import embeddings as emb_mod

    monkeypatch.setattr(emb_mod.settings, "gemini_api_key", "fake")
    monkeypatch.setattr(emb_mod.settings, "embedding_dim", 512)
    monkeypatch.setattr(emb_mod.settings, "gemini_embed_model", "gemini-embedding-001")

    raw = [float(i + 1) for i in range(512)]
    with patch.object(emb_mod, "_get_gemini_client", return_value=_fake_client([raw])):
        got = _embed_gemini("hello")

    assert len(got) == 512
    sum_sq = sum(x * x for x in got)
    assert abs(sum_sq - 1.0) < 1e-3, f"sum(v^2) = {sum_sq}, want within 1e-3 of 1.0"


def test_embed_gemini_does_not_normalize_at_3072(monkeypatch):
    """At native 3072 dim Gemini already returns unit vectors; don't touch,
    regardless of model alias."""
    from ogham import embeddings as emb_mod

    monkeypatch.setattr(emb_mod.settings, "gemini_api_key", "fake")
    monkeypatch.setattr(emb_mod.settings, "embedding_dim", 3072)
    monkeypatch.setattr(emb_mod.settings, "gemini_embed_model", "gemini-embedding-2")

    # Non-unit magnitude on purpose: if we normalize, this test fails.
    raw = [float(i + 1) for i in range(3072)]
    with patch.object(emb_mod, "_get_gemini_client", return_value=_fake_client([raw])):
        got = _embed_gemini("hello")

    # Raw first element passes through untouched (1.0 not 1.0/norm).
    assert got[0] == 1.0
