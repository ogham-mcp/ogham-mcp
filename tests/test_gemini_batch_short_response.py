"""Regression tests for OM-mcp issue #60.

Gemini's batchEmbedContents endpoint occasionally returns HTTP 200 with
fewer embeddings than items submitted. Before the fix, `_embed_gemini_batch`
silently truncated via zip() and `generate_embeddings_batch` raised
'Embedding batch completed with missing results' AFTER the API call --
no retry. These tests pin the retry semantics.
"""

from unittest.mock import patch

import pytest

from ogham.embeddings import _embed_gemini_batch


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]
        self.usage_metadata = None


class _ShortThenFullClient:
    """First call returns N-1 embeddings, subsequent calls return N."""

    def __init__(self, full_vectors):
        self._full = full_vectors
        self.calls = 0

        class _Models:
            def embed_content(inner_self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return _FakeResponse(self._full[:-1])
                return _FakeResponse(self._full)

        self.models = _Models()


class _AlwaysShortClient:
    """Every call returns one fewer embedding than requested."""

    def __init__(self, full_vectors):
        self._full = full_vectors
        self.calls = 0

        class _Models:
            def embed_content(inner_self, **kwargs):
                self.calls += 1
                return _FakeResponse(self._full[:-1])

        self.models = _Models()


_VECTORS_3 = [
    [0.1] * 512,
    [0.2] * 512,
    [0.3] * 512,
]


def _settings_with_gemini():
    """Patch settings to make the function entrypoint runnable in tests."""
    from ogham import embeddings as emb_mod

    return patch.multiple(
        emb_mod.settings,
        gemini_api_key="test-key",
        embedding_dim=512,
        gemini_embed_model="gemini-embedding-2",
    )


def test_short_response_triggers_retry_then_succeeds(monkeypatch):
    """A short response on the first call must retry and succeed on the second."""
    client = _ShortThenFullClient(_VECTORS_3)
    from ogham import embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    # Shrink retry wait so the test doesn't burn 3+ seconds on backoff
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini():
        out = _embed_gemini_batch(["a", "b", "c"])

    assert client.calls == 2
    assert len(out) == 3
    assert all(len(v) == 512 for v in out)


def test_persistently_wrong_count_still_raises(monkeypatch):
    """Never silently accept a mismatched response, even after falling back.

    This fake returns a count unrelated to the input size, so the per-input
    fallback cannot rescue it either. Pairing a text with the wrong vector is
    worse than an error, so the only acceptable outcome is a raise.
    """
    client = _AlwaysShortClient(_VECTORS_3)
    from ogham import embeddings as emb_mod

    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini(), pytest.raises(RuntimeError) as exc_info:
        _embed_gemini_batch(["a", "b", "c"])

    assert "embeddings for" in str(exc_info.value)


# --- TBU-208: a provider that deterministically ignores batching -----------


class _IgnoresBatchingClient:
    """Mimics gemini-embedding-2 from 2026-07-30: always ONE embedding.

    Verified against the live API -- batches of 2, 3 and 5 each returned a
    single embedding, while a batch of 1 returned one correctly. Retrying
    cannot win against this, because it is not a flake.
    """

    def __init__(self, vector_for):
        self.batch_sizes = []
        self._vector_for = vector_for

        class _Models:
            def embed_content(inner_self, **kwargs):
                contents = kwargs["contents"]
                self.batch_sizes.append(len(contents))
                return _FakeResponse([self._vector_for(contents[0])])

        self.models = _Models()


def _vector_for(text: str) -> list[float]:
    """Distinct, recoverable vector per input, so ordering bugs are visible."""
    return [float(ord(text[0]))] * 512


def test_provider_that_ignores_batching_falls_back_to_single_calls(monkeypatch):
    """The real TBU-208 case: degrade to per-input rather than fail the caller."""
    client = _IgnoresBatchingClient(_vector_for)
    from ogham import embeddings as emb_mod

    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini():
        out = _embed_gemini_batch(["a", "b", "c"])

    assert len(out) == 3
    # Order must survive the fallback, or text pairs with the wrong vector.
    assert [v[0] for v in out] == [float(ord(c)) for c in "abc"]
    # Probed the batch twice, then one call per input.
    assert client.batch_sizes == [3, 3, 1, 1, 1]


def test_batch_incapability_is_remembered_for_the_process(monkeypatch):
    """Having learned the model does not batch, stop paying for the probe.

    Without this, a 124k-row ingest re-discovers the same fact on every batch.
    """
    client = _IgnoresBatchingClient(_vector_for)
    from ogham import embeddings as emb_mod

    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini():
        _embed_gemini_batch(["a", "b", "c"])
        client.batch_sizes.clear()
        out = _embed_gemini_batch(["d", "e"])

    assert len(out) == 2
    assert client.batch_sizes == [1, 1], "second call should not re-probe the batch"


def test_reset_restores_probing(monkeypatch):
    """The negative of the memo: a cleared cache probes again."""
    client = _IgnoresBatchingClient(_vector_for)
    from ogham import embeddings as emb_mod

    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini():
        _embed_gemini_batch(["a", "b"])
        emb_mod._reset_gemini_batch_support()
        client.batch_sizes.clear()
        _embed_gemini_batch(["c", "d"])

    assert client.batch_sizes[0] == 2, "after reset the batch should be tried again"


def test_a_working_provider_never_falls_back(monkeypatch):
    """The fix must not cost N calls where one would do."""
    client = _ShortThenFullClient(_VECTORS_3)
    from ogham import embeddings as emb_mod

    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini():
        out = _embed_gemini_batch(["a", "b", "c"])

    assert len(out) == 3
    assert client.calls == 2, "one flake, one success -- no per-input fallback"
    assert "gemini-embedding-2" not in emb_mod._GEMINI_NO_BATCH


def test_single_input_failure_raises_immediately(monkeypatch):
    """With one input there is nothing to degrade to, so do not loop pointlessly."""
    from ogham import embeddings as emb_mod

    class _Empty:
        def __init__(self):
            self.calls = 0

            class _Models:
                def embed_content(inner_self, **kwargs):
                    self.calls += 1
                    return _FakeResponse([])

            self.models = _Models()

    client = _Empty()
    emb_mod._reset_gemini_batch_support()
    monkeypatch.setattr(emb_mod, "_get_gemini_client", lambda: client)
    monkeypatch.setenv("OGHAM_RETRY_FAST", "1")

    with _settings_with_gemini(), pytest.raises(RuntimeError):
        _embed_gemini_batch(["only"])

    assert client.calls == 1


def test_default_gemini_model_is_ga_alias():
    """Default config must point at the GA alias `gemini-embedding-2`,
    not the soon-to-be-retired `-preview` alias."""
    from ogham.config import Settings

    fresh = Settings(supabase_url="http://x", supabase_key="x")
    assert fresh.gemini_embed_model == "gemini-embedding-2"


# --- the same hole in every other provider (audit follow-up to TBU-208) -----


def test_every_provider_batch_asserts_the_count():
    """Only Gemini checked this until 2026-07-30, because issue #60 forced it.

    The caller pairs texts to embeddings positionally with zip() and caches the
    result under each text's key, so a provider that drops one item from the
    middle mispairs everything after it -- and the mispairing is cached.
    """
    import inspect

    from ogham import embeddings as emb_mod

    for provider in ("ollama", "openai", "mistral", "voyage", "onnx"):
        fn = getattr(emb_mod, f"_embed_{provider}_batch")
        src = inspect.getsource(fn)
        assert "_assert_full_batch(" in src, f"{provider} batch does not validate the count"


def test_assert_full_batch_rejects_short_and_long():
    from ogham.embeddings import _assert_full_batch

    _assert_full_batch("p", ["a", "b"], [[1.0], [2.0]])  # exact -> no raise
    for bad in ([[1.0]], [[1.0], [2.0], [3.0]]):
        with pytest.raises(RuntimeError, match="refusing to pair"):
            _assert_full_batch("p", ["a", "b"], bad)


def test_dead_transient_predicate_is_gone():
    """It claimed a short response was transient, which is no longer the policy.

    A stale predicate that contradicts the live retry rule is exactly the kind
    of artefact that makes the next reader trust the wrong thing.
    """
    from ogham import embeddings as emb_mod

    assert not hasattr(emb_mod, "_is_transient_gemini_error")


# --- TBU-209: request order must survive the response ----------------------


class _Indexed:
    """Mimics an OpenAI/Mistral embedding item: a vector plus its index."""

    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


def test_out_of_order_response_is_restored_to_request_order():
    """The failure this prevents is silent: each text gets its neighbour's vector.

    OpenAI models `index` as a REQUIRED field precisely because position in
    `data` is not the contract. Before this, we read the array as it arrived.
    """
    from ogham.embeddings import _order_by_index

    shuffled = [_Indexed(["c"], 2), _Indexed(["a"], 0), _Indexed(["b"], 1)]
    assert [i.embedding[0] for i in _order_by_index("p", shuffled, 3)] == ["a", "b", "c"]


def test_in_order_response_is_unchanged():
    from ogham.embeddings import _order_by_index

    items = [_Indexed(["a"], 0), _Indexed(["b"], 1)]
    assert _order_by_index("p", items, 2) == items


def test_provider_without_index_falls_through_untouched():
    """ollama and voyage return bare lists -- there is nothing better to use."""
    from ogham.embeddings import _order_by_index

    plain = [["a"], ["b"], ["c"]]
    assert _order_by_index("ollama", plain, 3) == plain


@pytest.mark.parametrize(
    "indices",
    [(0, 0, 1), (0, 1, 3), (1, 2, 3)],
    ids=["duplicate", "gap", "off-by-one"],
)
def test_incoherent_indices_raise_rather_than_guess(indices):
    """A set that is not exactly 0..n-1 means order cannot be established."""
    from ogham.embeddings import _order_by_index

    items = [_Indexed([str(i)], i) for i in indices]
    with pytest.raises(RuntimeError, match="refusing to guess"):
        _order_by_index("p", items, 3)


def test_openai_batch_reorders_before_pairing(monkeypatch):
    """End to end through the provider function, not just the helper."""
    from ogham import embeddings as emb_mod

    class _Resp:
        def __init__(self):
            self.data = [
                _Indexed([3.0] * 512, 2),
                _Indexed([1.0] * 512, 0),
                _Indexed([2.0] * 512, 1),
            ]
            self.usage = None

    class _Client:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                return _Resp()

    monkeypatch.setattr(emb_mod, "_get_openai_client", lambda: _Client())
    with patch.multiple(emb_mod.settings, openai_api_key="k", embedding_dim=512):
        out = emb_mod._embed_openai_batch(["a", "b", "c"])

    assert [v[0] for v in out] == [1.0, 2.0, 3.0], "response order leaked into the result"


# --- TBU-210: retry only what another attempt can fix -----------------------


class _APIError(Exception):
    """Shaped like google.genai.errors.APIError, which carries .code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(code):
    from ogham.embeddings import _is_rate_limit_error

    assert _is_rate_limit_error(_APIError(code, "server said no"))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 413])
def test_terminal_statuses_are_not_retried(code):
    """These fail identically on every attempt -- retrying only hides the cause."""
    from ogham.embeddings import _is_rate_limit_error

    assert not _is_rate_limit_error(_APIError(code, "bad request"))


def test_permanent_quota_failure_is_not_retried():
    """The concrete TBU-210 bug.

    Billing disabled and a disabled API both return 403 and both say "quota".
    The old `"quota" in msg.lower()` earned them six attempts and 93s of
    backoff before surfacing a cause that was fixed from the first call.
    """
    from ogham.embeddings import _is_rate_limit_error

    exc = _APIError(403, "Quota exceeded: billing has not been enabled for this project")
    assert not _is_rate_limit_error(exc)


def test_real_rate_limit_is_still_retried():
    from ogham.embeddings import _is_rate_limit_error

    assert _is_rate_limit_error(_APIError(429, "RESOURCE_EXHAUSTED: rate limit exceeded"))


def test_status_digits_embedded_in_prose_do_not_trigger_a_retry():
    """`"429" in msg` matched a request id or a token count. Whole tokens only."""
    from ogham.embeddings import _is_rate_limit_error

    for msg in (
        "invalid request: input had 1429 tokens, limit is 1024",
        "request id req_50034297 failed validation",
    ):
        assert not _is_rate_limit_error(RuntimeError(msg)), msg


def test_untyped_transport_errors_fall_back_to_the_message():
    """Not every failure carries a status; the fallback must still work."""
    from ogham.embeddings import _is_rate_limit_error

    assert _is_rate_limit_error(RuntimeError("503 UNAVAILABLE"))
    assert _is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED"))
    assert not _is_rate_limit_error(RuntimeError("connection reset by peer"))


def test_bool_code_is_not_read_as_a_status():
    """True == 1 in Python, so a flag must not be consumed as an HTTP status.

    The message is deliberately free of any retryable token: with the bool
    rejected, classification falls through to the message, and that fallback
    must then decide on its own merits.
    """
    from ogham.embeddings import _is_rate_limit_error

    assert not _is_rate_limit_error(_APIError(True, "malformed input"))
    # And the fallback still works when the message does justify a retry.
    assert _is_rate_limit_error(_APIError(False, "503 UNAVAILABLE"))
