from ogham.ingest import IngestRecord, run_ingest


class FakeIngestService:
    def __init__(self, existing=None, disabled=False, raise_keys=None, reject_keys=None):
        self.existing = set(existing or [])
        self.disabled = disabled
        self.raise_keys = raise_keys or set()
        self.reject_keys = reject_keys or set()
        self.stored = []

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        key = record.metadata["k"]
        if key in self.raise_keys:
            raise RuntimeError(f"boom {key}")
        if key in self.reject_keys:
            raise ValueError(f"content too short ({key})")
        if self.disabled:
            return {"status": "disabled"}
        self.stored.append((record, profile, source))
        return {"status": "stored", "id": "fake"}


def _rec(text, key):
    return IngestRecord(content=text, tags=[], metadata={"k": key})


def _run(items, svc, **kw):
    # items are (text, key) tuples; to_record builds an IngestRecord (or None for empty text)
    def to_record(item):
        text, key = item
        return _rec(text, key) if text is not None else None

    return run_ingest(
        items=items,
        to_record=to_record,
        service=svc,
        profile="work",
        source="unit",
        dedup_key_field="k",
        **kw,
    )


def test_run_ingest_first_run_stores_all():
    svc = FakeIngestService()
    r = _run([("a", "1"), ("b", "2")], svc)
    assert r["scanned"] == 2 and r["stored"] == 2
    assert [s[2] for s in svc.stored] == ["unit", "unit"]


def test_run_ingest_dedups_against_existing():
    svc = FakeIngestService(existing={"1", "2"})
    r = _run([("a", "1"), ("b", "2")], svc)
    assert r["stored"] == 0 and r["skipped_duplicate"] == 2 and svc.stored == []


def test_run_ingest_dry_run_stores_nothing():
    svc = FakeIngestService()
    r = _run([("a", "1")], svc, dry_run=True)
    assert r["stored"] == 1 and svc.stored == []


def test_run_ingest_none_or_empty_record_is_ignored():
    svc = FakeIngestService()
    r = _run([(None, "1"), ("   ", "2")], svc)
    assert r["skipped_ignored"] == 2 and r["stored"] == 0


def test_run_ingest_disabled_counts_disabled():
    svc = FakeIngestService(disabled=True)
    r = _run([("a", "1")], svc)
    assert r["disabled"] == 1 and r["stored"] == 0


def test_run_ingest_store_error_counts_error_and_continues():
    svc = FakeIngestService(raise_keys={"1"})
    r = _run([("a", "1"), ("b", "2")], svc)
    assert r["errors"] == 1 and r["stored"] == 1
    assert svc.stored[0][0].metadata["k"] == "2"


def test_run_ingest_mapping_error_counts_error_and_continues():
    svc = FakeIngestService()

    def to_record(item):
        if item == "bad":
            raise ValueError("cannot map")
        return IngestRecord(content="ok", tags=[], metadata={"k": item})

    r = run_ingest(
        items=["bad", "good"],
        to_record=to_record,
        service=svc,
        profile="work",
        source="unit",
        dedup_key_field="k",
    )
    assert r["scanned"] == 2 and r["errors"] == 1 and r["stored"] == 1


def test_run_ingest_uses_passed_existing_without_fetch():
    svc = FakeIngestService(existing={"should-not-be-used"})
    r = _run([("a", "1")], svc, existing=set())
    assert r["stored"] == 1  # passed existing (empty) wins over service.fetch


def test_run_ingest_stop_on_store_error_halts_the_loop():
    svc = FakeIngestService(raise_keys={"2"})
    r = _run([("a", "1"), ("b", "2"), ("c", "3")], svc, stop_on_store_error=True)
    assert r["stored"] == 1 and r["errors"] == 1
    assert r["stopped"] is True
    assert [s[0].metadata["k"] for s in svc.stored] == ["1"]
    assert r["scanned"] == 2  # item "3" was never reached


def test_run_ingest_store_value_error_is_permanent_and_never_stalls():
    # TBU-180: a store ValueError (e.g. content-too-short validation) is a
    # PERMANENT rejection -- it must be skipped (skipped_ignored) and the
    # drain must continue, even with stop_on_store_error=True, never stopped.
    svc = FakeIngestService(reject_keys={"2"})
    r = _run([("a", "1"), ("hi", "2"), ("c", "3")], svc, stop_on_store_error=True)
    assert r["skipped_ignored"] == 1
    assert r["errors"] == 0
    assert r["stopped"] is False
    assert r["stored"] == 2
    assert [s[0].metadata["k"] for s in svc.stored] == ["1", "3"]


def test_run_ingest_stop_on_store_error_does_not_stop_on_mapping_error():
    svc = FakeIngestService()

    def to_record(item):
        if item == "bad":
            raise ValueError("cannot map")
        return IngestRecord(content="ok", tags=[], metadata={"k": item})

    r = run_ingest(
        items=["bad", "good"],
        to_record=to_record,
        service=svc,
        profile="work",
        source="unit",
        dedup_key_field="k",
        stop_on_store_error=True,
    )
    assert r["scanned"] == 2 and r["errors"] == 1 and r["stored"] == 1
    assert r["stopped"] is False
