from typer.testing import CliRunner

from ogham.cli import app
from ogham.importers.obsidian import (
    ParsedNote,
    _normalize_body,
    compute_fingerprint,
    iter_vault_notes,
    parse_note,
)
from ogham.tools.import_obsidian import ingest_obsidian_impl


def test_fingerprint_stable_for_same_content():
    assert compute_fingerprint("hello world") == compute_fingerprint("hello world")


def test_fingerprint_ignores_trailing_whitespace_and_line_endings():
    assert compute_fingerprint("a\r\nb  \n") == compute_fingerprint("a\nb")


def test_fingerprint_changes_on_real_edit():
    assert compute_fingerprint("hello world") != compute_fingerprint("hello there")


def test_normalize_body_strips_trailing_ws_and_collapses_edge_newlines():
    assert _normalize_body("\n\nfoo  \r\nbar\t\n\n") == "foo\nbar"


def test_parse_note_no_frontmatter():
    note = parse_note("daily/2026-07-06.md", "just a plain note\n")
    assert isinstance(note, ParsedNote)
    assert note.content == "just a plain note"
    assert note.tags == []
    assert note.metadata["vault_path"] == "daily/2026-07-06.md"
    assert note.metadata["content_fingerprint"] == note.fingerprint
    assert "frontmatter" not in note.metadata


def test_parse_note_lifts_frontmatter_tags_list():
    raw = "---\ntags: [meeting, jamie]\ntitle: Standup\n---\nBody text\n"
    note = parse_note("m.md", raw)
    assert note.tags == ["meeting", "jamie"]
    assert note.content == "Body text"
    assert note.metadata["frontmatter"] == {"title": "Standup"}


def test_parse_note_tags_scalar_becomes_single_element_list():
    note = parse_note("m.md", "---\ntags: meeting\n---\nx\n")
    assert note.tags == ["meeting"]


def test_parse_note_malformed_frontmatter_is_treated_as_body():
    raw = "---\ntags: [unclosed\n---\nBody\n"
    note = parse_note("m.md", raw)
    # yaml error -> whole raw is body, no tags lifted
    assert note.tags == []
    assert "Body" in note.content


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_iter_vault_notes_finds_md_recursively_and_applies_ignores(tmp_path):
    _write(tmp_path, "root.md", "a")
    _write(tmp_path, "sub/nested.md", "b")
    _write(tmp_path, ".obsidian/workspace.md", "config")  # dot-dir -> skip
    _write(tmp_path, ".trash/old.md", "trashed")  # dot-dir -> skip
    _write(tmp_path, "notes.txt", "not markdown")  # not .md -> skip
    _write(tmp_path, "empty.md", "")  # empty -> skip

    found = [str(p.relative_to(tmp_path)) for p in iter_vault_notes(str(tmp_path))]
    assert found == ["root.md", "sub/nested.md"]


def test_iter_vault_notes_skips_oversize(tmp_path):
    _write(tmp_path, "big.md", "x" * 50)
    _write(tmp_path, "ok.md", "y")
    found = [p.name for p in iter_vault_notes(str(tmp_path), max_bytes=10)]
    assert found == ["ok.md"]


class _FakeService:
    def __init__(self, existing=None, disabled=False, raise_on=None):
        self.existing = set(existing or [])
        self.disabled = disabled
        self.raise_on = raise_on or set()
        self.stored = []

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        if record.content in self.raise_on:
            raise RuntimeError(f"boom storing {record.content!r}")
        if self.disabled:
            return {"status": "disabled"}
        self.stored.append((record, profile, source))
        return {"status": "stored", "id": "fake"}


def _vault(tmp_path):
    _write(tmp_path, "a.md", "note alpha")
    _write(tmp_path, "b.md", "note beta")
    return str(tmp_path)


def test_impl_first_run_stores_all(tmp_path):
    svc = _FakeService()
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work")
    assert r["scanned"] == 2 and r["stored"] == 2
    assert len(svc.stored) == 2
    assert svc.stored[0][2] == "obsidian"  # source stamped


def test_impl_second_run_dedups(tmp_path):
    from ogham.importers.obsidian import compute_fingerprint

    seen = {compute_fingerprint("note alpha"), compute_fingerprint("note beta")}
    svc = _FakeService(existing=seen)
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work")
    assert r["stored"] == 0 and r["skipped_duplicate"] == 2
    assert svc.stored == []


def test_impl_changed_note_stores_one(tmp_path):
    from ogham.importers.obsidian import compute_fingerprint

    svc = _FakeService(existing={compute_fingerprint("note alpha")})
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work")
    assert r["stored"] == 1 and r["skipped_duplicate"] == 1


def test_impl_dry_run_stores_nothing(tmp_path):
    svc = _FakeService()
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work", dry_run=True)
    assert r["stored"] == 2 and svc.stored == []


def test_impl_disabled_service_counts_disabled(tmp_path):
    svc = _FakeService(disabled=True)
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work")
    assert r["disabled"] == 2 and r["stored"] == 0


def test_impl_unreadable_file_counts_error_not_crash(tmp_path):
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe not utf8")
    _write(tmp_path, "good.md", "fine")
    svc = _FakeService()
    r = ingest_obsidian_impl(vault_path=str(tmp_path), service=svc, profile="work")
    assert r["errors"] == 1 and r["stored"] == 1


def test_impl_store_error_counts_error_and_scan_continues(tmp_path):
    svc = _FakeService(raise_on={"note alpha"})
    r = ingest_obsidian_impl(vault_path=_vault(tmp_path), service=svc, profile="work")
    assert r["errors"] == 1
    assert r["stored"] == 1
    assert len(svc.stored) == 1
    assert svc.stored[0][0].content == "note beta"


def test_impl_missing_vault_path_raises_value_error():
    import pytest

    svc = _FakeService()
    with pytest.raises(ValueError):
        ingest_obsidian_impl(vault_path="/no/such/vault", service=svc, profile="work")


def test_parse_note_json_safe_frontmatter_dates():
    import json

    raw = "---\ndate: 2026-07-06\ncreated: 2026-07-06 10:30:00\n---\nbody\n"
    note = parse_note("d.md", raw)
    assert note.metadata["frontmatter"]["date"] == "2026-07-06"
    json.dumps(note.metadata)  # must not raise


def test_cli_ingest_obsidian_dry_run(tmp_path, monkeypatch):
    _write(tmp_path, "a.md", "hello")

    # Swap the DB-backed service for the in-memory fake so the CLI test needs no DB.
    monkeypatch.setattr("ogham.tools.import_obsidian.DefaultIngestService", _FakeService)
    result = CliRunner().invoke(
        app, ["ingest-obsidian", str(tmp_path), "--profile", "work", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "scanned=1" in result.output
    assert "stored=1" in result.output
