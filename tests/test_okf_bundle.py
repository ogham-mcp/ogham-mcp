from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ogham.okf.bundle import export_okf_bundle, filter_expired, import_okf_bundle, write_index


def test_write_index_emits_okf_version_in_frontmatter(tmp_path: Path):
    manifest = {
        "producer": "ogham-mcp/0.15.0",
        "exported_at": "2026-06-17T09:00:00Z",
        "profile": "work",
    }
    write_index(tmp_path, manifest)

    index_path = tmp_path / "index.md"
    assert index_path.exists()
    text = index_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    # okf_version declaration goes in the root index.md per OKF spec §11
    assert "okf_version:" in text
    assert "'0.1'" in text or '"0.1"' in text or "0.1" in text
    assert "producer: ogham-mcp/0.15.0" in text
    assert "profile: work" in text


def test_write_index_yaml_round_trip(tmp_path: Path):
    write_index(tmp_path, {"producer": "p", "exported_at": "t", "profile": "default"})
    text = (tmp_path / "index.md").read_text(encoding="utf-8")
    yaml_block = text.split("---\n", 2)[1]
    parsed = yaml.safe_load(yaml_block)
    assert parsed["okf_version"] == "0.1"
    assert parsed["producer"] == "p"


def test_filter_expired_removes_past_expires_at():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    memories = [
        {"id": "a", "expires_at": past},
        {"id": "b", "expires_at": future},
        {"id": "c", "expires_at": None},
        {"id": "d"},  # no expires_at field
    ]
    kept = filter_expired(memories)
    kept_ids = {m["id"] for m in kept}
    assert kept_ids == {"b", "c", "d"}


def test_filter_expired_tolerates_datetime_expires_at():
    """TBU-162 audit: the Postgres backend (psycopg) returns expires_at as a
    real datetime object, not an ISO string. Must not raise."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    memories = [
        {"id": "a", "expires_at": past},
        {"id": "b", "expires_at": future},
    ]
    kept = filter_expired(memories)
    kept_ids = {m["id"] for m in kept}
    assert kept_ids == {"b"}


def _make_memory(id_: str, content: str = "x", tags: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "content": content,
        "tags": tags or [],
        "source": None,
        "created_at": "2026-06-17T00:00:00Z",
        "metadata": {},
    }


def test_export_okf_bundle_writes_one_file_per_memory(tmp_path: Path):
    memories = [
        _make_memory("7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99", "Decision content", ["type:decision"]),
        _make_memory("d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88", "Gotcha content", ["type:gotcha"]),
    ]
    manifest = {"producer": "ogham-mcp/0.15.0", "exported_at": "t", "profile": "work"}
    export_okf_bundle(memories, tmp_path, manifest)

    assert (tmp_path / "index.md").exists()
    memories_dir = tmp_path / "memories"
    files = list(memories_dir.glob("*.md"))
    assert len(files) == 2
    names = {f.name for f in files}
    assert "decision-content-7da3c025.md" in names
    assert "gotcha-content-d3c08af7.md" in names


def test_export_okf_bundle_is_atomic_no_partial_on_error(tmp_path: Path):
    # If a memory record is malformed mid-walk, the bundle dir MUST NOT be
    # left in a half-written state. We use temp-dir-then-rename.
    memories = [_make_memory("7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99")]
    target = tmp_path / "bundle"
    manifest = {"producer": "p", "exported_at": "t", "profile": "p"}
    export_okf_bundle(memories, target, manifest)
    assert target.exists()
    # Pre-existing target gets replaced atomically (not appended to)
    memories2 = [_make_memory("d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88")]
    export_okf_bundle(memories2, target, manifest)
    files = list((target / "memories").glob("*.md"))
    assert len(files) == 1
    assert "d3c08af7" in files[0].name


def test_export_okf_bundle_filters_expired_by_default(tmp_path: Path):
    past = "2020-01-01T00:00:00Z"
    expired = _make_memory("7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99")
    expired["expires_at"] = past
    fresh = _make_memory("d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88")
    manifest = {"producer": "p", "exported_at": "t", "profile": "p"}
    export_okf_bundle([expired, fresh], tmp_path, manifest)

    files = list((tmp_path / "memories").glob("*.md"))
    assert len(files) == 1
    assert "d3c08af7" in files[0].name


def test_import_okf_bundle_returns_memories_and_stats(tmp_path: Path):
    # Build a bundle by exporting first, then re-importing
    memories = [
        _make_memory("7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99", "First", ["type:decision"]),
        _make_memory("d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88", "Second", ["type:gotcha"]),
    ]
    manifest = {"producer": "p", "exported_at": "t", "profile": "p"}
    export_okf_bundle(memories, tmp_path, manifest)

    imported, stats = import_okf_bundle(tmp_path)
    assert len(imported) == 2
    ids = {m["id"] for m in imported}
    assert "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99" in ids
    assert "d3c08af7-3f2a-4d5b-a82c-6f9e1b2d4f88" in ids
    assert stats["total"] == 2
    assert stats["missing_id_count"] == 0
    # Content must survive export→import without mutation (guards against
    # trailing-newline accumulation and other body-corruption bugs).
    imported_by_id = {m["id"]: m for m in imported}
    for original in memories:
        rt = imported_by_id[original["id"]]
        assert rt["content"] == original["content"]


def test_import_okf_bundle_counts_missing_ids(tmp_path: Path):
    # Hand-author a bundle where one concept has no id frontmatter
    from ogham.okf.serialization import write_concept

    (tmp_path / "memories").mkdir(parents=True)
    write_concept(
        tmp_path / "index.md",
        {"okf_version": "0.1", "producer": "test", "exported_at": "t", "profile": "x"},
        "# Test bundle",
    )
    write_concept(
        tmp_path / "memories" / "with-id.md",
        {"type": "Memory", "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99", "tags": []},
        "has an id",
    )
    write_concept(
        tmp_path / "memories" / "no-id.md",
        {"type": "Memory", "tags": []},  # id absent
        "missing id",
    )

    imported, stats = import_okf_bundle(tmp_path)
    assert stats["total"] == 2
    assert stats["missing_id_count"] == 1


def test_import_okf_bundle_counts_malformed_concepts(tmp_path: Path):
    """Malformed concept files are skipped silently today; the stats counter
    surfaces the drop so the import tool can warn the user."""
    from ogham.okf.serialization import write_concept

    (tmp_path / "memories").mkdir()
    # Valid concept
    write_concept(
        tmp_path / "memories" / "good.md",
        {"type": "Memory", "id": "11111111-2222-3333-4444-555555555555", "tags": []},
        "ok",
    )
    # Malformed: no frontmatter
    (tmp_path / "memories" / "bad.md").write_text("no frontmatter here\n", encoding="utf-8")

    imported, stats = import_okf_bundle(tmp_path)
    assert stats["total"] == 1
    assert stats["skipped_count"] == 1


def test_import_okf_bundle_skips_index_and_log(tmp_path: Path):
    # Reserved filenames must NOT be parsed as concepts (spec §3.1)
    from ogham.okf.serialization import write_concept

    (tmp_path / "memories").mkdir(parents=True)
    write_concept(
        tmp_path / "index.md",
        {"okf_version": "0.1"},
        "# Index",
    )
    # log.md per spec §7 has no frontmatter
    (tmp_path / "log.md").write_text("# Log\n\n## 2026-06-17\n* Update\n", encoding="utf-8")
    write_concept(
        tmp_path / "memories" / "real-concept.md",
        {"type": "Memory", "id": "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99", "tags": []},
        "real",
    )
    imported, stats = import_okf_bundle(tmp_path)
    assert stats["total"] == 1
    assert imported[0]["id"] == "7da3c025-fa77-4f0b-9d2e-1ab84e6c3f99"
