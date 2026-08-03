"""Integrity checks on the ONNX model download (TBU-196 item H).

`ogham download-model` fetches a 1.3 GB archive over the network and extracts
it. Until now it did so with no integrity check whatsoever: a truncated
transfer, a corrupted proxy cache, or a swapped upstream asset would have been
unpacked and used to generate embeddings.

The pinned digest is the value GitHub publishes for the release asset:

    gh api repos/yuniko-software/bge-m3-onnx/releases/tags/1.01 \
      --jq '.assets[] | "\\(.name) \\(.size) \\(.digest)"'

It was also verified locally on 2026-07-28 by downloading the archive and
hashing it -- 1322654161 bytes, sha256 fef1d045...ed8e -- so the constant is
not taken on trust from a single source.

As with the migration rules, the negative tests are the point: a check nobody
has watched fail is indistinguishable from a comment.
"""

import hashlib
from pathlib import Path

import pytest


def test_registry_pins_a_digest_and_size():
    """The pin must exist. Without it every other test here is theatre."""
    from ogham.cli import MODEL_REGISTRY

    entry = MODEL_REGISTRY["bge-m3"]
    assert len(entry["sha256"]) == 64, "sha256 must be a full hex digest"
    assert int(entry["size"]) > 0
    assert entry["url"].startswith("https://"), "model must be fetched over TLS"


def test_registry_matches_the_verified_upstream_values():
    """Pinned values verified against GitHub's published digest AND a local hash."""
    from ogham.cli import MODEL_REGISTRY

    entry = MODEL_REGISTRY["bge-m3"]
    assert entry["size"] == 1322654161
    assert entry["sha256"] == "fef1d045ace47593bd7f149be2bfd72658625ad2786b0d3a79a90d48f7e5ed8e"


# --- the verifier, both ways ----------------------------------------------


@pytest.fixture
def archive(tmp_path):
    payload = b"pretend this is a 1.3GB onnx archive"
    p = tmp_path / "onnx.zip"
    p.write_bytes(payload)
    return p, hashlib.sha256(payload).hexdigest(), len(payload)


def test_accepts_a_matching_archive(archive):
    from ogham.cli import verify_archive

    path, digest, size = archive
    assert verify_archive(path, digest, size) == []


def test_rejects_a_truncated_archive(archive):
    """The common real-world failure: an interrupted transfer."""
    from ogham.cli import verify_archive

    path, digest, size = archive
    path.write_bytes(path.read_bytes()[:10])
    problems = verify_archive(path, digest, size)
    assert problems and "size mismatch" in problems[0]


def test_rejects_tampered_content_of_the_right_length(archive):
    """Same byte count, different bytes -- only the hash catches this."""
    from ogham.cli import verify_archive

    path, digest, size = archive
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    assert len(path.read_bytes()) == size
    problems = verify_archive(path, digest, size)
    assert problems and "sha256 mismatch" in problems[0]


def test_rejects_a_missing_archive(tmp_path):
    from ogham.cli import verify_archive

    problems = verify_archive(tmp_path / "nope.zip", "0" * 64, 1)
    assert problems and "does not exist" in problems[0]


def test_size_is_checked_before_hashing(tmp_path, monkeypatch):
    """Hashing 1.3 GB to discover a truncated file would be wasteful."""
    import ogham.cli as cli

    p = tmp_path / "a.zip"
    p.write_bytes(b"short")

    called = False
    real_sha256 = hashlib.sha256

    def spy(*args, **kwargs):
        nonlocal called
        called = True
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(cli.__dict__.get("hashlib", hashlib), "sha256", spy, raising=False)
    problems = cli.verify_archive(p, "0" * 64, 999999)
    assert problems and "size mismatch" in problems[0]
    assert not called, "hashing should be skipped when the size already disagrees"


def test_download_aborts_and_extracts_nothing_when_verification_fails(tmp_path, monkeypatch):
    """End to end: a bad archive must not reach the extraction step."""
    import zipfile
    from unittest.mock import patch

    from typer.testing import CliRunner

    from ogham.cli import app

    dest = tmp_path / "model"

    def fake_urlretrieve(url, filename, reporthook=None):
        # Well-formed zip, wrong bytes -- passes as a zip, fails the digest.
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr("bge_m3_model.onnx", "not the real model")
        return filename, None

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        result = CliRunner().invoke(app, ["download-model", "bge-m3", "--path", str(dest)])

    assert result.exit_code == 1
    assert "verification" in result.output.lower()
    assert not (dest / "bge_m3_model.onnx").exists(), "nothing may be extracted"


def test_download_model_still_short_circuits_when_present(tmp_path):
    """The existing-model path must not start a 1.3 GB download to verify it."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from ogham.cli import app

    dest = tmp_path / "model"
    dest.mkdir()
    for name in ("bge_m3_model.onnx", "bge_m3_model.onnx_data"):
        (dest / name).write_bytes(b"stub")

    with patch("urllib.request.urlretrieve") as fetch:
        result = CliRunner().invoke(app, ["download-model", "bge-m3", "--path", str(dest)])

    assert result.exit_code == 0
    fetch.assert_not_called()


def test_no_model_url_is_plain_http():
    """A digest check is worth much less over a channel anyone can rewrite."""
    from ogham.cli import MODEL_REGISTRY

    for name, entry in MODEL_REGISTRY.items():
        assert entry["url"].startswith("https://"), f"{name} is not fetched over TLS"


def test_expected_files_are_declared(tmp_path):
    from ogham.cli import MODEL_REGISTRY

    assert MODEL_REGISTRY["bge-m3"]["expected_files"] == [
        "bge_m3_model.onnx",
        "bge_m3_model.onnx_data",
    ]


def test_verify_archive_reads_in_chunks(tmp_path):
    """A 1.3 GB archive must not be slurped into memory to be hashed."""
    import inspect

    from ogham.cli import verify_archive

    src = inspect.getsource(verify_archive)
    assert "iter(" in src and "read(" in src, "should stream the file, not read() it whole"
    assert ".read()" not in src.replace("fh.read(1024 * 1024)", ""), "no whole-file read"
    # and it still works
    p = Path(tmp_path) / "x.zip"
    p.write_bytes(b"abc")
    assert verify_archive(p, hashlib.sha256(b"abc").hexdigest(), 3) == []
