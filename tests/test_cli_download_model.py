"""Tests for `ogham download-model` (ONNX BGE-M3 model fetch).

Regression context: the command was contributed in PR #14 (commit cdff557,
2026-03-28) and silently deleted three days later by the v0.8.5 cli.py
restructure (commit 3e18eea), while onnx_embedder.py and the README kept
telling users to run it. Reported as ogham-mcp#68 after four months.

The final test in this module is the guard that stops it recurring.
"""

import re
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")


def _registered_commands() -> set[str]:
    """Command and sub-app names as typer exposes them on the CLI."""
    from ogham.cli import app

    names = set()
    for command in app.registered_commands:
        if command.name:
            names.add(command.name)
        elif command.callback is not None:
            names.add(command.callback.__name__.replace("_", "-"))
    # Sub-apps (e.g. `ogham hooks ...`) are groups, not commands.
    for group in app.registered_groups:
        if group.name:
            names.add(group.name)
        elif group.typer_instance is not None and group.typer_instance.info.name:
            names.add(group.typer_instance.info.name)
    return names


def test_download_model_is_registered():
    """The command referenced by onnx_embedder and the README must exist."""
    assert "download-model" in _registered_commands()


def test_unknown_model_exits_nonzero():
    from ogham.cli import app

    result = runner.invoke(app, ["download-model", "not-a-real-model"])
    assert result.exit_code == 1
    assert "not-a-real-model" in result.output


def test_already_downloaded_skips_network(tmp_path):
    """An existing model short-circuits before any download is attempted."""
    from ogham.cli import app

    for name in ("bge_m3_model.onnx", "bge_m3_model.onnx_data"):
        (tmp_path / name).write_bytes(b"stub")

    with patch("urllib.request.urlretrieve") as mock_fetch:
        result = runner.invoke(app, ["download-model", "bge-m3", "--path", str(tmp_path)])

    assert result.exit_code == 0
    mock_fetch.assert_not_called()
    assert "already exists" in result.output


def test_successful_download_extracts_expected_files(tmp_path):
    """A well-formed archive lands both model files in the destination."""
    from ogham.cli import app

    dest = tmp_path / "dest"
    archive_src = tmp_path / "src"
    archive_src.mkdir()
    for name in ("bge_m3_model.onnx", "bge_m3_model.onnx_data"):
        (archive_src / name).write_bytes(b"payload")

    def fake_urlretrieve(url, filename, reporthook=None):
        with zipfile.ZipFile(filename, "w") as zf:
            for name in ("bge_m3_model.onnx", "bge_m3_model.onnx_data"):
                zf.write(archive_src / name, arcname=f"onnx/{name}")
        return filename, None

    # A stub archive cannot match the pinned digest, so the integrity gate is
    # bypassed here: this test covers EXTRACTION. Verification has its own
    # coverage in test_download_model_integrity.py, including the negative case
    # that a failing digest extracts nothing.
    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve),
        patch("ogham.cli.verify_archive", return_value=[]),
    ):
        result = runner.invoke(app, ["download-model", "bge-m3", "--path", str(dest)])

    assert result.exit_code == 0
    assert (dest / "bge_m3_model.onnx").read_bytes() == b"payload"
    assert (dest / "bge_m3_model.onnx_data").read_bytes() == b"payload"


def test_path_traversal_member_is_rejected(tmp_path):
    """A zip member escaping the extraction root aborts the install."""
    from ogham.cli import app

    dest = tmp_path / "dest"

    def fake_urlretrieve(url, filename, reporthook=None):
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr("../escaped.onnx", "payload")
        return filename, None

    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve),
        patch("ogham.cli.verify_archive", return_value=[]),
    ):
        result = runner.invoke(app, ["download-model", "bge-m3", "--path", str(dest)])

    assert result.exit_code == 1
    assert not (tmp_path / "escaped.onnx").exists()


def test_missing_expected_file_cleans_up_partials(tmp_path):
    """An archive missing a required file leaves no half-installed model."""
    from ogham.cli import app

    dest = tmp_path / "dest"

    def fake_urlretrieve(url, filename, reporthook=None):
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr("bge_m3_model.onnx", "payload")  # _data absent
        return filename, None

    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve),
        patch("ogham.cli.verify_archive", return_value=[]),
    ):
        result = runner.invoke(app, ["download-model", "bge-m3", "--path", str(dest)])

    assert result.exit_code == 1
    assert not (dest / "bge_m3_model.onnx").exists()


# --- the guard against this class of regression ---------------------------


def test_every_command_named_in_source_strings_exists():
    """No source file may tell a user to run a command that isn't registered.

    This is the check that would have caught ogham-mcp#68 in March: the
    error message in onnx_embedder.py outlived the command it named.
    """
    registered = _registered_commands()
    src = Path(__file__).resolve().parent.parent / "src" / "ogham"

    # Only instruction-shaped references: "Run 'ogham x'" or a backticked
    # `ogham x`. A bare "ogham x" inside an arbitrary string (e.g. the
    # search query "ogham boot warmup") is not telling the user to run
    # anything, so it must not trip this guard.
    pattern = re.compile(
        r"(?:[Rr]un|[Tt]ry|[Uu]se)\s+['\"`]ogham ([a-z][a-z0-9-]*)"
        r"|`ogham ([a-z][a-z0-9-]*)"
    )

    missing = []
    for path in sorted(src.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            # The Go CLI (ogham-cli) has its own command set; not our concern.
            if "ogham-cli" in line or "Go CLI" in line:
                continue
            for match in pattern.finditer(line):
                verb = match.group(1) or match.group(2)
                if verb in registered:
                    continue
                missing.append(f"{path.relative_to(src.parent.parent)}:{line_no} -> 'ogham {verb}'")

    assert not missing, "Source references commands that do not exist:\n" + "\n".join(missing)
