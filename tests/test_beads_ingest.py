from typer.testing import CliRunner

from ogham.cli import app
from ogham.importers.beads import (
    BeadsClient,
    _default_run,
    map_comment_to_record,
    map_issue_to_record,
)
from ogham.tools.import_beads import import_beads_impl


class _FakeRun:
    """Records (args, cwd) and returns a scripted stdout keyed by the argv tuple."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, args, cwd):
        self.calls.append((args, cwd))
        return self.responses.get(tuple(args), "[]")


def test_list_issues_argv_and_parse():
    run = _FakeRun({("bd", "list", "--all", "--json", "--limit", "0"): '[{"id": "bd-a1"}]'})
    client = BeadsClient("/proj", run=run)
    issues = client.list_issues()
    assert issues == [{"id": "bd-a1"}]
    assert run.calls[0] == (["bd", "list", "--all", "--json", "--limit", "0"], "/proj")


def test_list_comments_argv():
    run = _FakeRun({("bd", "comments", "bd-a1", "--json"): '[{"id": "c1", "text": "hi"}]'})
    client = BeadsClient("/proj", run=run)
    assert client.list_comments("bd-a1") == [{"id": "c1", "text": "hi"}]


def test_empty_output_is_empty_list():
    client = BeadsClient("/proj", run=lambda args, cwd: "")
    assert client.list_issues() == []


def test_non_json_output_raises_clean_error():
    client = BeadsClient("/proj", run=lambda args, cwd: "not json at all")
    try:
        client.list_issues()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "bd" in str(e)


def test_run_error_propagates():
    def boom(args, cwd):
        raise RuntimeError("bd failed (exit 1)")

    try:
        BeadsClient("/proj", run=boom).list_issues()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "exit 1" in str(e)


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_default_run_falls_back_to_stdout_when_stderr_empty(monkeypatch):
    # bd writes its JSON error payload to stdout and leaves stderr empty.
    monkeypatch.setattr(
        "ogham.importers.beads.subprocess.run",
        lambda *a, **kw: _FakeCompletedProcess(1, stdout='{"error":"boom"}', stderr=""),
    )
    try:
        _default_run(["bd", "comments", "bd-a1", "--json"], "/proj")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "boom" in str(e)


def test_default_run_missing_binary_raises_clean_error(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr("ogham.importers.beads.subprocess.run", boom)
    try:
        _default_run(["bd", "list", "--all", "--json", "--limit", "0"], "/proj")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "not found on PATH" in str(e)


def test_map_issue_to_record():
    issue = {
        "id": "bd-a1",
        "title": "Fix login",
        "description": "null check needed",
        "status": "open",
        "priority": 0,
        "issue_type": "task",
        "labels": ["bug", "p0"],
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-07-01T00:00:00Z",
        "comment_count": 1,
    }
    rec = map_issue_to_record(issue)
    assert rec is not None
    assert rec.content == "Fix login\n\nnull check needed"
    assert rec.metadata["beads_key"] == "issue:bd-a1"
    assert rec.metadata["priority"] == 0  # 0 is valid (P0), not dropped
    assert rec.tags == ["bug", "p0"]


def test_map_issue_skips_empty():
    assert map_issue_to_record({"id": "bd-x", "title": "", "description": ""}) is None
    assert map_issue_to_record({"title": "no id"}) is None


def test_map_comment_to_record():
    rec = map_comment_to_record("bd-a1", {"id": "c9", "text": "repro on staging", "author": "kev"})
    assert rec is not None
    assert rec.content == "repro on staging"
    assert rec.metadata["beads_key"] == "comment:c9"
    assert rec.metadata["issue_id"] == "bd-a1"


def test_map_comment_skips_empty():
    assert map_comment_to_record("bd-a1", {"id": "c1", "text": ""}) is None


class _FakeBeads:
    """issues: list[dict]; comments: {issue_id: [comment,...]}; raise_for: set of ids."""

    def __init__(self, issues, comments=None, raise_for=None):
        self._issues = issues
        self._comments = comments or {}
        self._raise = raise_for or set()
        self.comment_calls = []

    def list_issues(self):
        return self._issues

    def list_comments(self, issue_id):
        self.comment_calls.append(issue_id)
        if issue_id in self._raise:
            raise RuntimeError("comments boom")
        return self._comments.get(issue_id, [])


class _FakeSvc:
    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.stored = []

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        self.stored.append(record)
        return {"status": "stored", "id": "x"}


def _issue(iid, comment_count=0, title="T", body="B"):
    return {
        "id": iid,
        "title": title,
        "description": body,
        "comment_count": comment_count,
        "labels": [],
    }


def test_impl_stores_issues_and_fans_out_only_when_comments():
    beads = _FakeBeads(
        issues=[_issue("bd-a1", comment_count=0), _issue("bd-a2", comment_count=1)],
        comments={"bd-a2": [{"id": "c9", "text": "a comment"}]},
    )
    svc = _FakeSvc()
    import_beads_impl(client=beads, service=svc, profile="work")
    keys = {rec.metadata["beads_key"] for rec in svc.stored}
    assert keys == {"issue:bd-a1", "issue:bd-a2", "comment:c9"}
    assert beads.comment_calls == ["bd-a2"]  # NOT called for bd-a1 (comment_count 0)


def test_impl_dedups_existing():
    beads = _FakeBeads(issues=[_issue("bd-a1")])
    svc = _FakeSvc(existing={"issue:bd-a1"})
    r = import_beads_impl(client=beads, service=svc, profile="work")
    assert r["skipped_duplicate"] == 1 and svc.stored == []


def test_impl_comment_fetch_error_continues():
    beads = _FakeBeads(issues=[_issue("bd-a1", comment_count=1)], raise_for={"bd-a1"})
    svc = _FakeSvc()
    import_beads_impl(client=beads, service=svc, profile="work")
    # the issue is still stored even though its comment fetch raised
    assert {rec.metadata["beads_key"] for rec in svc.stored} == {"issue:bd-a1"}


def test_cli_missing_beads_dir_errors(monkeypatch):
    monkeypatch.delenv("BEADS_DIR", raising=False)
    result = CliRunner().invoke(app, ["import-beads", "--profile", "work"])
    assert result.exit_code == 1
    assert "BEADS_DIR" in result.output


def test_cli_beads_dir_not_a_directory_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("BEADS_DIR", str(tmp_path / "nope"))
    result = CliRunner().invoke(app, ["import-beads", "--profile", "work"])
    assert result.exit_code == 1
    assert "BEADS_DIR" in result.output


def test_cli_dry_run_reports_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("BEADS_DIR", str(tmp_path))  # a real dir
    monkeypatch.setattr("ogham.cli.BeadsClient", lambda beads_dir: _FakeBeads(issues=[]))
    monkeypatch.setattr("ogham.cli.DefaultIngestService", _FakeSvc)
    result = CliRunner().invoke(app, ["import-beads", "--profile", "work", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "scanned=0" in result.output
