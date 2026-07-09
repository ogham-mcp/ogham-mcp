import re

import httpx
from typer.testing import CliRunner

from ogham.cli import app
from ogham.importers.github_issues import (
    GitHubClient,
    map_comment_to_record,
    map_issue_to_record,
)
from ogham.tools.import_github import _since_iso, import_github_impl


def _client(handler):
    return GitHubClient("TOKEN", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_list_issues_request_shape_and_pagination():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["accept"] = request.headers.get("accept")
        return httpx.Response(200, json=[{"id": i, "number": i} for i in range(100)])  # full page

    issues, next_page = _client(handler).list_issues("o/r", since="2026-01-01T00:00:00Z", page=1)
    assert len(issues) == 100 and next_page == 2  # full page -> more
    assert "/repos/o/r/issues" in seen["url"]
    assert "state=all" in seen["url"] and "since=2026-01-01" in seen["url"]
    assert "page=1" in seen["url"]
    assert seen["auth"] == "Bearer TOKEN"
    assert seen["accept"] == "application/vnd.github+json"


def test_list_issues_partial_page_is_last():
    def handler(request):
        return httpx.Response(200, json=[{"id": 1, "number": 1}])  # < per_page

    issues, next_page = _client(handler).list_issues("o/r", page=1)
    assert len(issues) == 1 and next_page is None


def test_list_comments_parses():
    def handler(request):
        assert "/repos/o/r/issues/5/comments" in str(request.url)
        return httpx.Response(200, json=[{"id": 9, "body": "a comment"}])

    comments, next_page = _client(handler).list_comments("o/r", 5, page=1)
    assert comments == [{"id": 9, "body": "a comment"}] and next_page is None


def test_client_non_2xx_raises_clean_error():
    def handler(request):
        return httpx.Response(404, json={"message": "Not Found"})

    try:
        _client(handler).list_issues("o/r")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "404" in str(e) and "TOKEN" not in str(e)


def test_map_issue_to_record():
    issue = {
        "id": 100,
        "number": 7,
        "title": "Fix the bug",
        "body": "it crashes",
        "state": "open",
        "html_url": "http://x/7",
        "user": {"login": "kev"},
        "updated_at": "2026-07-01T00:00:00Z",
        "labels": [{"name": "bug"}, {"name": "p1"}],
    }
    rec = map_issue_to_record("o/r", issue)
    assert rec is not None
    assert rec.content == "Fix the bug\n\nit crashes"
    assert rec.metadata["github_key"] == "issue:100"
    assert rec.metadata["repo"] == "o/r" and rec.metadata["number"] == 7
    assert rec.tags == ["bug", "p1"]


def test_map_issue_skips_pull_request_and_empty():
    pr_issue = {"id": 1, "pull_request": {"url": "x"}, "title": "PR"}
    assert map_issue_to_record("o/r", pr_issue) is None
    assert map_issue_to_record("o/r", {"id": 2, "title": "", "body": ""}) is None


def test_map_comment_to_record():
    comment = {"id": 55, "body": "looks good", "html_url": "http://x", "user": {"login": "kev"}}
    rec = map_comment_to_record("o/r", {"number": 7}, comment)
    assert rec is not None
    assert rec.content == "looks good"
    assert rec.metadata["github_key"] == "comment:55"
    assert rec.metadata["issue_number"] == 7


def test_map_comment_skips_empty():
    assert map_comment_to_record("o/r", {"number": 7}, {"id": 1, "body": ""}) is None


def test_since_iso_format():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", _since_iso(30))


class _FakeGH:
    """Serves issues per repo and comments per (repo, issue_number).

    issues: {repo: [issue, ...]}   comments: {(repo, number): [comment, ...]}
    raise_comments_for: set of issue numbers whose list_comments raises.
    """

    def __init__(self, issues, comments=None, raise_comments_for=None):
        self._issues = issues
        self._comments = comments or {}
        self._raise = raise_comments_for or set()

    def list_issues(self, repo, since=None, page=1, per_page=100):
        return (self._issues.get(repo, []) if page == 1 else []), None

    def list_comments(self, repo, issue_number, page=1, per_page=100):
        if issue_number in self._raise:
            raise RuntimeError("comment fetch boom")
        return (self._comments.get((repo, issue_number), []) if page == 1 else []), None


class _FakeSvc:
    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.stored = []

    def fetch_existing_keys(self, profile, source, key_field):
        return set(self.existing)

    def store(self, record, profile, source):
        self.stored.append(record)
        return {"status": "stored", "id": "x"}


def _issue(iid, num, title="T", body="B"):
    return {
        "id": iid,
        "number": num,
        "title": title,
        "body": body,
        "labels": [],
        "user": {"login": "k"},
    }


def test_impl_stores_issue_and_comments_filters_prs():
    pr = {"id": 2, "number": 11, "pull_request": {"url": "x"}, "title": "PR"}
    gh = _FakeGH(
        issues={"o/r": [_issue(1, 10), pr]},
        comments={("o/r", 10): [{"id": 99, "body": "a comment"}]},
    )
    svc = _FakeSvc()
    r = import_github_impl(client=gh, service=svc, profile="work", repos=["o/r"])
    keys = {rec.metadata["github_key"] for rec in svc.stored}
    assert keys == {"issue:1", "comment:99"}  # PR (id 2) filtered out
    assert r["stored"] == 2


def test_impl_dedups_existing():
    gh = _FakeGH(issues={"o/r": [_issue(1, 10)]}, comments={("o/r", 10): [{"id": 99, "body": "c"}]})
    svc = _FakeSvc(existing={"issue:1"})
    r = import_github_impl(client=gh, service=svc, profile="work", repos=["o/r"])
    assert r["skipped_duplicate"] == 1  # issue:1 skipped
    assert {rec.metadata["github_key"] for rec in svc.stored} == {"comment:99"}


def test_impl_multi_repo_totals():
    gh = _FakeGH(issues={"a/b": [_issue(1, 1)], "c/d": [_issue(2, 1)]})
    svc = _FakeSvc()
    r = import_github_impl(client=gh, service=svc, profile="work", repos=["a/b", "c/d"])
    assert r["stored"] == 2


def test_impl_comment_fetch_error_continues():
    gh = _FakeGH(issues={"o/r": [_issue(1, 10)]}, raise_comments_for={10})
    svc = _FakeSvc()
    import_github_impl(client=gh, service=svc, profile="work", repos=["o/r"])
    # the issue is still stored even though its comment fetch raised
    assert {rec.metadata["github_key"] for rec in svc.stored} == {"issue:1"}


class _PagedGH:
    """Returns scripted (items, next_page) tuples per page for issues and comments."""

    def __init__(self, issue_pages, comment_pages):
        self._issue_pages = issue_pages
        self._comment_pages = comment_pages

    def list_issues(self, repo, since=None, page=1, per_page=100):
        return self._issue_pages[page - 1]

    def list_comments(self, repo, issue_number, page=1, per_page=100):
        return self._comment_pages[page - 1]


def test_impl_paginates_issues_and_comments():
    # page1 has an issue -> next_page 2; page2 empty -> stop
    gh = _PagedGH(
        issue_pages=[([_issue(1, 10)], 2), ([], None)],
        comment_pages=[
            ([{"id": 91, "body": "c1"}], 2),
            ([{"id": 92, "body": "c2"}], None),
        ],  # 2 comment pages for issue 10
    )
    svc = _FakeSvc()
    import_github_impl(client=gh, service=svc, profile="work", repos=["o/r"])
    keys = {rec.metadata["github_key"] for rec in svc.stored}
    # issue-loop + comment-loop both aggregated + terminated
    assert keys == {"issue:1", "comment:91", "comment:92"}


class _FailingRepoGH(_FakeGH):
    def list_issues(self, repo, since=None, page=1, per_page=100):
        if repo == "bad/repo":
            raise RuntimeError("404 not found")
        return super().list_issues(repo, since, page, per_page)


def test_impl_bad_repo_does_not_sink_others():
    gh = _FailingRepoGH(issues={"good/repo": [_issue(1, 1)]})
    svc = _FakeSvc()
    repos = ["bad/repo", "good/repo"]
    r = import_github_impl(client=gh, service=svc, profile="work", repos=repos)
    assert r["errors"] == 1  # the bad repo is counted
    # good repo still stored
    assert {rec.metadata["github_key"] for rec in svc.stored} == {"issue:1"}


def test_cli_missing_token_errors(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["import-github", "--profile", "work"])
    assert result.exit_code == 1
    assert "GITHUB_TOKEN" in result.output


def test_cli_missing_repos_errors(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "TOKEN")
    monkeypatch.delenv("GITHUB_REPOS", raising=False)
    result = CliRunner().invoke(app, ["import-github", "--profile", "work"])
    assert result.exit_code == 1
    assert "GITHUB_REPOS" in result.output


def test_cli_dry_run_reports_counts(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "TOKEN")
    monkeypatch.setenv("GITHUB_REPOS", "o/r")
    monkeypatch.setattr("ogham.cli.GitHubClient", lambda token: _FakeGH(issues={"o/r": []}))
    monkeypatch.setattr("ogham.cli.DefaultIngestService", _FakeSvc)
    result = CliRunner().invoke(app, ["import-github", "--profile", "work", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "scanned=0" in result.output
