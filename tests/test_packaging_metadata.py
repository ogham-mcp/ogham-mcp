"""Guards on the `[project]` metadata that becomes the PyPI page (TBU-201).

For most visitors the PyPI page is the first and only page they see. Until
0.17.1 it showed a one-line summary and nothing else -- no README body, no
repository link, no docs link -- because `[project]` declared only name,
version, description, requires-python and dependencies. The published wheel's
METADATA carried a zero-byte description, so this was baked in at build time
rather than being a PyPI display problem.

Two properties are worth defending, and they fail differently:

1. The fields exist in pyproject.toml. Checked here.
2. They survive into the built wheel's METADATA. Checked in CI against the
   real artifact (`.github/workflows/publish.yml`), because only a build can
   prove that.

This file also guards a sync hazard specific to this project: `make sync`
does NOT copy pyproject.toml, so the dev and public repos drift by default
and have done so before (v0.10.2, v0.10.1). These tests run in both repos
against whichever pyproject.toml sits above them, so a public repo that
never got the fields fails its own suite.

As elsewhere in this suite, the negative tests are the point: a check nobody
has watched fail is indistinguishable from a comment.
"""

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Anything less and the "description" is a stub, not a README. The real one is
# ~54k chars; this only has to be big enough to catch an empty or placeholder
# file being wired up by accident.
MIN_README_CHARS = 2000

REQUIRED_URL_KEYS = {"Homepage", "Repository", "Issues"}


# --- pure rules ------------------------------------------------------------


def metadata_violations(project: dict) -> list[str]:
    """Report `[project]` keys that are missing or present-but-empty.

    Presence alone is not enough: `keywords = []` satisfies a naive `in` check
    while shipping exactly the metadata we are trying to stop shipping.
    """
    problems = []
    for key in ("readme", "license", "keywords", "classifiers", "urls"):
        if key not in project:
            problems.append(f"missing [project].{key}")
        elif not project[key]:
            problems.append(f"empty [project].{key}")
    return problems


def url_violations(urls: dict) -> list[str]:
    """Report project URLs that are absent, insecure, or malformed."""
    problems = []
    for key in sorted(REQUIRED_URL_KEYS - set(urls)):
        problems.append(f"missing [project.urls].{key}")
    for key, value in sorted(urls.items()):
        if not value.startswith("https://"):
            problems.append(f"{key} is not https: {value!r}")
        elif value != value.strip():
            problems.append(f"{key} has surrounding whitespace: {value!r}")
    return problems


def classifier_violations(classifiers: list[str], license_expr: str | None) -> list[str]:
    """A license expression and a license classifier must not both be declared.

    PEP 639 supersedes the `License ::` classifiers; declaring both is
    contradictory, and build backends reject it. Catching it here gives a
    readable failure instead of a backend traceback at release time.
    """
    problems = []
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    if license_expr and license_classifiers:
        problems.append(
            f"license expression {license_expr!r} declared alongside {license_classifiers}"
        )
    if not license_expr and not license_classifiers:
        problems.append("no license expression and no license classifier")
    return problems


# --- the real file ---------------------------------------------------------


@pytest.fixture(scope="module")
def project() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]


def test_project_metadata_is_complete(project):
    assert metadata_violations(project) == []


def test_project_urls_are_usable(project):
    assert url_violations(project["urls"]) == []


def test_license_is_declared_exactly_once(project):
    license_expr = project.get("license") if isinstance(project.get("license"), str) else None
    assert classifier_violations(project.get("classifiers", []), license_expr) == []


def test_dependencies_did_not_fall_into_a_subtable(project):
    """`[project.urls]` placed above `dependencies` silently swallows it.

    TOML binds every key after a sub-table header to that sub-table. Writing
    the urls block in the middle of `[project]` therefore moves `dependencies`
    into `[project.urls]` and produces a wheel that installs nothing -- with no
    error from either the parser or the build. Caught during TBU-201.
    """
    assert isinstance(project.get("dependencies"), list)
    assert len(project["dependencies"]) >= 10
    assert "dependencies" not in project["urls"]


def test_declared_readme_exists_and_has_content(project):
    readme = REPO_ROOT / project["readme"]
    assert readme.is_file(), f"{project['readme']} is declared but absent"
    assert len(readme.read_text()) >= MIN_README_CHARS


def test_declared_license_files_exist(project):
    for pattern in project["license-files"]:
        assert list(REPO_ROOT.glob(pattern)), f"license-files pattern matched nothing: {pattern}"


def test_sdist_ships_the_files_the_metadata_points_at(project):
    """A `readme`/`license-files` the sdist excludes breaks the sdist build."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        include = tomllib.load(fh)["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert project["readme"] in include
    for pattern in project["license-files"]:
        assert pattern in include


# NOTE: "the README contains no relative links" is deliberately NOT tested here.
# PyPI renders the description off pypi.org, so relative links 404 there -- but
# this repo's README is dev-local by design (CLAUDE.md; excluded from sync) and
# points at files that exist only here (BACKLOG.md, tests/BENCH.md, docs/images/).
# Those cannot be rewritten to absolute URLs without inventing public paths.
# The rule therefore belongs where the artifact is built, against the real
# description: `.github/workflows/publish.yml` in the public repo.
#
# Corollary worth remembering: the Makefile `make build` / `make publish`
# fallback builds from THIS repo, so using it would publish a description full
# of dead links. Another reason the OIDC path is primary (TBU-197/198/199).


# --- the negative cases ----------------------------------------------------


def test_missing_fields_are_reported():
    bare = {"name": "x", "version": "1", "dependencies": []}
    problems = metadata_violations(bare)
    assert len(problems) == 5
    assert "missing [project].readme" in problems
    assert "missing [project].urls" in problems


def test_present_but_empty_fields_are_reported():
    """The regression this is really guarding: declared, and still useless."""
    hollow = {"readme": "", "license": "", "keywords": [], "classifiers": [], "urls": {}}
    problems = metadata_violations(hollow)
    assert len(problems) == 5
    assert all(p.startswith("empty ") for p in problems)


def test_http_url_is_rejected():
    problems = url_violations(
        {
            "Homepage": "http://ogham-mcp.dev",
            "Repository": "https://github.com/ogham-mcp/ogham-mcp",
            "Issues": "https://github.com/ogham-mcp/ogham-mcp/issues",
        }
    )
    assert problems == ["Homepage is not https: 'http://ogham-mcp.dev'"]


def test_absent_required_url_is_reported():
    problems = url_violations({"Homepage": "https://ogham-mcp.dev"})
    assert "missing [project.urls].Repository" in problems
    assert "missing [project.urls].Issues" in problems


def test_license_declared_twice_is_rejected():
    problems = classifier_violations(["License :: OSI Approved :: MIT License"], "MIT")
    assert problems and "alongside" in problems[0]


def test_license_declared_nowhere_is_rejected():
    assert classifier_violations(["Intended Audience :: Developers"], None) == [
        "no license expression and no license classifier"
    ]


def test_license_via_classifier_only_is_accepted():
    """The pre-PEP-639 form is still valid; the rule must not demand ours."""
    assert classifier_violations(["License :: OSI Approved :: MIT License"], None) == []
