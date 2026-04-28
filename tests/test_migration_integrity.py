"""Upgrade-path migration integrity guards.

Defends against the v0.9.2 regression class: a stray or misnumbered migration
file sorting after the canonical RRF fix and silently overwriting it. See
CHANGELOG [0.9.2] for the full incident write-up.

Also enforces dual-tree parity: every file in ``src/ogham/sql/migrations/``
(the package-bundled tree that test fixtures load) must be byte-identical to
its counterpart in ``sql/migrations/`` (the canonical tree that ``upgrade.sh``
applies). Closes the v0.12-era hazard where a fix to one tree silently
diverged from the other.
"""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "sql" / "migrations"
PACKAGE_MIGRATIONS_DIR = REPO_ROOT / "src" / "ogham" / "sql" / "migrations"
ROLLBACK_DIR = MIGRATIONS_DIR / "rollback"


def _top_level_migrations() -> list[Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())


def test_no_unnumbered_migrations():
    """Every top-level migration filename must start with a digit.

    `sql/upgrade.sh` applies migrations in alphabetical order. Any .sql file
    whose name begins with a non-digit character (e.g. ``update_search_function.sql``)
    sorts after numbered migrations and silently overrides them.
    """
    offenders = [p.name for p in _top_level_migrations() if not p.name[0].isdigit()]
    assert not offenders, (
        f"Unnumbered migration(s) found in sql/migrations/: {offenders}. "
        "Unnumbered files sort after numbered ones and override them on upgrade."
    )


def test_later_hybrid_search_migrations_preserve_rrf():
    """Later migrations may evolve hybrid search, but must keep true RRF.

    017 introduced the v0.9.2 RRF fix. Later migrations such as 021 may
    legitimately re-define hybrid_search_memories, but they must preserve
    the position-based RRF formula rather than silently reintroducing the
    older raw-score fusion regression.
    """
    migrations = _top_level_migrations()
    rrf_fix = next((p for p in migrations if p.name == "017_rrf_bm25.sql"), None)
    assert rrf_fix is not None, "expected 017_rrf_bm25.sql at top-level sql/migrations/"

    later = [p.name for p in migrations if p.name > rrf_fix.name]
    broken_pattern = "semantic_weight * coalesce(s.similarity"

    for name in later:
        content = (MIGRATIONS_DIR / name).read_text().lower()
        if "create or replace function hybrid_search_memories" not in content:
            continue
        assert "1.0 / (rrf_k + coalesce(" in content, (
            f"{name} redefines hybrid_search_memories but does not preserve "
            "the true Reciprocal Rank Fusion formula"
        )
        assert broken_pattern not in content, (
            f"{name} redefines hybrid_search_memories with the broken raw-score fusion pattern"
        )


def test_017_rrf_bm25_is_functional_and_uses_rrf():
    """017 must contain a real RRF formula, not just a docs comment.

    The v0.8.3–v0.9.1 version of this file was comment-only. The v0.9.2 rewrite
    restores it to a functional migration with position-based RRF.
    """
    content = (MIGRATIONS_DIR / "017_rrf_bm25.sql").read_text()
    assert "create or replace function hybrid_search_memories" in content.lower(), (
        "017_rrf_bm25.sql must define hybrid_search_memories, not just document it"
    )
    assert "1.0 / (rrf_k + coalesce(" in content, (
        "017_rrf_bm25.sql must use true Reciprocal Rank Fusion: "
        "1.0 / (rrf_k + rank_ix), not raw-score linear combination"
    )
    broken_pattern = "semantic_weight * coalesce(s.similarity"
    assert broken_pattern not in content, (
        "017_rrf_bm25.sql contains the broken raw-score fusion pattern"
    )


def test_update_search_function_sql_does_not_exist():
    """The v0.9.1-era stray migration must stay removed."""
    stray = MIGRATIONS_DIR / "update_search_function.sql"
    assert not stray.exists(), (
        "sql/migrations/update_search_function.sql was removed in v0.9.2 "
        "because it silently overrode true RRF. Do not reintroduce it."
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dual_tree_parity():
    """Every file in src/ogham/sql/migrations/ must hash-match its canonical twin.

    Two trees of migrations exist in the repo:
      * ``sql/migrations/``                   -- canonical, what ``upgrade.sh`` reads
      * ``src/ogham/sql/migrations/``         -- bundled in the wheel, what tests load

    Forward migrations (NN_*.sql) live at the top level of both. Rollback
    migrations (DANGER_NN_*.sql) live inline in the package tree but in a
    ``rollback/`` subdir of canonical. We resolve both layouts here.

    A v0.12-era release shipped with these trees silently drifted -- a fix
    applied to one was missed in the other. This guard makes that class of
    drift impossible to merge.
    """
    if not PACKAGE_MIGRATIONS_DIR.is_dir():
        return  # package tree absent -- nothing to compare (Phase B has landed)

    drift = []
    missing = []
    for pkg_file in sorted(PACKAGE_MIGRATIONS_DIR.glob("*.sql")):
        if pkg_file.name.startswith("DANGER_"):
            canonical = ROLLBACK_DIR / pkg_file.name
        else:
            canonical = MIGRATIONS_DIR / pkg_file.name

        if not canonical.exists():
            missing.append(f"{pkg_file.name} (expected at {canonical.relative_to(REPO_ROOT)})")
            continue

        if _hash(pkg_file) != _hash(canonical):
            drift.append(
                f"  {pkg_file.relative_to(REPO_ROOT)}\n"
                f"    vs {canonical.relative_to(REPO_ROOT)}\n"
                f"    -> sha256 mismatch"
            )

    if missing:
        raise AssertionError(
            "Package-tree migrations missing canonical twin in sql/migrations/:\n  "
            + "\n  ".join(missing)
            + "\n\nReconcile by copying the canonical version, or remove the orphan."
        )
    if drift:
        raise AssertionError(
            "Dual-tree migration drift detected -- canonical and package trees disagree:\n"
            + "\n".join(drift)
            + "\n\nFix: copy the canonical sql/migrations/<file>.sql over the package "
            "src/ogham/sql/migrations/<file>.sql so they hash-match. The canonical "
            "tree is the source of truth."
        )
