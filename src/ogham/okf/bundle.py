"""OKF bundle (directory) read/write orchestration."""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ogham.okf.concept import frontmatter_to_memory, memory_to_frontmatter
from ogham.okf.identity import make_filename
from ogham.okf.serialization import read_concept, write_concept

_OKF_VERSION = "0.1"


def write_index(bundle_dir: Path, manifest: dict) -> None:
    """Write the bundle-root index.md with okf_version declaration.

    Per OKF spec §11 + §6: the bundle-root index.md is the ONLY index.md where
    frontmatter is permitted, and is where the supported OKF version is declared.
    """
    frontmatter = {"okf_version": _OKF_VERSION, **manifest}
    body = (
        "# Memories\n\n"
        "This bundle was produced by Ogham. "
        "See individual concept files in `memories/`.\n"
    )
    write_concept(bundle_dir / "index.md", frontmatter, body)


def filter_expired(memories: list[dict]) -> list[dict]:
    """Drop memories whose expires_at is in the past.

    Memories with no expires_at or None are kept (no expiration). Default
    behaviour on export per the v0.15 design decision.
    """
    now = datetime.now(timezone.utc)
    kept = []
    for m in memories:
        expires = m.get("expires_at")
        if not expires:
            kept.append(m)
            continue
        try:
            # Postgres backend (psycopg) returns expires_at as a real datetime
            # object; the Supabase REST backend returns an ISO string (#TBU-162
            # audit -- same root cause as the hybrid_search datetime crash).
            if isinstance(expires, datetime):
                ts = expires
            else:
                ts = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        except (ValueError, AttributeError, TypeError):
            kept.append(m)  # unparseable = keep (safe default)
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts > now:
            kept.append(m)
    return kept


def export_okf_bundle(
    memories: list[dict[str, Any]],
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    include_viewer: bool = False,
) -> None:
    """Export a list of memories to an OKF v0.1 bundle directory.

    Atomicity guarantee:
    - Fresh targets (bundle_dir absent): os.rename(staging, bundle_dir) is a
      single syscall -- either the target exists with full contents or it does not.
    - Existing targets: shutil.rmtree(bundle_dir) then shutil.move(staging, bundle_dir).
      Between those two syscalls a SIGKILL leaves the old bundle gone and the new
      bundle still in the temp staging directory. The TemporaryDirectory context manager
      cleans up staging on normal exit but NOT after SIGKILL, so the new bundle may
      be orphaned in a `.okf-tmp-*` sibling. True crash-safety for in-place updates
      would require a backup-rename pattern; this is not needed for v1 self-hosted usage.

    Pre-existing target is replaced, not merged.
    Filters expired memories by default per v0.15 design decision.

    If include_viewer is True, also writes a self-contained viewer.html at the
    bundle root after the atomic rename completes. The viewer is regenerated
    fresh on every export -- it never partially updates.
    """
    bundle_dir = Path(bundle_dir)
    fresh = filter_expired(memories)

    # Use a sibling temp dir so the final rename is on the same filesystem.
    parent = bundle_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".okf-tmp-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        write_index(staging, manifest)
        memories_dir = staging / "memories"
        memories_dir.mkdir()
        for memory in fresh:
            fm = memory_to_frontmatter(memory)
            body = memory.get("content") or ""
            filename = make_filename(memory)
            write_concept(memories_dir / filename, fm, body)
        # Atomic replace
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        shutil.move(str(staging), str(bundle_dir))

    if include_viewer:
        from ogham.okf.viewer import build_viewer

        build_viewer(bundle_dir)


_RESERVED_FILENAMES = {"index.md", "log.md"}


def import_okf_bundle(bundle_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read an OKF bundle directory into a list of memory dicts + stats.

    Stats includes `total` (concepts found) and `missing_id_count` (concepts
    that arrived without our id extension -- they will become NEW memories,
    not upserts, when the caller writes them).

    Reserved filenames (index.md, log.md per spec §3.1) are skipped at every
    directory level.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise ValueError(f"{bundle_dir} is not a directory")

    memories: list[dict] = []
    missing_id_count = 0
    skipped_count = 0
    for md_path in sorted(bundle_dir.rglob("*.md")):
        if md_path.name in _RESERVED_FILENAMES:
            continue
        try:
            fm, body = read_concept(md_path)
        except ValueError:
            # Malformed concept -- skip but surface the drop in stats so the
            # import tool can warn the operator.
            skipped_count += 1
            continue
        memory = frontmatter_to_memory(fm, body)
        if memory["id"] is None:
            missing_id_count += 1
        memories.append(memory)

    stats = {
        "total": len(memories),
        "missing_id_count": missing_id_count,
        "skipped_count": skipped_count,
    }
    return memories, stats
