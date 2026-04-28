"""Pre-migration parity baseline.

Establishes what we expect BEFORE any lifecycle columns exist. After
migration 025 lands, this same test must still pass with the new
columns returning defaults (stage='fresh', stage_entered_at=created_at).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ogham.service import store_memory_enriched

DEFAULT_GO_BIN = Path("/Users/kevinburns/Developer/web-projects/ogham-cli/ogham")
GO_BIN = os.environ.get("OGHAM_GO_BIN") or (str(DEFAULT_GO_BIN) if DEFAULT_GO_BIN.exists() else "")


def _can_connect() -> bool:
    """Check if Postgres backend is configured and reachable."""
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()
        backend._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(not _can_connect(), reason="Postgres backend not configured or unreachable"),
    pytest.mark.skipif(not GO_BIN, reason="Go Ogham CLI unavailable; set OGHAM_GO_BIN"),
]


def test_python_write_go_read_parity(pg_test_profile):
    result = store_memory_enriched(
        content="parity baseline test",
        profile=pg_test_profile,
        source="test",
        tags=["type:parity"],
    )
    mid = str(result["id"])

    out = subprocess.check_output(
        [
            GO_BIN,
            "search",
            "parity baseline",
            "--profile",
            pg_test_profile,
            "--limit",
            "1",
            "--json",
        ],
        text=True,
    )
    rows = json.loads(out)
    assert len(rows) == 1
    assert rows[0]["id"] == mid
    assert rows[0]["content"] == "parity baseline test"
    assert rows[0]["profile"] == pg_test_profile


def test_post_migration_parity(pg_fresh_db, pg_test_profile):
    """After migrations 025 + 026, a Python write + Go read still matches.

    Go is reading a table that went through both migrations -- 025 added
    stage columns, 026 dropped them (moved to memory_lifecycle). If the
    Go pgx query path uses explicit column selection it works. If it
    uses SELECT * it breaks here -- and that would be a v0.4.0-grade
    mistake to discover in production, so we catch it NOW.
    """
    from pathlib import Path as _P

    mig_025 = _P(__file__).parent.parent / "sql/migrations/025_memory_lifecycle.sql"
    mig_026 = _P(__file__).parent.parent / "sql/migrations/026_memory_lifecycle_split.sql"
    pg_fresh_db.apply_sql(mig_025)
    pg_fresh_db.apply_sql(mig_026)

    # Reuse the body of test_python_write_go_read_parity.
    # We can't just call it -- fixtures are per-test. So repeat the
    # logic here.
    result = store_memory_enriched(
        content="post-migration parity check",
        profile=pg_test_profile,
        source="test",
        tags=["type:parity", "phase:post-migration"],
    )
    mid = str(result["id"])
    out = subprocess.check_output(
        [
            GO_BIN,
            "search",
            "post-migration parity",
            "--profile",
            pg_test_profile,
            "--limit",
            "1",
            "--json",
        ],
        text=True,
    )
    rows = json.loads(out)
    assert len(rows) == 1, f"expected 1 row from Go, got {len(rows)}: {out[:200]}"
    assert rows[0]["id"] == mid
    assert rows[0]["content"] == "post-migration parity check"
    assert rows[0]["profile"] == pg_test_profile
