# Rollback migrations — danger zone

Each file here **undoes** a forward migration in `../`. They exist for
development and recovery. Running one on a healthy production DB will
destroy state.

## Why they are prefixed `DANGER_`

So they are visually obvious in a file listing, and so no one
automates `psql < rollback.sql` without reading the warnings in the
file header.

## Why each has a session-variable guard

Piping `psql $DATABASE_URL < DANGER_xxx.sql` fails **by design** with a
`Rollback refused` error. The file only does work if the caller set
`ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING'` in the same session.

## To intentionally run a rollback

```bash
psql "$DATABASE_URL" <<'EOF'
SET ogham.confirm_rollback = 'I-KNOW-WHAT-I-AM-DOING';
\i sql/migrations/rollback/DANGER_026_memory_lifecycle_split.sql
EOF
```

The `SET` is session-scoped — it does not persist, does not affect
concurrent sessions, does not leak.

## What each rollback does

| File | Reverts | Data impact |
|------|---------|-------------|
| `DANGER_025_memory_lifecycle.sql` | Forward migration 025 | Drops `memories.stage`, `memories.stage_entered_at`, `profile_settings.decay_lambda`, `profile_settings.decay_beta`. Data in those columns is lost. |
| `DANGER_026_memory_lifecycle_split.sql` | Forward migration 026 | Re-adds stage cols to `memories`, backfills from `memory_lifecycle`, drops the `memory_lifecycle` table + triggers. Lifecycle history preserved via backfill. Python server v0.11.0+ will fail after this — downgrade the server to v0.10.x first. |
| `DANGER_044_unforce_rls_non_supabase.sql` | Forward migration 044 | Re-applies `FORCE ROW LEVEL SECURITY` to `entities`/`memory_entities`/`entity_edges`/`entity_edge_predicates`/`entity_aliases`. On a non-Supabase install this **re-introduces the TBU-163 lockout** (deny-all for the non-superuser table owner) — no data is lost, but the tables become unreadable/unwritable again. No legitimate reason to run this outside testing the migration harness itself. |

## Recovering from an accidental rollback

Forward migrations are idempotent. Re-apply them:

```bash
./sql/upgrade.sh "$DATABASE_URL"
```

For 026 specifically, all memories will land at `stage='fresh'` again —
real lifecycle state (STABLE / EDITING transitions) is not recovered
because the forward migration backfills from `memories.stage` which the
rollback dropped. Consider snapshotting `memory_lifecycle` before any
rollback.
