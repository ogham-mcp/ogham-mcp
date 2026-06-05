# shared/ — Cross-stack data artifact

Versioned data consumed by both the Python `ogham-mcp` server and the Go
`ogham-cli` client. Edit here. Do not edit consumer-side copies.

## Files

| File | Purpose |
|------|---------|
| `schema.yaml` | Schema version + dialect lock for regex patterns. |
| `hooks_config.yaml` | Signal/noise filtering rules + secret-masking regex patterns. |
| `CHANGELOG.md` | Per-version change history for the shared-data tag stream. |

## Consumption contract

- **Python (`ogham-mcp`)** loads via `importlib.resources` from the
  in-tree copy. The dev repo's `Makefile sync` target rsyncs `shared/`
  to the public repo; the public repo's `pyproject.toml` packages
  `shared/` into the wheel.
- **Go (`ogham-cli`)** vendors a copy at `internal/native/shared/`
  using `git subtree pull --prefix=internal/native/shared <public-url>
  shared-data-vX.Y.Z --squash`. The files are read at build time via
  `//go:embed shared/*`.

## Drift prevention

Both consumers run a CI gate that computes a SHA-256 manifest of the
embedded copy and compares it against the manifest pinned in the
`shared-data-vX.Y.Z` tag the consumer last vendored from. Any mismatch
fails CI with a remediation hint.

## Regex dialect

All regex patterns in this directory must compile in **RE2** (Go's
default regex engine; Python's `re` accepts the same subset). RE2
forbids backreferences and variable-length lookarounds. The
`shared/schema.yaml` `regex_dialect: re2` field is the contract; both
consumers' parity tests parse every `pattern:` value with RE2 and fail
on any incompatible expression.

## Cutting a new shared-data version

The shared-data version stream is **independent** of the Python
package version (`ogham-mcp 0.X.Y`) and the Go CLI version
(`ogham-cli vX.Y.Z`). It ships via its own Makefile targets:

```
# In the dev repo, edit shared/* and bump shared/schema.yaml + CHANGELOG.md
make sync-shared                     # mirror dev/shared/ → public/shared/
cd $(PUB_REPO) && git diff shared/   # review
cd $(PUB_REPO) && git add shared/ && git commit -m "shared-data v0.1.0"
make shared-tag VERSION=0.1.0        # tags public as shared-data-v0.1.0
cd $(PUB_REPO) && git push origin main && git push origin shared-data-v0.1.0
```

Consumers then bump:

```
# Go (ogham-cli)
git subtree pull --prefix=internal/native/shared $(PUB_REPO) shared-data-v0.1.0 --squash

# Python (ogham-mcp) — picks up shared/ in the next make sync TAG=vX.Y.Z
```

## Why this exists

Prior to v0.9 the same patterns lived in `src/ogham/hooks_config.yaml`
in the Python repo, with a header comment instructing maintainers to
keep parallel copies in sync by hand. That convention had already
failed empirically for `languages/*.yaml` (the two repos drifted to
~100-line differences on `en.yaml` alone before the v0.9 council
caught it). Promoting `hooks_config.yaml` to a versioned artifact with
a CI parity gate prevents the same failure mode for security-critical
secret-detection patterns, where a missed regex on one side leaks
credentials into memory.

Language YAMLs (`languages/`) will follow this same mechanism in a
follow-up; they need a separate decision on union-vs-namespace because
both consumers have legitimately divergent keys (Python's
`wiki_compile` strings, Go's `recurrence_patterns` etc.).
