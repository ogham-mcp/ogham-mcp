# shared-data changelog

The shared-data version stream is independent of the Python package
version and the Go CLI version. Each entry below corresponds to a
`shared-data-vX.Y.Z` tag in the public `ogham-mcp` repo.

## 0.1.0 (unreleased)

- Initial extraction of `hooks_config.yaml` from
  `src/ogham/hooks_config.yaml` in the Python repo. Content is
  byte-identical to the v0.14.3 source; the move is structural only.
- Schema lock: `regex_dialect: re2` declared in `schema.yaml`. All 45
  secret-detection patterns verified RE2-compatible at extraction
  time.
- Consumers (Python via `importlib.resources`, Go via `//go:embed`
  through a `git subtree` vendor) wire up in subsequent commits;
  parity-hash CI gates land alongside.
