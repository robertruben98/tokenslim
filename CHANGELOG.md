# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-09

First tagged release and PyPI publication.

### Changed

- **Distribution renamed to `tokenslim-ai` on PyPI.** The bare `tokenslim`
  name is taken by an unrelated project, so installing that name would fetch
  someone else's package. Install with `pip install tokenslim-ai`. The
  importable package (`import tokenslim`) and the `tokenslim` CLI command are
  unchanged.
- Updated every install reference (README, `install.sh`, `install.ps1`,
  `docs/`, and in-code extras hints) to `tokenslim-ai`.

### Added

- `release.yml` workflow: pushing a `v*` tag builds the package with hatchling,
  publishes to PyPI via Trusted Publishing, and creates a GitHub Release whose
  notes come from this changelog.
- This changelog.

## [0.3.0]

- Pre-changelog baseline: the milestone-1 feature set (content-type detection,
  per-type compressors, reversible CCR store, semantic + prefix caching, CLI,
  and framework integrations). Never published to PyPI.

[0.4.0]: https://github.com/robertruben98/tokenslim/releases/tag/v0.4.0
