# Changelog

All notable changes to this project are documented here. The format is inspired by Keep a Changelog.

## [Unreleased]

### Removed

- `vdapseisutils.core.swarmmpl2` and `vdapseisutils.core.swarmmpl3` packages.
- `vdapseisutils.core.swarmmpl.v2` and `.v3` subpackages (implementation moved to top-level `core/swarmmpl/` modules).

### Changed

- Panel-based multi-trace layout: `SwarmClipboard` and v3 helpers live in `vdapseisutils.core.swarmmpl.clipboard`, `panel.py`, `timeaxes.py`, and `convenience.py` (no separate v3 import path).
- `vdapseisutils.core.swarmmpl` no longer emits a package-level deprecation warning on import.

## [0.2.0] — 2026-04-01

### Added

- `vdapseisutils.core.swarmmpl.v2` and `vdapseisutils.core.swarmmpl.v3` subpackages as the single on-disk homes for the former `swarmmpl2` clipboard helpers and `swarmmpl3` time-axis / panel stack (API v1 §13.4).

### Changed

- `vdapseisutils.plot.swarm` now imports v3 types only from `vdapseisutils.core.swarmmpl` (no direct dependency on shim module paths for first-party code).
- Legacy `swarmmpl2` and `swarmmpl3` packages are thin re-export shims over `core.swarmmpl.v2` / `v3`; scheduled removal no earlier than **v0.3.0** (deprecation messages updated).

### Deprecated

- `vdapseisutils.core.maps.maps` remains a **deprecated re-export stub** (`DeprecationWarning` on import). Prefer `vdapseisutils.core.maps` (package) or `core.maps.map`, `.utils`, etc. (API v1 §13.5).
- `vdapseisutils.core.swarmmpl`, `core.swarmmpl2`, and `core.swarmmpl3` import paths remain deprecated in favor of `vdapseisutils.plot.swarm`.

### Removed

- Unused draft modules under `core/swarmmpl/`: `clipboard_001.py`, `heli_dev_fig.py`, `heli_v2.py`, `heli_v3.py`.
- `core/swarmmpl3/test.py` (large standalone test script; not part of the public API).
- Backup tree `core/maps.bak/` moved to `archive/maps.bak/` locally (ignored by `*.bak`-style rules if present under that path).

### Notes

- Pyplot registration: `vdapseisutils.plot.mpl.register_pyplot()` (API v1 §3).
- Backend-neutral helpers: `vdapseisutils.compute` (catalog, waveforms, maps scaffold).

See `.local/api-v1-coord/API_V1_CANONICAL.md` §8–§9 and §13.4–§13.5 for policy.

[0.2.0]: https://github.com/jwellik/vdapseisutils/compare/v0.1.0...v0.2.0
