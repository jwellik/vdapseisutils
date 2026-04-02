# `obspy_ext` migration — subtasks and copy-paste prompts

This document breaks the ObsPy extension reorganization into **ordered subtasks**. Each section includes **copy-paste text** you can drop into a separate agent window.

**Repository:** `vdapseisutils`  
**End state:** Top-level package `vdapseisutils/obspy_ext/` (sibling to `core/`), replacing the scattered `utils/obspyutils/` + `core/datasource/` patterns, with deprecation shims where needed.

### T0 — Locked team conventions (summary)

These three rules apply to **all** follow-on tasks (T1–T12); details below expand them.

1. **Package layout:** New code lives under `vdapseisutils/obspy_ext/` (not new growth in `utils/obspyutils/` or `core/datasource/` for greenfield ObsPy extension work).
2. **Client naming:** `DataSource` is a **public alias** of `VClient` (one class, two names).
3. **Catalog indexing:** `VCatalog.events` holds **real** `VEvent` instances; `VCatalog.__getitem__(i)` returns `self.events[i]` (slice → `VCatalog`). **No** proxy pattern or `_original_event` / dual-object indexing.

The module docstring in `vdapseisutils/obspy_ext/__init__.py` repeats this for developers at import time.

---

## Agreed design decisions (do not re-litigate in sub-windows)

### 1. `VClient` + `DataSource` alias

- **One facade class:** `VClient` in `obspy_ext`.
- **Public alias:** `DataSource = VClient` (same class or assignment after class definition).
- **Construction:** Accept both (a) current `VClient` heuristics (path, short name, port-based detection, `client_type=`) and (b) **URI-style** strings from the old `DataSource` (`fdsnws://`, `sds://`, `waveserver://`, `seedlink://`, `neic://`, …).
- **Internals:** Hold the underlying ObsPy client (`self._client` or `self.client` — pick one public name, document the other as property).
- **Waveforms:** All `get_waveforms` paths (single NSLC, list, bulk when applicable) go through **one** internal module, e.g. `obspy_ext/client/_fetch.py`, migrated from `core/datasource/clientutils.py` (chunking, empty traces, merge/dtype behavior). **No duplicate** chunking logic in `VClient` vs old `DataSource`.
- **Return types:** `get_stations` → `VInventory`, `get_events` → `VCatalog`, `get_waveforms` (and helpers) → `VStream` once aligned with `read()` wrappers.

### 2. `VCatalog.__getitem__` / `VEvent` — long-term clean model

**Chosen approach: canonical `VEvent` instances in the catalog list (single object identity).**

- Every event stored in `VCatalog.events` should be a **`VEvent` subclass of `obspy.core.event.Event`** (not a proxy with copied attributes).
- **`__getitem__(i)`** returns **`self.events[i]`** directly (or the slice equivalent as `VCatalog`), with **no** `_original_event` / `_parent_catalog` attribute mirroring.
- When wrapping an existing `Catalog`, **convert once** at `VCatalog` construction (and on `append` / `+=` / similar) so plain `Event` objects become `VEvent` instances **in place** in the internal list.
- Remove or avoid the current pattern that copies `dir(event)` attributes onto a new object while keeping a hidden reference to the “real” event.

Implementation detail is up to the implementer (e.g. shallow copy of ObsPy sub-objects into a new `VEvent`, or ObsPy-supported copy constructors), but the **invariant** is: **what lives in `catalog.events` is what you get by index**, and it is always a `VEvent`.

### 3. `VStreamID` (rename of `waveID`)

- Subclass of `WaveformStreamID`; canonical name **`VStreamID`**.
- Keep **`waveID` as a deprecated alias** (warning optional) for one or more releases.

### 4. I/O convenience

- **`read()`** → ObsPy `read` + wrap as **`VStream`**.
- **`read_events()`** → ObsPy `read_events` + wrap as **`VCatalog`**.
- Optionally **`read_inventory()`** → **`VInventory`** for symmetry.

---

## Subtask order (recommended)

Work **in order** where dependencies exist; some tasks can run in parallel after **T1** and **T2** land.

| ID | Title |
|----|--------|
| T0 | Coordinator: branch, conventions, and “definition of done” |
| T1 | Create `obspy_ext` package skeleton + public `__all__` |
| T2 | Add `VStreamID` + deprecate `core.datasource.waveID` imports |
| T3 | Move `clientutils` → `obspy_ext/client/_fetch.py` + unify imports |
| T4 | Implement merged `VClient` (`DataSource` alias) using `_fetch` + URI parsing |
| T5 | Wire `VClient.get_waveforms` / bulk / inventory helpers → `VStream` / `V*` returns |
| T6 | Add `read`, `read_events` (+ optional `read_inventory`) in `obspy_ext/io.py` |
| T7 | Refactor `VCatalog` / `VEvent` indexing (canonical list, no proxy) |
| T8 | Relocate stream ops dedup (`streamutils` / `stream/utils`) under `obspy_ext` |
| T9 | Relocate catalog / inventory extended types + thin shims from `utils/obspyutils` |
| T10 | Remove empty stubs; fix `obspyutils/__init__.py` and root imports |
| T11 | Deprecation shims: `core.datasource` → `obspy_ext`; `utils.obspyutils` re-exports |
| T12 | Tests + smoke import of public API |

---

## T0 — Coordinator: branch, conventions, definition of done

**Scope:** Open a branch; agree that subtasks T1–T12 use consistent naming; each sub-window ends with **tests or grep verification** and a **commit** (if your workflow uses per-task commits).

**Copy-paste for sub-window:**

```text
You are working in the vdapseisutils repo on the obspy_ext migration.

Read docs/obspy_ext_subtasks.md for full context and design decisions.

Your role (T0): Create a git branch named e.g. feature/obspy-ext-migration, ensure the team conventions are clear: (1) new code lives under vdapseisutils/obspy_ext/; (2) VClient has alias DataSource; (3) VEvent is stored as real instances in VCatalog.events with __getitem__ returning self.events[i]—no proxy pattern.

Do not implement application features beyond scaffolding unless a later task asks. Push branch if remotes exist and the user wants that.

Commit with a clear message when done.
Use commit trailer: Co-authored-by: Pasquale
```

---

## T1 — Create `obspy_ext` package skeleton

**Scope:** Add `vdapseisutils/obspy_ext/__init__.py` with a documented public API placeholder (re-export nothing broken yet, or minimal exports). Add empty subpackages as needed: `obspy_ext/client/`, optional `obspy_ext/io/`. No large moves yet.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T1: Create the top-level package vdapseisutils/obspy_ext/ as a sibling to vdapseisutils/core/.

Requirements:
- Add obspy_ext/__init__.py with module docstring describing purpose (ObsPy extensions, unified client, I/O helpers).
- Add obspy_ext/client/__init__.py (can be minimal).
- Optionally add obspy_ext/io/__init__.py stub for later read/read_events.
- Do not break existing imports; no need to wire everything yet.

Commit when done. Trailer: Co-authored-by: Pasquale
```

---

## T2 — `VStreamID` + migrate off `core.datasource.waveID`

**Scope:** Move/reimplement `waveID` as `VStreamID` under `obspy_ext` (e.g. `obspy_ext/stream_id.py`). Export from `obspy_ext.__init__`. Update internal imports (`VClient`, `clientutils`, inventory/catalog code) to use `obspy_ext`. Leave `core/datasource/waveID.py` as a **shim** that imports and re-exports `VStreamID` / `waveID` with `DeprecationWarning` on import or on class access (choose one consistent approach).

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md (VStreamID / waveID section).

Task T2:
1. Add obspy_ext/stream_id.py: class VStreamID(WaveformStreamID) with behavior equivalent to current vdapseisutils.core.datasource.waveID.waveID (including parse_wave_id helper—relocate as private or public as appropriate).
2. Set waveID = VStreamID or a thin deprecated subclass; document.
3. Update all in-repo imports from vdapseisutils.core.datasource.waveID to obspy_ext (grep for waveID and core.datasource.waveID).
4. Replace core/datasource/waveID.py with a backward-compatible shim that re-exports and emits DeprecationWarning.

Run tests or minimal import smoke: python -c "from vdapseisutils.obspy_ext import VStreamID".

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T3 — Move waveform fetch helper to `obspy_ext/client/_fetch.py`

**Scope:** Move `get_waveforms_from_client` and any tightly coupled helpers from `core/datasource/clientutils.py` into `obspy_ext/client/_fetch.py`. Update `DataSource` (old class) and any callers to import from the new location. Preserve behavior (chunking, empty trace, dtype/merge quirks) — refactor only structure first.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T3: Migrate waveform download logic from vdapseisutils/core/datasource/clientutils.py into vdapseisutils/obspy_ext/client/_fetch.py (e.g. get_waveforms_from_client and private helpers). Keep behavior identical.

Update vdapseisutils/core/datasource/DataSource.py and any other callers to import from obspy_ext.client._fetch (or a thin public wrapper in obspy_ext.client if you prefer).

Leave clientutils.py as a shim re-exporting from obspy_ext with DeprecationWarning, or delete if no external users—grep the repo first.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T4 — Merged `VClient` with `DataSource` alias + URI construction

**Scope:** Port `utils/obspyutils/client.py` `VClient` into `obspy_ext/client/facade.py` (or `vclient.py`). Merge URI-based construction from `core/datasource/DataSource.__create_client` into the same class `__init__`. Expose `DataSource = VClient` at end of module or in `obspy_ext/__init__.py`. Preserve `__getattr__` delegation to underlying client.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md (VClient + DataSource section).

Task T4:
1. Implement VClient in obspy_ext (new module under obspy_ext/client/), combining:
   - Current VClient auto-detect logic from vdapseisutils/utils/obspyutils/client.py
   - URI/scheme parsing from vdapseisutils/core/datasource/DataSource.py (__create_client): fdsnws://, sds://, waveserver://, earthworm://, winston://, wws://, seedlink://, neic://
2. Single class; at module or package level set: DataSource = VClient
3. Document constructor modes in docstring with examples for both styles.
4. Replace or shim old utils/obspyutils/client.py to re-export from obspy_ext with deprecation warning if needed.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T5 — `VClient` methods: route all `get_waveforms` through `_fetch`; return `VStream`

**Scope:** Ensure `get_waveforms` / bulk paths call `_fetch.get_waveforms_from_client` (or a single internal orchestrator) instead of duplicating logic. Return `VStream(...)` consistently. `get_stations` / `get_events` already return `VInventory` / `VCatalog` — verify after moves.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T5:
1. In obspy_ext VClient, ensure every code path that downloads waveforms uses the shared helper in obspy_ext/client/_fetch.py (no duplicate chunking/empty-trace logic).
2. Return VStream for get_waveforms (and get_waveforms_bulk if applicable) by wrapping ObsPy Stream; import VStream from its canonical module (obspy_ext or interim path—follow current package layout after T9 if already moved).
3. Align get_waveforms_from_inventory (or equivalent) with the same return types.
4. Add minimal tests or a short script verifying VClient("IRIS") or a file path still works.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T6 — `read`, `read_events`, optional `read_inventory`

**Scope:** Add `obspy_ext/io.py` with thin wrappers around `obspy.read`, `obspy.read_events`, optionally `obspy.read_inventory`, returning `VStream`, `VCatalog`, `VInventory`. Export from `obspy_ext.__init__`.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T6: Add vdapseisutils/obspy_ext/io.py:
- read(*args, **kwargs) -> VStream(obspy.read(...))
- read_events(*args, **kwargs) -> VCatalog(obspy.read_events(...))
- read_inventory(*args, **kwargs) -> VInventory(obspy.read_inventory(...))  [optional but preferred]

Export these from obspy_ext/__init__.py. Docstring note: users who need plain ObsPy types should call obspy directly.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T7 — `VCatalog` / `VEvent`: canonical list, remove proxy `__getitem__`

**Scope:** Refactor `catalog/core.py` (or post-move location) so:

- `VCatalog` construction normalizes `events` to `VEvent` instances.
- `__getitem__` for int returns `self.events[i]` (typed as `VEvent`); slices return `VCatalog` wrapping the same event objects.
- Remove `_original_event` / `_parent_catalog` mirroring from the indexing path; adjust any methods that relied on them (`sort_picks`, etc.) to operate on `self` as a real `Event` subclass.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md (VCatalog / VEvent decision).

Task T7: Refactor VCatalog and VEvent for a single canonical object identity:
- VEvent subclasses obspy.core.event.Event.
- When building VCatalog from an existing Catalog, convert each Event to VEvent once and store in self.events (replace or wrap in a way that preserves ObsPy semantics—no dual proxy objects).
- VCatalog.__getitem__(int) must return self.events[i] directly; no copying attributes from a separate "original" event.
- Remove reliance on _original_event and _parent_catalog for indexing; fix sort_picks and any other methods that referenced them.
- Grep for VEvent( and __getitem__ in catalog package and update tests/callers.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T8 — Stream operations deduplication

**Scope:** Merge `utils/obspyutils/streamutils.py` and `utils/obspyutils/stream/utils.py` into one module under `obspy_ext` (e.g. `obspy_ext/stream/_ops.py`). `VStream.preprocess` / related methods should call this single implementation.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T8: Deduplicate stream helpers:
1. Merge streamutils.py and stream/utils.py into obspy_ext (single module).
2. Update VStream in stream/core.py (or obspy_ext after move) to use the shared functions internally.
3. Leave temporary shims in old paths with DeprecationWarning if external code might import them—grep repo.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T9 — Move extended types from `utils/obspyutils` into `obspy_ext`

**Scope:** Move `VCatalog` package, `VInventory`, `VStream`/`VTrace`, `VUTCDateTime`, and related mixins into `obspy_ext` subpackages (`catalog/`, `inventory/`, `stream/`, `time.py`, etc.). Update internal imports. This is a large mechanical move; prefer one PR or several commits by subpackage.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T9: Relocate ObsPy extension implementations from vdapseisutils/utils/obspyutils/ into vdapseisutils/obspy_ext/ with a clear layout, e.g.:
- obspy_ext/catalog/
- obspy_ext/inventory/
- obspy_ext/stream/
- obspy_ext/time.py (VUTCDateTime)

Update all in-repo imports. Keep vdapseisutils/utils/obspyutils/ as thin re-export shims (DeprecationWarning) for one release cycle if practical.

Run import smoke: from vdapseisutils.obspy_ext import VCatalog, VInventory, VStream, VUTCDateTime, VClient

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T10 — Fix broken package `__init__` files and root exports

**Scope:** Fix `utils/obspyutils/__init__.py` circular/self-import; align `vdapseisutils/__init__.py` exports with real class names (`Stream` vs `VStream` etc.). Prefer exporting extended types from `obspy_ext` at the top level **or** document that users should `from vdapseisutils.obspy_ext import ...`.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T10:
1. Fix vdapseisutils/utils/obspyutils/__init__.py — remove self-import, define explicit __all__ or minimal exports pointing to obspy_ext.
2. Fix vdapseisutils/__init__.py so it does not import non-existent names from obspy.py; align with obspy_ext public API.
3. Remove or populate empty stub files (stream.py, stream/__init__.py, etc.) per earlier audit.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T11 — Deprecation shims: `core.datasource` and old paths

**Scope:** `core/datasource/__init__.py` re-exports `VClient`/`DataSource`, `VStreamID`, `_fetch` helpers if needed. Document migration path in module docstrings. Fix any known broken imports (`catalogutils` / `hypoinverse`, `convertNSLCstr`).

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T11:
1. Populate vdapseisutils/core/datasource/__init__.py to re-export from obspy_ext (VClient/DataSource, VStreamID, etc.) with DeprecationWarning.
2. Optionally deprecate old DataSource class module in favor of VClient alias—merge behavior should already be in VClient from T4; old DataSource.py can become a shim or be removed if unused.
3. Grep for known broken patterns from the audit (catalogutils hypoinverse import, convertNSLCstr) and fix if still present.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## T12 — Tests and public API smoke

**Scope:** Add or extend tests under `tests/` for: `VStreamID` parsing; `VClient` URI + heuristic construction; one `get_waveforms` path through `_fetch`; `read`/`read_events` wrappers; `VCatalog[0]` returns same object as `catalog.events[0]` and `isinstance(..., VEvent)`.

**Copy-paste for sub-window:**

```text
Repo: vdapseisutils. Read docs/obspy_ext_subtasks.md.

Task T12:
1. Add pytest tests (or extend existing) for obspy_ext: VStreamID, VClient/DataSource construction (at least one URI and one heuristic), read/read_events wrappers return correct types.
2. Add a test that for a VCatalog built from a small Catalog, catalog.events[i] is VEvent and catalog[i] is catalog.events[i] (identity or at least same object—use `is` per T7 design).
3. Run pytest and fix failures.

Commit. Trailer: Co-authored-by: Pasquale
```

---

## Optional later cleanup (not blocking)

- Remove `utils/obspyutils/` entirely after deprecation period.
- Rename any remaining `StreamV` / `CatalogV` in `obspy.py` to aliases or delete.
- Unify `catalogutils` functions with `VCatalog` methods / single backend for `to_txyzm` vs `catalog2txyzm`.
- Documentation site or README section “Migrating to obspy_ext”.

---

## One-line tracker (for your notes)

`T0 branch → T1 skeleton → T2 VStreamID → T3 _fetch → T4 VClient+URI+DataSource alias → T5 VStream returns → T6 read* → T7 VEvent canonical → T8 stream dedup → T9 big move → T10 inits → T11 shims → T12 tests`
