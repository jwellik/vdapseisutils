# Gallery and tutorials — execution plan (coordinator)

Use this file as the **single source of truth** for a multi-window effort: update checkboxes and notes here; spawn subtasks in other windows by pointing to a **phase + task id** below.

## Goals (unchanged)

- One **canonical** layout for examples, gallery, and data references.
- **Real volcano** examples where it matters; **small** data in-repo; **large** data fetched or optional.
- **Runnable outputs** (e.g. PNG under a predictable path) suitable for **smoke tests** and optional **visual regression**.
- **Easy to find** from README / `docs/maps_volcano_api_v1.md` and short “gallery index” links.

---

## Plain language: where big files live

| Approach | What it is | When to use |
|----------|------------|-------------|
| **Normal git** | File committed like code. | Only if small (rule of thumb: **&lt; ~1 MB per file**, total gallery data **&lt; ~10–20 MB** unless you accept slower clones). |
| **Git LFS** | Git stores a **pointer**; the real blob lives on an LFS server. `git clone` still pulls LFS objects unless you skip them. | Medium files (e.g. **1–100 MB**), team already uses LFS, you want **versioned** data next to commits. **Cost/hosting** and **CI checkout** need LFS enabled. |
| **GitHub Release asset** | You upload a `.zip` or `.tar` with data to a **tagged release**. Repo stays light; users get a URL like `.../releases/download/vX.Y/datasets.tar.gz`. | **Larger** bundles, infrequent updates, simple hosting without another vendor. |
| **Zenodo** | Archive with a **DOI** (citable); upload files once per “version”. | **Publication-grade** citation, stable long-term URL, good when data is **large** or you want it **outside** GitHub. |
| **Pooch** | Small **Python library**: your code says “download `file.mseed` from this URL if missing, verify **SHA256**, cache in `~/.cache/...`”. | **Best default** for “large or optional” data: repo stays small, **reproducible** (hash), **offline** after first fetch. URL can point to **Release**, **Zenodo**, S3, etc. |

**Practical split for this project**

- **In-repo (no LFS):** trimmed miniseed (minutes, one or two channels), tiny QuakeML, tiny grids if any — whatever keeps the repo pleasant.
- **Pooch + URL:** anything bigger or optional; the **URL** is either a **GitHub Release** attachment or **Zenodo** file link. You do **not** have to use both — pick one hosting style and stick to it.
- **LFS:** optional shortcut if you prefer blobs **in the same repo** and already pay for LFS bandwidth; not required if you use pooch + release.

---

## Target layout (create this structure first)

Repo root (names can be tweaked, but keep the **separation of roles**):

```text
examples/                    # Fast, CI-friendly scripts (keep; extend)
gallery/
  README.md                  # Index: what to run, what needs network/data
  notebooks/                 # Narrative tutorials (.ipynb); paths via datasets API only
  scripts/                   # Canonical “gallery” scripts mirroring notebooks where useful
  _build/                    # Generated PNGs (gitignored) or baselines (git-tracked if using pytest-mpl)
  data/
    README.md                # Table: volcano, file, size, source, license/citation
    fixtures/                # Small committed binaries (miniseed, quakeml, …)
vdapseisutils/
  datasets/                  # NEW: loaders, registry, pooch registry (optional __init__ exports)
```

**Deprecate / merge**

- **`local_examples/`:** merge unique scripts into `examples/` or `gallery/scripts/`, then remove duplicates.
- **`~/PROJECTS/GALLERY`:** cherry-pick into `gallery/`; do not reference absolute paths from the package.
- **`gallery/output/`** (ad hoc): migrate to `gallery/data/fixtures/` or pooch-backed cache; notebooks use loaders only.

**Docs / README**

- Fix stale **`from vdapseisutils.gallery import ...`** unless you add a real `vdapseisutils.gallery` package; prefer links to `gallery/README.md` and commands.

---

## Scripts vs notebooks (canonical set)

| Artifact | Role |
|----------|------|
| **`gallery/scripts/*.py`** | **Source of truth** for CI: deterministic `Agg`, fixed `figsize`/`dpi`, write to `gallery/_build/...`. Easy to pytest. |
| **`gallery/notebooks/*.ipynb`** | **Teaching**: prose, intermediate cells, plots. Should call the **same helpers** as scripts (shared functions in `vdapseisutils.datasets` or thin `gallery/scripts/_common.py`) so notebooks do not drift. |

**Recommendation:** maintain **both** for the same stories where it matters; for thin demos, **script-only** is enough.

---

## Coverage matrix (canonical gallery — check off as done)

Track each cell: `script` / `notebook` / `both` / `n/a`.

| Area | Topic | Notes |
|------|--------|--------|
| **Maps** | Minimal `Map` + scatter | Extend `examples/map_minimal.py` pattern or duplicate under `gallery/scripts` with real fixture coords. |
| **Maps** | `VolcanoFigure` layout, cross-section | Align with `examples/volcano_figure_layout.py`, `cross_section_standalone.py`. |
| **Maps** | Hillshade / terrain | Real data via PyGMT cache; document network on first run. |
| **SwarmMPL** | Helicorder (dayplot-style + extras) | Real miniseed fixture + optional longer window via pooch. |
| **SwarmMPL** | Clipboard / waveform panel | Start from `examples/swarm_clipboard_minimal.py`; add fixture variant. |
| **SwarmMPL** | Time axis / v3 stack (if public API) | One focused script. |
| **obspy_ext** | Smallest public surface: one import path, one typical call | Pick 2–3 functions actually meant for users (see `docs/obspy_ext_subtasks.md`); avoid sandbox. |

---

## Phases and subtasks (for parallel windows)

### Phase 0 — Layout scaffolding

- [ ] **0.1** Create directories: `gallery/notebooks/`, `gallery/scripts/`, `gallery/data/fixtures/`, `vdapseisutils/datasets/`, `gallery/_build/` (gitignore `_build/` unless using checked-in baselines).
- [ ] **0.2** Add `gallery/data/README.md` with column template (filename, volcano, size, origin, citation).
- [ ] **0.3** Add `gallery/README.md` index (commands, env vars, “first run downloads”).
- [ ] **0.4** Add minimal `vdapseisutils/datasets/` module: `fixtures_dir()`, `resolve_path(name)`, placeholder for pooch registry.
- [ ] **0.5** Move existing notebooks from `gallery/*.ipynb` → `gallery/notebooks/` (update any internal links); remove `.ipynb_checkpoints` from tracking if present.

### Phase 1 — Small committed datasets

- [ ] **1.1** Choose 1–2 **trimmed** miniseed windows (e.g. Gareloi + one other) and one small QuakeML if needed; document in `gallery/data/README.md`.
- [ ] **1.2** Copy files into `gallery/data/fixtures/` (or `tests/data/` if you prefer tests-only — but gallery should read via `vdapseisutils.datasets`).
- [ ] **1.3** Implement `load_helicorder_demo_stream()` (name TBD) that reads from fixtures only.

### Phase 2 — Canonical scripts (maps)

- [ ] **2.1** `gallery/scripts/maps_minimal_fixture.py` — Map + scatter using fixture-related lon/lat or committed catalog snippet.
- [ ] **2.2** `gallery/scripts/maps_volcano_figure_fixture.py` — VolcanoFigure / cross-section path using same data policy as examples.
- [ ] **2.3** Optional: `gallery/scripts/maps_hillshade_*.py` — document “network required on first run”; no large data in git.

### Phase 3 — Canonical scripts (swarmmpl)

- [ ] **3.1** `gallery/scripts/swarm_helicorder_fixture.py` — Helicorder + output PNG.
- [ ] **3.2** `gallery/scripts/swarm_clipboard_fixture.py` — Clipboard (or equivalent) on fixture stream.
- [ ] **3.3** Update `gallery/notebooks/Helicorder_tutorial.ipynb` to use **only** dataset loaders (no `./output/...` relative paths unless under fixtures).

### Phase 4 — obspy_ext

- [ ] **4.1** From `docs/obspy_ext_subtasks.md`, pick **user-facing** entry points and add **one script per theme** (max 3 small scripts).
- [ ] **4.2** Prefer **no** large data; use ObsPy built-ins or tiny fixtures.

### Phase 5 — Large / optional data (pooch)

- [ ] **5.1** Decide host: **GitHub Release** vs **Zenodo** (one is enough).
- [ ] **5.2** Add optional dependency group in `pyproject.toml` (e.g. `gallery` extra with `pooch`).
- [ ] **5.3** Register one optional file in `vdapseisutils/datasets/` with URL + SHA256; `load_*_extended()` uses pooch, falls back with clear error if offline and not cached.
- [ ] **5.4** Document in `gallery/README.md`.

### Phase 6 — CI and docs

- [ ] **6.1** Add a **single** pytest or nox job: run all `gallery/scripts/*.py` with timeout (skip hillshade if `--no-network`).
- [ ] **6.2** Optional: `pytest-mpl` baselines for 1–2 figures.
- [ ] **6.3** Update root `README.md` and `docs/maps_volcano_api_v1.md` with the new paths; remove or fix `vdapseisutils.gallery` import instructions.

---

## Coordinator conventions

- When a sub-window finishes a task, it **edits this file**: check the box, add one line under the task (PR branch, commit hash, or “blocked: …”).
- Naming: prefer `gallery/scripts/<area>_<topic>_fixture.py` for anything using committed data.
- Never commit **machine-specific** paths; only paths relative to repo or pooch cache.

---

## Open decisions (fill in as you go)

- Maximum **in-repo** gallery data size (MB): ___  
- Pooch base URL (Release or Zenodo): ___  
- Whether to use **Git LFS** at all (yes/no): ___
