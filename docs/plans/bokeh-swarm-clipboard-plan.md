# Plan: Bokeh Swarm-style Clipboard (and later Helicorder)

Branch: **`feature/bokeh-swarm-clipboard`**  
Focus for this document: **Clipboard** first; **Helicorder** is scoped as a follow-on on the same initiative.

---

## Goals

1. Provide an **interactive (Bokeh)** alternative to the matplotlib Swarm-style clipboard, suitable for notebooks and light web embedding, without replacing the matplotlib implementations.
2. Reuse **backend-neutral** waveform and spectrogram preparation (`prepare_waveform_series`, `compute_spectrogram` from `vdapseisutils.compute.waveforms`) so numerical results stay aligned with existing MPL code paths.
3. Mirror **meaningful API surface** from the reference matplotlib types (constructor inputs, modes, sync behavior, common overlays), following the precedent set by Bokeh **`Map` / `CrossSection`** under `vdapseisutils.core.maps.bokeh` (see `docs/plans/bokeh-maps-plan.md`).
4. Ship **documentation**: gallery notebook **`gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`** (mirrors **`Clipboard_Tutorial_A.ipynb`** Examples 1–3; see **Gallery notebook** below) and smoke tests once primitives exist.
5. First **usable** milestone must support **`save`/standalone HTML** (Bokeh `output_file` + `save`, or equivalent embedding pattern used in maps gallery notebooks—not notebooks-only).

Non-goals for Clipboard v1 (unless explicitly promoted): full Swarm desktop parity (pick editing, clipboard buffers), server-scale streaming, pixel-perfect matplotlib styling, and restoring the stale **`vdapseisutils.swarmbk`** reference in the repo-root `__init__.py` (that import path does not exist today).

---

## Decisions (2026-05-05)

| Topic | Decision |
|-------|----------|
| API parity target | **`SwarmClipboard`** matplotlib type first; legacy **`ClipboardClass`** compatibility deferred unless explicitly requested. |
| Downsampling | **Deferred for meow:** no committed algorithm, caps, or user-facing kwargs until after an initial Clipboard MVP; notebook/HTML work proceeds without a downsampling design review. |
| Delivery | Primary iteration in **notebooks**, but **HTML export / `save`** is required on the **first usable drop** (same seriousness as maps Bokeh work). |
| Package layout | **`vdapseisutils.core.swarmmpl.bokeh`**, mirroring **`vdapseisutils.core.maps.bokeh`**; optional **`[bokeh]`** extra if the core package stays matplotlib-first for minimal installs. |

---

## Audit: what “SwarmMPL Clipboard” is today

There are **two** clipboard stacks in `vdapseisutils.core.swarmmpl.clipboard`; callers should be assumed to use either depending on age and docs.

### 1. Legacy: `Clipboard` factory + `ClipboardClass` (`matplotlib.figure.Figure` subclass)

**Role:** One **matplotlib subfigure per trace**. Each subfigure is filled by `plot_trace()`, which stacks waveform and/or spectrogram axes via a gridspec (`mode`: `"wg"`, `"w"`, `"g"`).

**Constructor / state (`ClipboardClass.__init__`):**

- **`st`**: `ObsPy.Stream` (copied); **traces are not merged** (explicit design vs `Stream.plot()`).
- **`mode`**: `"wg"` | `"w"` | `"g"` (waveform + spectrogram, waveform only, spectrogram only).
- **`tick_type`**: `"datetime"` | `"relative"` (seconds-style axis when not datetime).
- **`sync_waves`**: if `True`, all panels share the union of absolute trace times; if `False`, each trace uses its own start/end (with layout implications for tick clutter).
- **`force_length`**: pad shorter traces so all panels share the same nominal duration on the x-axis.
- **Figure sizing**: default height scales with trace count.

**Primary workflow:** `fig = Clipboard(st, ...)` then **`fig.plot()`**, which calls **`_set_axes()`** to apply shared x-limits, datetime formatters (`mdates.AutoDateLocator` / `ConciseDateFormatter`), relative labeling behavior, and selective removal of middle x-axis labels when `sync_waves` is `True`.

**Notable methods:**

| Method | Behavior |
|--------|----------|
| `set_wave` / `set_spectrogram` | Merge dicts into default plotting kwargs. |
| `set_tlim` | Apply same x-limits to all axes in all subfigures. |
| `set_alim` | Y-limits on waveform axes (`mode` must not be spectrogram-only). |
| `set_flim` | Spectrogram frequency y-limits. |
| `set_prange` | Stub (color/power range for spectrogram not implemented). |
| `axvline` | Vertical markers across panels; uses **`t2axiscoords`** for datetime vs relative interpretation. |
| `scroll_traces` | Shift x-limits (and internal `time_lim`) per trace index / seconds lists—**interactive/navigation-oriented**. |
| `remove_labels` | Strip ticks and axis text (layout cleanup). |
| `plot_peak_value` | **Raster overlay** (`imshow`) behind waveform axes; windowed max/mean aggregation; optional colorbar on figure. |

**Helpers used by panels:**

- **`plot_wave`**: `prepare_waveform_series`, datetime or relative x-vector, right y-axis ticks, scientific y formatting.
- **`plot_spectrogram`**: optional resample, **`compute_spectrogram`**, `pcolormesh`, log freq optional.

### 2. Modern panel stack: `SwarmClipboard` + `Panel` + `TimeAxes`

**Role:** Preferred **v3** layout in README / `examples/swarm_clipboard_minimal.py`: one matplotlib figure, multiple **`Panel`** regions, each holding one or more **`TimeAxes`** wrappers.

**Constructor highlights:**

- **`data`**: `None`, `Stream`, or iterable of traces → builds panels eagerly when data present.
- **`sync_waves`**: global time alignment across panels (distinct name from legacy but analogous concept).
- **`tick_type`**: documented as `"absolute"` / relative variants—**string vocabulary overlaps legacy but is not identical** (`"datetime"` vs `"absolute"`).
- **`mode`**: `"w"` | `"g"` | `"wg"` per trace panels.
- Layout uses **`title_space`**, **`panel_spacing`**, **`panel_height`**.

**Richer API than legacy:**

| Method | Behavior |
|--------|----------|
| `plot_trace` / `plot_horizontals` | Overlay additional traces on matched panels (metadata/station targeting). |
| `axvline` | Metadata-aware targeting (`stations`, `networks`, `ids`). |
| `plot_catalog` | Origins and picks via **`TimeAxes.axvline`** on matched panels. |
| `plot_peak_value` | Same conceptual overlay as legacy, applied to first waveform `TimeAxes` per panel. |
| `set_xlim`, `set_wlim`, `set_slim`, `set_tick_type` | Panel-level coordination helpers. |

**Dependencies:** `TimeAxes` (`timeaxes.py`) implements **`plot_waveform`**, **`plot_spectrogram`**, tick formatting, **`axvline`**, etc., still on matplotlib.

### 3. Ancillary: `TimeSeries` axes subclass

Custom `matplotlib.axes.Axes` for generic time-series + **`plot_catalog`** using **`prep_catalog_data_mpl`**. Useful for maps/time-series cross-links but **not** the main clipboard layout; note for future if Bokeh clipboard gains catalog overlays on a dedicated axis.

### 4. Public entry points

- **`vdapseisutils.plot.swarm`**: exports `Clipboard`, `ClipboardClass`, `SwarmClipboard`, etc.
- **`register_pyplot()`** (`vdapseisutils.plot.mpl`): **`plt.clipboard`** / **`plt.swarm`** → legacy factory.

---

## Design directions for Bokeh Clipboard

### Recommended parity target (Clipboard)

**Phase A (locked):** Implement Bokeh figures that match **`SwarmClipboard`** semantics (multi-panel, metadata targeting, catalog hooks)—canonical path per stakeholder decision.

**Phase B (optional / later):** Add a thin legacy-compat layer mapping **`ClipboardClass`** kwargs (`tick_type="datetime"`, `sync_waves`, `force_length`) onto the same internal layout engine if notebook users ask for drop-in parity with **`gallery/SwarmMPL/Clipboard_Tutorial_A.ipynb`**.

### Layout model

- **One Bokeh `figure`** or **`gridplot`** of linked child figures: each trace maps to a vertical stack (waveform row + spectrogram row) analogous to MPL height ratios `[1, 3]` for `"wg"`.
- **Linked ranges:** When `sync_waves` is `True`, share **`DataRange1d`** / `Range1d` across panels for x; when `False`, independent ranges per panel.
- **Datetime x:** Bokeh expects numeric datetime in ms; convert from Python `datetime` consistently with `compute_spectrogram` time bins and waveform samples.

### Glyphs

| Content | Bokeh approach |
|---------|----------------|
| Waveforms | `line` or `multi_line` from ColumnDataSource (downsampling policy **deferred**; may plot full series at MVP). |
| Spectrograms | `image` or `ColorMapper`-backed glyph; balance fidelity vs payload size (same tension as MPL `pcolormesh`). |
| Peak raster | `image_rgba` or `image` with palette; match extent to waveform x/y range. |
| Catalog / picks | `vertical_span` / `span` or infinite-height `line` at converted times; reuse catalog prep helpers where possible. |

### Interactivity (incremental)

1. **Pan / zoom / reset** with **linked x** when synced.  
2. **Hover tooltips** for time (formatted) and amplitude on waveforms; frequency/time/power on spectrograms where feasible.  
3. **`scroll_traces`-like** behavior: optional **callbacks** or documented pattern using `Range1d` updates (may stay notebook-driven initially rather than built-in buttons).

### Packaging / imports

Follow maps precedent:

- **Locked:** **`vdapseisutils.core.swarmmpl.bokeh`** exporting a Bokeh clipboard type (e.g. **`SwarmClipboardBk`** / **`BokehSwarmClipboard`**) with docstring cross-links to **`SwarmClipboard`**.
- Optional **`[bokeh]` extra** in `pyproject.toml` if Bokeh remains optional for minimal installs (match how maps/tests gate Bokeh).

### HTML export (first usable drop)

- Expose a small, documented API surface for **`save(path)`** and/or **`to_html()`** on the figure/grid wrapper, delegating to Bokeh **`save`** with **`CDN`** resources unless offline bundles are explicitly requested later.
- Gallery notebook cells should demonstrate both **`show(notebook)`** (or inline) and **writing a `.html`** next to maps notebooks.

---

## Gallery notebook: `gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`

Add a notebook **alongside** **`gallery/SwarmMPL/Clipboard_Tutorial_A.ipynb`**, structured for the same three worked examples, but using the **Bokeh** clipboard type from **`vdapseisutils.core.swarmmpl.bokeh`** (SwarmClipboard semantics). Opening markdown should note that Tutorial A uses **legacy matplotlib `Clipboard`**, while this notebook targets **`SwarmClipboard`-style** Bokeh API (method names may differ from Tutorial A, e.g. **`set_wlim` / `set_slim`** vs **`set_alim` / `set_flim`**).

| Section | Tutorial A intent | Bokeh notebook deliverable |
|--------|-------------------|---------------------------|
| **Example 1** — Clipboard figure and axes | Read Gareloi miniSEED, slice to a 10‑minute window, default waveform + spectrogram, title | Same data path and slice; build Bokeh multi-panel figure; show inline; include **`save(...)`** to HTML |
| **Example 2** — “Pensive-like” plot | Bandpass filter; **`set_spectrogram`** / **`set_wave`**; **`plot`**; multiple **`axvline`** times; **`set_alim`** / **`set_flim`**; optional small **`remove_labels`‑style** thumbnail | Equivalent kwargs on Bokeh type; vertical markers at same UTC strings; waveform and spectrogram y‑limits; compact layout variant if supported |
| **Example 3** — Relative time, unsynced traces | Augustine FI miniSEED; **`mode="w"`**, **`sync_waves=False`**, **`tick_type="relative"`**; **`scroll_traces`** | Independent x‑ranges per panel where applicable; relative time formatting; time‑shift / pan semantics via **`scroll_traces`‑equivalent** helper or documented **`Range1d`** update once implemented |

Reuse the **same example data files** as Tutorial A where paths exist under **`vdapseisutils/data/waveforms/`**, or document a portable relative path pattern consistent with other gallery notebooks.

---

## Phased work checklist

### Phase 0 — Scaffold

- [x] ~~Choose module path~~ — **`vdapseisutils.core.swarmmpl.bokeh`** with **`SwarmClipboardBk`** (does not shadow MPL **`Clipboard`**).
- [x] Extend **`examples/swarm_clipboard_minimal.py`** with API sketch + optional **`--save-bokeh-html PATH`**.
- [x] Dependency check: **`bokeh`** remains a core **`pyproject.toml`** dependency (same stack as maps notebooks); no extra pin required for Phase 0.
- [x] **`SwarmClipboardBk.save()`** writes standalone HTML via Bokeh **`save`** + **`CDN`** defaults.

### Phase 1 — Waveform-only multi-panel

- [x] Build N-panel vertical layout for `mode="w"` (column of Bokeh figures).
- [x] Implement `sync_waves` True/False range linking (shared **`Range1d`** vs independent).
- [x] Map **`tick_type`** behavior: **`absolute`** / **`datetime`** / **`time`** → datetime axis; **`relative`** → linear seconds (aligned to global **`t0`** when **`sync_waves`**).
- [x] Smoke tests: **`tests/test_swarm_clipboard_bokeh.py`** (including mismatched trace start times).

### Phase 2 — Spectrograms

- [ ] Add `mode="g"` and `"wg"` using shared `compute_spectrogram`.
- [ ] Color bar / palette defaults consistent with MPL (`inferno_u`, db scale).

### Phase 3 — Overlays and navigation

- [ ] `axvline` equivalent (global and metadata-targeted).
- [ ] `plot_peak_value` raster behind waveforms.
- [ ] `plot_trace` / `plot_horizontals` overlays (at least station-matched path).
- [ ] `plot_catalog` origins/picks (reuse ObsPy structures).

### Phase 4 — Docs + parity hardening

- [ ] Add **`gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`** with **Example 1**, **Example 2**, and **Example 3** aligned to **`Clipboard_Tutorial_A.ipynb`** (see **Gallery notebook** section above).
- [ ] `pytest` smoke tests (optional image baseline deferred).
- [ ] README section: “Bokeh clipboard” under Swarm plotting.

### Phase 5 — Helicorder (separate milestone)

- [ ] Audit **`vdapseisutils.core.swarmmpl.heli`** (and pyplot `Helicorder`) the same way.
- [ ] Bokeh day-long scrolling strips, optional channel stacking—reuse waveform prep and datetime axis patterns from Clipboard work.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Large notebook payloads (spectrogram matrices) | Document payload/runtime caveats for meow; revisit when downsampling is addressed. |
| Tick-type vocabulary drift (`absolute` vs `datetime`) | Single internal enum/normalizer shared by MPL wrappers and Bokeh. |
| Feature duplication vs MPL | Keep thin Bokeh facade; push math to `compute.waveforms`. |

---

## Follow-up discussion

1. **Downsampling (deferred for meow):** Re-open when MVP clipboard + **`Clipboard_tutorial_bokeh`** notebook are stable—algorithm, caps, interaction with **`save`**, and spectrogram vs waveform policies.

2. **Legacy Clipboard tutorials:** If users demand **`ClipboardClass`** keyword parity without migrating to **`SwarmClipboard`**, scope Phase B compat layer effort.

3. **Helicorder coupling:** Prefer **`vdapseisutils.core.swarmmpl.bokeh`** shared utilities (**datetime conversion**, toolbars, grid layout helpers) when the Helicorder milestone starts—no separate package fork.
