# Plotly interactive views: Map and CrossSection

Plan for optional Plotly-backed interactive versions of **Map** and **CrossSection**, implemented in order: **CrossSection → Map**.

**Out of scope (this branch):** Plotly ports of **SwarmMPL** waveform/spectrogram UIs, including multi-trace **Clipboard** / **SwarmClipboard**. Those stay matplotlib-first; this document used to list Clipboard phases—they are intentionally dropped.

---

## Branch policy (required)

All work for this effort—dependencies, new modules, examples, tests, and doc updates listed in this plan—happens **only** on branch:

**`feature/plotly-interactive-views`**

- **Do not** implement Plotly ports on `main` (or other topic branches) unless you are merging this branch.
- Start every Cursor session with `git checkout feature/plotly-interactive-views` (after fetching/pulling).
- Prompts below assume you are **already on** this branch; do not create a second Plotly branch unless the team renames this one deliberately.

---

## Overall approach

- Stay on **`feature/plotly-interactive-views`** for the full lifecycle of the Plotly work; merge to `main` only when ready.
- Add **`plotly`** as an optional dependency group if packaging supports it (e.g. `[plotly]` extras); keep Matplotlib/Cartopy paths **unchanged** on `main` until parity is acceptable.
- Prefer **new modules** (e.g. `vdapseisutils/.../plotly/` or `experimental/`) that **reuse existing data prep** (`prep_catalog_data_mpl`-style inputs, `elev_profile`, geoutils) and only replace the **rendering** layer.
- For each component, ship a **small example script** and a short **parity checklist** (what matches Swarm/VolcanoFigure workflows vs what is deferred).

---

## CrossSection — three parts

| Part | Focus |
|------|--------|
| **1** | Deps, module skeleton, API survey, reuse geometry/profile data without Matplotlib axes (on `feature/plotly-interactive-views`). |
| **2** | Core Plotly figure: profile line, heatmap/scatter for events, depth/distance axes, basic styling aligned with `CROSSSECTION_DEFAULTS`. |
| **3** | Legend/magnitude styling, HTML export, example + parity notes vs matplotlib `CrossSection`. |

### Prompt — CrossSection Part 1

```
You are working in the vdapseisutils repo; you must be on branch `feature/plotly-interactive-views` (all Plotly work lives on this branch only).

Goal: CrossSection Plotly port — Part 1 of 3 — foundation only. Do not replicate full matplotlib CrossSection yet.

Tasks:
1. Add plotly as an optional dependency if the project uses extras; otherwise document in pyproject/requirements for a [plotly] extra.
2. Create a new package/module path for Plotly cross-sections (e.g. vdapseisutils.core.maps.plotly.cross_section or similar — match repo conventions).
3. Read vdapseisutils/core/maps/cross_section.py and list the public methods/properties downstream code relies on (VolcanoFigure, examples/cross_section_standalone.py).
4. Implement a thin data layer: functions or a small class that computes the same inputs CrossSection needs for plotting (points, profile from elev_profile.TopographicProfile, depth_extent, filtered catalog/inventory prep) without creating matplotlib Figure/Axes. Reuse existing utilities (prep_catalog_data_mpl, geoutils) where possible.
5. Add minimal tests or a smoke script that loads example data and asserts the data layer returns sane arrays/ranges.

Deliverables: new module(s), dependency wiring, short README comment or module docstring explaining scope. No full interactive figure yet unless trivial placeholder go.Figure() to prove imports work.

Constraints: do not break existing matplotlib CrossSection; keep changes isolated.
```

### Prompt — CrossSection Part 2

```
You are working in vdapseisutils on branch `feature/plotly-interactive-views` — CrossSection Plotly Part 2 of 3.

Prerequisites: Part 1 data layer exists.

Goal: Build the core interactive Plotly figure for a vertical cross-section.

Tasks:
1. Implement a function or class (e.g. CrossSectionPlotly) that takes the Part 1 data structures + same high-level kwargs as matplotlib CrossSection where reasonable (label, depth_extent, points/origin/azimuth/radius_km, etc.).
2. Use plotly.graph_objects: Heatmap or Scatter for seismic/catalog visualization, Scatter for topo profile; correct axis titles (Depth km vs distance along profile).
3. Match key visual defaults from vdapseisutils/core/maps/defaults.py (CROSSSECTION_DEFAULTS, TICK_DEFAULTS where applicable) as Plotly layout/update_traces.
4. Support saving to HTML and optional show() in notebooks.

Deliverables: working figure from examples/cross_section_standalone.py-style inputs (adapt imports to Plotly entry point); document any intentional API differences in the class docstring.

Constraints: defer MagLegend pixel-perfect parity to Part 3; no VolcanoFigure wiring yet unless trivial.
```

### Prompt — CrossSection Part 3

```
You are working in vdapseisutils on branch `feature/plotly-interactive-views` — CrossSection Plotly Part 3 of 3 (final).

Goal: Polish and parity with matplotlib CrossSection for typical VolcanoFigure workflows.

Tasks:
1. Add magnitude legend or colorbar behavior comparable to MagLegend usage in cross_section.py (Plotly colorbar, discrete vs continuous as appropriate).
2. Add example script under examples/ mirroring cross_section_standalone.py but for Plotly (e.g. examples/cross_section_plotly_standalone.py).
3. Write a short PARITY.md or section in module docstring: what matches matplotlib CrossSection, what is unsupported (e.g. custom spine bounds), workarounds.
4. Run existing tests; fix any regressions; add a small test that instantiates CrossSectionPlotly with minimal fake data if the repo has a test pattern for maps.

Deliverables: examples + docs; optional integration hook comment for VolcanoFigure (do not fully refactor VolcanoFigure unless explicitly requested).

Constraints: keep matplotlib CrossSection as default public API until maintainers decide to expose Plotly in __init__.py.
```

---

## Map — three parts

| Part | Focus |
|------|--------|
| **1** | Basemap strategy: `scattergeo` vs Mapbox; deps/tokens; minimal interactive map + extent + scatter points. |
| **2** | Catalog/inventory overlays, styling parity with map defaults; optional raster/tile strategy for terrain. |
| **3** | Hillshade/raster overlay or documented simplification; titles, scale/north analogs; example + VolcanoFigure integration notes. |

### Prompt — Map Part 1

```
You are working in vdapseisutils on branch `feature/plotly-interactive-views` — Map Plotly Part 1 of 3.

Goal: Choose and implement the minimal interactive basemap for geographic plots without porting all of map.py yet.

Tasks:
1. Read vdapseisutils/core/maps/map.py (class Map, key methods) and defaults used for volcano/catalog plots.
2. Decide documented approach: plotly scattergeo with natural earth / open-street style vs mapbox (note: Mapbox may need token). Implement one path first with a clear extension point for the other.
3. New module e.g. vdapseisutils/core/maps/plotly/map_plotly.py with function/class MapPlotly or plot_map_interactive that accepts lon/lat extent and plots empty or light base + optional single scatter test.
4. Document dependency and any API keys in module docstring.

Deliverables: minimal working HTML map from a tiny example; no Cartopy/PyGMT removal from existing Map.

Constraints: do not duplicate massive hillshade logic in Part 1; stub terrain as TODO.
```

### Prompt — Map Part 2

```
You are working in vdapseisutils on branch `feature/plotly-interactive-views` — Map Plotly Part 2 of 3.

Prerequisites: Part 1 basemap works.

Goal: Overlay seismic catalog and inventory-style data with styling approaching current Map behavior.

Tasks:
1. Reuse prep_catalog_data_mpl or shared prep functions from vdapseisutils/core/maps/utils.py so data paths match matplotlib Map.
2. Plot events as scatter (size by magnitude, color by depth or time) using Plotly traces; add hover templates (time, mag, depth, id).
3. Add legend/colorbar behavior consistent with HEATMAP_DEFAULTS / map styling where possible.
4. Extend example to load real or example catalog like existing map examples.

Deliverables: example script examples/map_plotly_minimal.py (or similar); document limits vs Cartopy tile quality.

Constraints: still defer full PyGMT hillshade to Part 3; if needed use simple topographic tile layer if chosen stack supports it.
```

### Prompt — Map Part 3

```
You are working in vdapseisutils on branch `feature/plotly-interactive-views` — Map Plotly Part 3 of 3 (final).

Goal: Terrain/hillshade parity strategy and integration story.

Tasks:
1. Either: (A) integrate precomputed raster as layout image / heatmap on mapbox/geo axes, or (B) document intentional simplification (e.g. satellite/OSM only) and keep PyGMT hillshade for matplotlib-only export.
2. If (A), add a small pipeline from existing PyGMT helpers in map.py to numpy RGB or elevation grid consumed by Plotly (performance and extent must be correct).
3. Add title/subtitle/layout parity with TITLE_DEFAULTS/SUBTITLE_DEFAULTS where applicable; approximate scale bar (annotation or UI note) — full geographic scale bar may be approximate in Plotly.
4. PARITY notes: compare to Map + VolcanoFigure expectations; optional stub for future VolcanoFigure dual-backend.

Deliverables: updated example, docstring parity section, no regression to matplotlib Map.

Constraints: large downloads should be cached consistently with existing patterns if code already exists.
```

---

## Summary

| Component        | Part 1                 | Part 2                 | Part 3                      |
|------------------|------------------------|------------------------|-----------------------------|
| **CrossSection** | Data layer + skeleton | Core `go.Figure`       | Legend + examples + parity  |
| **Map**          | Basemap strategy       | Catalog overlays       | Terrain/parity              |

There are **six** phased prompts above (three per component). Use **one prompt per Cursor chat** in order to keep each session focused.
