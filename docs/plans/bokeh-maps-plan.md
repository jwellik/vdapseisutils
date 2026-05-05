# Plan: Bokeh `Map` and `CrossSection` (feature/bokeh-maps)

This document drafts an implementation plan for Matplotlib/Cartopy-parity map types backed by **Bokeh**, with emphasis on **the same public API as** `vdapseisutils.core.maps.map.Map` and `cross_section.CrossSection`, and on **terrain basemaps** that match the historical REDPy recipe (already implemented for Cartopy in `map_tiles.add_arcgis_terrain`).

---

## Repository note: `REDPy-prerelease-2.0.0` will be removed

The **`REDPy-prerelease-2.0.0/`** tree under this repository is **not** a long-term dependency. It will eventually be **deleted**. New vdapseisutils code must:

- **Never import** from `REDPy-prerelease-2.0.0` or assume that directory exists.
- **Encode behavior** (terrain URLs, two-layer stack, zoom heuristics) **inside** `vdapseisutils`—today primarily **`core/maps/map_tiles.py`** and the forthcoming Bokeh map module—so runtime behavior continues to match what REDPy did, without carrying REDPy as vendored code.

The REDPy section below is a **design reference** (what that project did); the **canonical implementation target** for “terrain like REDPy” is **`add_arcgis_terrain`** plus shared helpers, extended to Bokeh tile layers.

---

## Goals

1. Introduce Bokeh-backed figure types whose **`__init__` signatures match** `Map` and `CrossSection` (same `origin`, `radial_extent_km`, `map_extent`, `points`, `depth_extent`, etc.), modulo figure construction (`fig` / `plt.figure` vs Bokeh `figure`, `dpi`/`figsize` → width/height or documented mapping).
2. **Mirror the Matplotlib `Map` / `CrossSection` method surface** so call sites can swap backends with minimal changes: same **names**, same **positional/keyword arguments**, and **equivalent data semantics** (catalog prep, projections, defaults). See **API parity** below.
3. Reuse **backend-neutral** logic: `radial_extent2map_extent`, `prep_catalog_data_mpl`, `choose_scale_bar_length`, `project2line`, `TopographicProfile`, defaults from `defaults.py`, zoom helpers from `map_tiles.py`.
4. **Defer** fine control of Bokeh graticule ticks, axis formatters, and degree labels (“for meow”); use plain Mercator ranges and default ticks until a later pass.
5. **`add_terrain()`** on the Bokeh map must follow the **same stacked-tile recipe** as `map_tiles.add_arcgis_terrain` / historical REDPy (Esri hillshade + Carto overlay), not a divergent default.

Non-goals for the first milestone: world Orthographic inset (`add_world_location_map`), PyGMT hillshade parity, pixel-perfect matplotlib kwargs passthrough.

---

## Current status (2026-05-05)

Implemented now:

- `vdapseisutils.core.maps.bokeh.Map` exists and is wired for terrain parity (Esri hillshade + Carto overlay).
- `Map` methods implemented: `plot`, `scatter`, `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak`, `plot_line`.
- Bokeh `Map` tile parity includes **`add_google_terrain`**, **`add_google_street`**, and **`add_google_satellite`** (XYZ URLs shared with Cartopy via `map_tiles`; cache/ssl kwargs match MPL swallow/no-op pattern).
- `Map` helpers implemented: `set_title`, `set_subtitle`, `set_catalog_subtitle`, `add_scalebar` (native `ScaleBar` + fallback), `add_world_location_map`.
- Aspect-locked zoom behavior implemented (`match_aspect=True`, `BoxZoomTool.match_aspect=True`).
- Hover behavior implemented for map scatter-family methods plus automatic hover defaults for inventory/catalog.
- New transect helpers implemented on both MPL and Bokeh maps: `plot_cross_section` and alias `plot_transect`.
- `vdapseisutils.core.maps.bokeh.CrossSection` exists with constructor parity, profile rendering, `plot`, `scatter`, `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak`, `plot_heatmap`, title/subtitle helpers, `set_titles`, and `set_catalog_subtitle`.
- **CrossSection scatter-family parity:** matplotlib-style kwargs (`color`/`c`, `size`/`s`, edge widths/colors, `**{'c','s'}` catalog passes, `cmap` for numeric/`time` coloring, inventory defaults) covered beyond basic smoke where tested.
- `CrossSection` adds fixed corner labels `A` and `A'` (screen-space); temporary debug mode exists via `debug_corner_labels=True` for label diagnostics.
- **`plot_heatmap`** on Bokeh `Map` and `CrossSection`: MPL-like calling modes and quad rendering; default **color bar** with `colorbar=False` escape hatch (aligned kwargs naming between backends).
- **MPL `Map.plot_heatmap` / `CrossSection.plot_heatmap`** attach a **Matplotlib colorbar by default** (`colorbar=False`, `colorbar_label` popped before `pcolormesh`) so the density legend story matches Bokeh.
- Heatmap bin sizing and **histogram bin edges** use shared helpers in **`core/maps/heatmap_utils.py`** (`heatmap_bin_step`, **`histogram_bin_edges`**) so `histogram2d` does not drop samples at the padded range max (MPL + Bokeh).
- MPL **`CrossSection.set_horiz_extent()`** supports **endpoint `points` mode** (uses geodesic **`length`** when **`radius`** is unset).
- Bokeh smoke tests include google tile stacking, catalog cmap/kwargs/inventory edge cases, and MPL stack smoke tests cover heatmap colorbar defaults.
- Gallery notebooks: `gallery/Mapping_tutorial_bokeh.ipynb` and `gallery/CrossSection_tutorial_bokeh.ipynb` (Spurr, St Helens, Crater Lake, Kilauea, Yellowstone).

Remaining high-priority gaps:

- **`add_hillshade`** on Bokeh maps / richer raster parity remains explicitly deferred ("for meow").

---

## API parity (must match MPL `Map` / `CrossSection`)

The Bokeh map type should expose the **same instance methods** as `vdapseisutils.core.maps.map.Map` wherever feasible, with arguments passed through in the same order and meaning; internally, translate to Bokeh glyphs/sources.

**`Map`-aligned methods (target parity):**

| Method | Notes |
|--------|--------|
| `info` | Same printed summary from `properties`. |
| `set_ticks`, `set_ticks_outside` | Stub or minimal behavior until graticule work; signature unchanged. |
| `add_hillshade` | PyGMT/raster path when implemented; signature unchanged. |
| `add_scalebar` | Same length logic; rendering via Bokeh geometry/labels. |
| `plot`, `scatter` | Same `(lat, lon, …)` convention as MPL `Map`. |
| `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak` | Same defaults kwargs (`PLOT_*_DEFAULTS`); drive `ColumnDataSource` + glyphs. |
| `plot_line` | Same endpoints and label behavior. |
| `plot_heatmap` | Same histogram/binning inputs; Bokeh renderer underneath. |
| `add_terrain`, `add_arcgis_terrain`, `add_google_*` | Same entry points as MPL map; terrain defaults per `add_arcgis_terrain` stack. |
| `add_world_location_map` | Later phase (complex); keep signature when implemented. |
| `set_title`, `set_subtitle`, `set_titles`, `set_catalog_subtitle` | Same kwargs pattern as MPL helpers. |

**`CrossSection`-aligned methods:** `plot`, `scatter`, `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak`, `plot_heatmap`, `set_depth_extent`, `set_horiz_extent`, `set_title`, `set_subtitle`, `set_titles`, `set_catalog_subtitle`, plus constructor-driven profile behavior.

**Naming:** Prefer a class name that imports cleanly (e.g. `BokehMap` / `BokehCrossSection`) while keeping **method names identical** to the MPL classes so documentation and muscle memory stay aligned; alternatively a submodule `from vdapseisutils.core.maps.bokeh_map import Map` if shadowing is acceptable in that namespace only.

---

## REDPy terrain audit (historical reference — `REDPy-prerelease-2.0.0`)

### Where it lives (reference only)

- **`REDPy-prerelease-2.0.0/redpy/outputs/mapping.py`**: `_get_tiles()`, `_set_up_map_image()`, Folium basemap helpers. Do **not** import this from shipping vdapseisutils code once the vendored tree is removed.

### How static Cartopy maps get terrain

1. **`_get_tiles()`** builds two **`cartopy.io.img_tiles.GoogleTiles`** instances with **custom `url` templates** and **disk cache** paths:
   - **Base (hillshade):** Esri *World Hillshade*  
     `https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}`
   - **Overlay (reference):** Carto *Positron no labels*  
     `https://tiles.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png`

2. **`_set_up_map_image()`** creates a **GeoAxes** with `projection=terrain.crs`, sets extent, then:
   - `ax.add_image(terrain, zoom)`
   - `ax.add_image(overlay, zoom, alpha=0.5)`  
   So the **visual** is **hillshade tiles** plus a **semi-transparent** light basemap for land/water context.

### How interactive Folium maps get basemaps

- **`_add_basemaps()`** registers the **same Esri World Hillshade** URL as the default `folium.TileLayer` (“World Shaded Relief”), plus optional imagery and OpenTopoMap layers.

### Takeaway for vdapseisutils Bokeh

REDPy does **not** use a bespoke raster pipeline for “terrain”; it uses **standard XYZ/WMTS-style URLs** and a **fixed two-layer stack** (hillshade + dimmed reference). Encode that behavior only in **`map_tiles`** + the Bokeh map class. Bokeh uses **two tile renderers** sharing the same Web Mercator extent and zoom, with the overlay at reduced alpha—matching **`add_arcgis_terrain`**, not a one-off URL list in the Bokeh module alone.

---

## How this repo already matches REDPy (Cartopy `Map.add_terrain`)

`vdapseisutils.core.maps.map_tiles.add_arcgis_terrain` implements the **same pair of URLs** and **`alpha=0.5` on the overlay**, with a comment crediting the REDPy/Alicia Hotovec Ellis approach. Zoom when `zoom='auto'` uses **`_calculate_auto_zoom_arcgis(radial_extent_km)`**.

**Incorporation strategy for Bokeh `add_terrain()`:**

1. **Do not invent a third terrain scheme** for the default path: call a shared internal helper, e.g. `_terrain_tile_urls()` or reuse constants next to `add_arcgis_terrain`, so Cartopy and Bokeh stay **one definition** of URL templates and attribution strings.
2. **Zoom:** Reuse `_calculate_auto_zoom_arcgis` (and the same `radial_extent_km` from `Map.properties` / Bokeh mirror) so auto-zoom matches current matplotlib behavior.
3. **Bokeh wiring:**
   - Build a Bokeh `figure` with **`x_axis_type="mercator"`** and **`y_axis_type="mercator"`** (or equivalent in the target Bokeh major version).
   - Attach **two** `TileRenderer` / `add_tile` layers (API detail depends on Bokeh 3.x patterns): bottom = Esri hillshade, top = Carto `light_nolabels` with **`alpha=0.5`** (or the same value as `map_tiles` for consistency).
4. **Tile URL placeholders:** Bokeh tile sources typically expect `{X}`, `{Y}`, `{Z}` or `{x}`, `{y}`, `{z}` per model docs—normalize templates in one place when binding to `XYZTileSource` / `WMTSTileSource`.
5. **Esri Y-order:** Cartopy’s tile class may hide TMS vs XYZ details. During implementation, verify whether the Esri layer needs a **TMS flipped-Y** flag for Bokeh; if tiles appear north-south mirrored, flip per Bokeh’s tile source options.
6. **Attribution:** Pass Esri/Carto **attribution** text analogous to Folium’s `attr=` in REDPy, for license compliance in HTML output.
7. **Optional later:** Expose `add_arcgis_terrain`-style kwargs (`cache`, `verbose`, `ssl_verify`) on Bokeh `add_terrain()`; initial version can keep defaults and match `Map.add_terrain()` surface.

---

## Architecture sketch

| Piece | Suggestion |
|--------|------------|
| **REDPy** | **No imports.** Behavior copied into vdapseisutils; vendored `REDPy-prerelease-2.0.0/` may be deleted independently. |
| **Package layout** | Package **`vdapseisutils.core.maps.bokeh`** (`bokeh/__init__.py`, `bokeh/map.py`). Import class as **`from vdapseisutils.core.maps.bokeh import Map`** so the public name stays **`Map`** without colliding with **`from vdapseisutils.core.maps import Map`** (matplotlib/Cartopy). Top-level **`import bokeh`** still resolves to the PyPI library. |
| **Classes** | **`Map`** inside the `bokeh` subpackage; **`CrossSection`** TBD (`bokeh/cross_section.py` or similar). **Methods** mirror MPL `Map` / `CrossSection` per **API parity**. |
| **Coordinates** | Lon/lat WGS84 → Web Mercator (m) for glyphs and tiles; reuse `map_extent` + **pyproj** / geoutils. |
| **Figure handle** | `self.figure` = Bokeh `Figure`; optional `show()`, `save()` helpers. |
| **Tiles** | Default **`add_terrain()`** = shared URL/zoom with **`add_arcgis_terrain`** (Esri + Carto). **`add_google_*`** implemented on Bokeh `Map` using shared XYZ helpers in **`map_tiles`**. |

---

## Phased work

### Phase 1 — `BokehCrossSection` (linear axes)

- Mirror `CrossSection.__init__` geometry (`points` / `origin` + `azimuth` + `radius_km`), `TopographicProfile`, spine/profile styling approximated with Bokeh lines and axis limits. **(done for current parity target)**
- Implement `plot`, `scatter`, `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak`, `plot_heatmap` using Bokeh glyphs; reuse `prep_catalog_data_mpl` and `project2line`. **(done for smoke-level parity)**
- Title/subtitle/catalog subtitle aligned with existing methods (simpler than map). **(done, including `set_titles` and `set_catalog_subtitle`)**
- **Out of scope for meow:** Matplotlib `path_effects` on A/A′ labels; use solid background on labels or accept softer styling.

### Phase 2 — `Map` (Bokeh subpackage) core

- Constructor parity with `Map`: extent, `properties` dict, Mercator figure, ranges from `map_extent`. **Done for core path:** `vdapseisutils.core.maps.bokeh.Map` + **`add_terrain()`** (Esri + Carto via shared URLs in `map_tiles`).
- **`add_terrain()` / `add_arcgis_terrain()`:** two-layer Esri + Carto tiles; **single source of truth** for URLs in **`map_tiles`** (`ARCGIS_WORLD_HILLSHADE_URL`, `CARTO_LIGHT_NOLABELS_URL`).
- **`plot_heatmap`** + **`add_google_*`** landed on Bokeh `Map`; **`add_hillshade`** still deferred.
- **Explicitly deferred:** custom graticule / degree tick layout; rely on Bokeh defaults.

**Gallery:** `gallery/Mapping_tutorial_bokeh.ipynb` exercises Bokeh `Map` (Augustine + Kīlauea examples, terrain, catalog/inventory).

### Phase 3 — Richer map layers

- `plot_heatmap` (**done** on Bokeh `Map` / `CrossSection`; MPL heatmaps share bin helpers + default colorbar).
- `add_scalebar` (segment + label in canvas or data coords; reuse `choose_scale_bar_length` + geodesic width).
- `add_hillshade` (PyGMT raster as `image` glyph) if still needed when tiles are insufficient — **deferred**.

### Phase 4 — Polish

- Hover tools for catalog points, export PNG/SVG, kwargs adapter for common matplotlib→Bokeh aliases.
- World inset; residual kwargs parity as needed (scatter/catalog/inventory deepen incrementally).

---

## Dependencies

- **Runtime:** `bokeh` per project policy (e.g. `>=3.0`), existing `pyproj`/geodesy stack, optional `xyzservices` if it simplifies tile URLs.
- **Tests:** Smoke tests that build figures without network where possible now include Bokeh `Map` + `CrossSection` (`tests/test_bokeh_maps_smoke.py`); optional visual regression deferred.

---

## Validation

- Side-by-side screenshot comparison (optional): same extent, `add_terrain()` only, Cartopy `Map` vs `BokehMap` to confirm hillshade + overlay character match.
- Catalog scatter: same event count and approximate positions after projection.
- Cross-section: same projected x for a fixed catalog slice as matplotlib `CrossSection` within float tolerance.

---

## Open questions

- Exact **public import path** (`vdapseisutils.core.maps.BokehMap` vs `from vdapseisutils.core.maps.bokeh_map import Map`) and whether to add **`register_bokeh`** or similar beside `register_pyplot()` in `plot/mpl.py`.
- Factor **`get_default_terrain_tile_sources()`** (or equivalent) into `map_tiles.py` so Cartopy `add_arcgis_terrain` and Bokeh `add_terrain` stay synchronized without duplication.

---

## References (in-repo)

- **Canonical terrain behavior:** `vdapseisutils/core/maps/map_tiles.py` — `add_arcgis_terrain` (same recipe as historical REDPy).
- **MPL API to mirror:** `vdapseisutils/core/maps/map.py` — `Map`; `cross_section.py` — `CrossSection`.
- **Historical REDPy reference** (until directory deleted): `REDPy-prerelease-2.0.0/redpy/outputs/mapping.py` — `_get_tiles`, `_set_up_map_image`, `_add_basemaps`.
