# Branch Summary: feature/bokeh-maps

## Metadata
- **Branch**: feature/bokeh-maps
- **Status**: closed
- **Opened on**: 2026-05-04
- **Closed on**: 2026-05-05
- **Merged into**: main
- **Merge strategy**: fast-forward
- **Last updated**: 2026-05-05

## Accomplishments
- **Bokeh `Map` / `CrossSection`**: Mercator figures with **`add_terrain`** (Esri + Carto stack), **`add_google_*`** XYZ tiles, core plotting (**`plot_heatmap`**, catalog/inventory scatter-family parity tested), **`add_scalebar`**, titles, locator layout (**`add_world_location_map`**), gallery notebooks beside matplotlib tutorials.
- **Shared stack**: **`map_tiles`** URLs/attribution for Cartopy + Bokeh; **`heatmap_utils`** (**`heatmap_bin_step`**, **`histogram_bin_edges`**) shared by MPL and Bokeh; MPL **`plot_heatmap`** default **colorbars** aligned with Bokeh; MPL **`CrossSection.set_horiz_extent`** supports **`points`** profiles via geodesic **`length`** when **`radius`** is unset.
- **Deferred intentionally (post-merge):** Bokeh **`add_hillshade`** / richer raster parity; custom **graticule** / degree ticks (**`set_ticks`** stubs); **`register_bokeh`** and optional **`get_default_terrain_tile_sources()`**; Phase **4** polish (PNG/SVG export, systematic matplotlib→Bokeh kwargs adapter); optional visual regression checks vs Cartopy terrain.

## Planned work
- Branch is **closed** after fast-forward merge to **`main`**. Remaining scope is captured under **Deferred work (explicit)** below.

## Deferred work (explicit)
- **`add_hillshade` on Bokeh `Map`**: optional PyGMT (or similar) raster / **`image`** glyph path when XYZ terrain is not enough; keep MPL **`add_hillshade`** behavior as the reference.
- **Axis and graticule polish**: meaningful **`set_ticks`** / **`set_ticks_outside`** (and degree formatters) on Bokeh maps instead of default Mercator ticks only.
- **Packaging / API ergonomics**: consider **`register_bokeh`** alongside **`register_pyplot()`** in **`plot/mpl.py`**; factor **`get_default_terrain_tile_sources()`** (or equivalent) in **`map_tiles.py`** so Cartopy **`add_arcgis_terrain`** and Bokeh **`add_terrain`** stay synchronized without drift.
- **Phase 4 polish**: stronger hover/export story, broader kwargs adapter, incremental scatter/catalog/inventory edge cases beyond current smoke tests.
- **Cross-section label styling**: matplotlib **`path_effects`** on **A** / **A′** equivalents in Bokeh (called out as “for meow” in the plan); accept softer styling or solid label backgrounds until then.
- **Optional validation**: side-by-side screenshots (Cartopy **`Map`** vs Bokeh **`Map`** with **`add_terrain()`** only); numeric checks that projected cross-section distances match MPL within tolerance.

## Executed work
- Created branch **`feature/bokeh-maps`** and delivered Bokeh map and cross-section modules, terrain and heatmap integration, google tiles, notebook demos, plan/timeline updates, and expanding smoke tests (`tests/test_bokeh_maps_smoke.py`, **`tests/test_maps_stack_smoke.py`** heatmap/colorbar checks).
- Resolved editable-install/packaging troubleshooting notes in notebook documentation for mixed conda/pip environments.
- Updated branch timeline metadata and accomplishments as progress landed; closed branch documentation after merge readiness review.

## Back-and-forth / iteration notes
- User requested nested naming (`vdapseisutils.core.maps.bokeh.Map`), Mount Augustine demos, then **`gallery/Mapping_tutorial_bokeh.ipynb`** beside **`Mapping_tutorial.ipynb`** with a Kīlauea figure matching the matplotlib tutorial cells.
- Grid/tick customization remains intentionally deferred for now.

## Problems + resolutions
- `pyproj` EPSG lookup failed (`proj_create: no database context specified`) in one environment; fixed by switching to explicit PROJ4 CRS strings in Bokeh map transformer.
- Environment had stale pip uninstall metadata; documented cleanup steps and kernel restart guidance in notebook.

## Validation
- **`pytest -q tests/test_bokeh_maps_smoke.py tests/test_maps_stack_smoke.py`** passes (terrain + google tiles, heatmaps, catalog/scatter kwargs, MPL heatmap colorbars).

## Final changelog-style outcome
- **Merged to `main`** (fast-forward): adds Bokeh **`Map`** / **`CrossSection`** under **`vdapseisutils.core.maps.bokeh`**, shared **`heatmap_utils`**, google XYZ parity, MPL heatmap colorbar defaults, gallery notebooks, and smoke coverage for the above; defer **`add_hillshade`** and other polish per **Deferred work (explicit)**.
