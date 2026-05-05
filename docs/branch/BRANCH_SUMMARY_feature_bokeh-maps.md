# Branch Summary: feature/bokeh-maps

## Metadata
- **Branch**: feature/bokeh-maps
- **Status**: open
- **Opened on**: 2026-05-04
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-05

## Accomplishments
- **`vdapseisutils.core.maps.bokeh.Map`**: proj-safe Mercator init; **`add_terrain`** / **`add_google_*`**; plotting surface including **`plot_heatmap`**; **`add_scalebar`**, titles, **`add_world_location_map`**.
- **`vdapseisutils.core.maps.bokeh.CrossSection`**: MPL-like **`plot`**, **`scatter`**, **`plot_catalog`**, **`plot_inventory`**, **`plot_heatmap`**, with matplotlib-style kwargs parity covered in smoke tests (**`color`/`c`**, **`cmap`**, inventory marker/edges).
- **`map_tiles.py`**: shared ArcGIS/Carto terrain URLs plus Google XYZ templates/attribution consumed by Cartopy and Bokeh tile layers.
- **MPL heatmaps**: default **`colorbar`** on **`Map.plot_heatmap`** and **`CrossSection.plot_heatmap`**; shared **`heatmap_utils.histogram_bin_edges`** prevents **`histogram2d`** from dropping samples at the padded range ends (MPL + Bokeh).
- **MPL `CrossSection.set_horiz_extent`**: supports **`points`** profiles using geodesic **`length`** when **`radius`** is unset; gallery Bokeh notebooks mirror the matplotlib tutorials.

## Planned work
- **`add_hillshade`** / raster parity on Bokeh maps when needed beyond XYZ terrain stacks.
- Optional: **`register_bokeh`**, shared **`get_default_terrain_tile_sources()`**, finer graticule/tick formatting.

## Executed work
- Created branch **`feature/bokeh-maps`** and delivered Bokeh map and cross-section modules, terrain and heatmap integration, google tiles, notebook demos, plan/timeline updates, and expanding smoke tests (`tests/test_bokeh_maps_smoke.py`, **`tests/test_maps_stack_smoke.py`** heatmap/colorbar checks).
- Resolved editable-install/packaging troubleshooting notes in notebook documentation for mixed conda/pip environments.
- Updated branch timeline metadata and accomplishments as progress landed.

## Back-and-forth / iteration notes
- User requested nested naming (`vdapseisutils.core.maps.bokeh.Map`), Mount Augustine demos, then **`gallery/Mapping_tutorial_bokeh.ipynb`** beside **`Mapping_tutorial.ipynb`** with a Kīlauea figure matching the matplotlib tutorial cells.
- Grid/tick customization remains intentionally deferred for now.

## Problems + resolutions
- `pyproj` EPSG lookup failed (`proj_create: no database context specified`) in one environment; fixed by switching to explicit PROJ4 CRS strings in Bokeh map transformer.
- Environment had stale pip uninstall metadata; documented cleanup steps and kernel restart guidance in notebook.

## Validation
- **`pytest -q tests/test_bokeh_maps_smoke.py tests/test_maps_stack_smoke.py`** passes (terrain + google tiles, heatmaps, catalog/scatter kwargs, MPL heatmap colorbars).

## Final changelog-style outcome
- Pending merge: adds Bokeh **`Map`** / **`CrossSection`** under **`vdapseisutils.core.maps.bokeh`**, shared **`heatmap_utils`**, google XYZ parity, MPL heatmap colorbar defaults, gallery notebooks, and smoke coverage for the above.
