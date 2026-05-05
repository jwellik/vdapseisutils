# Branch Summary: feature/bokeh-maps

## Metadata
- **Branch**: feature/bokeh-maps
- **Status**: open
- **Opened on**: 2026-05-04
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-05
P26-05-05
P26-05-05
P26-05-05
P26-05-05

## Accomplishments
- Added **`vdapseisutils.core.maps.bokeh.Map`** and stabilized projection initialization for environments with missing `proj.db` EPSG context.
- Implemented practical parity methods on Bokeh `Map`: **`plot()`**, **`scatter()`**, **`plot_catalog()`**, **`plot_inventory()`**, **`plot_volcano()`**, **`plot_peak()`**, and **`plot_line()`**, plus **`add_scalebar()`**, **`set_title()`** / **`set_catalog_subtitle()`**, and **`add_world_location_map()`** (side-by-side layout).
- Centralized ArcGIS/Carto terrain URL constants in **`map_tiles.py`** so Cartopy and Bokeh terrain paths share defaults.
- **`gallery/Mapping_tutorial_bokeh.ipynb`**: Bokeh parallels to `Mapping_tutorial.ipynb` (Augustine, regional peaks, Kīlauea with IRIS inventory + catalog + locator layout).

## Planned work
- Implement Bokeh `CrossSection` with MPL-compatible constructor and plotting methods.
- Extend Bokeh `Map` with **`plot_heatmap`**, optional hillshade raster support, and richer kwargs parity with MPL `Map`.
- Add tests and more gallery examples for catalog/inventory workflows.

## Executed work
- Created branch **`feature/bokeh-maps`** and delivered initial Bokeh map module, terrain integration, notebook demos, and documentation updates.
- Resolved editable-install/packaging troubleshooting notes in notebook documentation for mixed conda/pip environments.
- Updated branch timeline metadata and accomplishments as progress landed.

## Back-and-forth / iteration notes
- User requested nested naming (`vdapseisutils.core.maps.bokeh.Map`), Mount Augustine demos, then **`gallery/Mapping_tutorial_bokeh.ipynb`** beside **`Mapping_tutorial.ipynb`** with a Kīlauea figure matching the matplotlib tutorial cells.
- Grid/tick customization remains intentionally deferred for now.

## Problems + resolutions
- `pyproj` EPSG lookup failed (`proj_create: no database context specified`) in one environment; fixed by switching to explicit PROJ4 CRS strings in Bokeh map transformer.
- Environment had stale pip uninstall metadata; documented cleanup steps and kernel restart guidance in notebook.

## Validation
- Smoke tests pass for Bokeh `Map` import, terrain rendering call, and plotting methods (`plot`, `scatter`, `plot_line`, `plot_volcano`, `plot_peak`, `plot_inventory`, `plot_catalog`). Notebook exercises **`show(fig.layout)`** for the world-locator case.

## Final changelog-style outcome
- Pending merge: adds a new Bokeh map API surface under `vdapseisutils.core.maps.bokeh` plus gallery examples and terrain-sharing utilities.
