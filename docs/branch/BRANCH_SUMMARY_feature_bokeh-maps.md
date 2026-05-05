# Branch Summary: feature/bokeh-maps

## Metadata
- **Branch**: feature/bokeh-maps
- **Status**: open
- **Opened on**: 2026-05-04
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-04
P26-05-04
P26-05-04
P26-05-04
P26-05-05

## Accomplishments
- Added **`vdapseisutils.core.maps.bokeh.Map`**: Web Mercator `bokeh.plotting.figure`, constructor aligned with MPL `Map`, **`add_terrain()`** (Esri + Carto) using shared URL constants in **`map_tiles`**. Public import: **`from vdapseisutils.core.maps.bokeh import Map`**.
- Refactored **`map_tiles.add_arcgis_terrain`** to use **`ARCGIS_WORLD_HILLSHADE_URL`** and **`CARTO_LIGHT_NOLABELS_URL`** (single source of truth for Cartopy and Bokeh).
- Added **`gallery/bokeh/Mount_Augustine_map.ipynb`**: inline Bokeh map of Mount Augustine, Alaska.
- Drafted and updated **`docs/plans/bokeh-maps-plan.md`**: API parity, REDPy removal policy, nested `Map` naming, branch-doc maintenance.

## Planned work
- Implement **Bokeh `CrossSection`** (linear axes; `TopographicProfile`, `project2line`, `prep_catalog_data_mpl`).
- Extend Bokeh **`Map`**: **`plot_catalog()`, `plot_inventory()`, `plot_volcano()`, `plot_peak()`, `plot_line()`, `scatter()`, `plot()`**, heatmap, scale bar, etc., matching MPL `Map`.
- Optional `register_*` hook beside `plot/mpl.py` if useful; tests and docstrings.

## Executed work
- Created branch **`feature/bokeh-maps`**; landed **`core/maps/bokeh/`** package, **`map_tiles`** URL constants, **`gallery/bokeh`** notebook, plan doc updates, branch docs/timeline.

## Back-and-forth / iteration notes
- User requested branch summary and timeline **updated whenever meaningful progress lands** (planning counts until code exists).
- Grid/tick polish for Bokeh maps explicitly deferred (“for meow”) in the written plan.

## Problems + resolutions
- None yet.

## Validation
- Smoke test: `from vdapseisutils.core.maps.bokeh import Map`; **`Map(...); add_terrain()`** builds a figure without error.

## Final changelog-style outcome
- Pending: merge to main will note introduction of Bokeh map API and any new deps or entry points.
