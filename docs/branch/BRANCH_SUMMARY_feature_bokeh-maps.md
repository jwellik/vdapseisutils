# Branch Summary: feature/bokeh-maps

## Metadata
- **Branch**: feature/bokeh-maps
- **Status**: open
- **Opened on**: 2026-05-04
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-04

## Accomplishments
- Drafted **`docs/plans/bokeh-maps-plan.md`**: phased approach for Bokeh-backed map types, API parity with Matplotlib `Map` / `CrossSection`, and terrain basemaps aligned with **`map_tiles.add_arcgis_terrain`** (Esri hillshade + Carto overlay, same as historical REDPy behavior).
- Documented that **`REDPy-prerelease-2.0.0/`** will be removed from the repo; shipping code must not depend on it—behavior is encoded in vdapseisutils only.
- Clarified requirements: same **`plot_catalog()`, `plot_inventory()`, `add_terrain()`,** and related method signatures as the Cartopy/Matplotlib map classes; defer detailed graticule/tick customization initially.
- Established maintenance expectation: **branch summary and `BRANCH_TIMELINE.md`** stay current as implementation progresses (update accomplishments/executed work and bump **Last updated**, then run `python scripts/branch_docs.py regenerate-timeline` or rely on pre-commit).

## Planned work
- Implement **`BokehCrossSection`** first (linear axes; reuse `TopographicProfile`, `project2line`, `prep_catalog_data_mpl`).
- Implement **`BokehMap`** with Mercator figure, Web Mercator coordinates, **`add_terrain()`** via shared tile URLs/zoom with `map_tiles.py`.
- Add catalog/inventory/volcano/peak/line/heatmap methods matching MPL APIs; optional `register_*` hook beside `plot/mpl.py` if useful.
- Factor shared **`get_default_terrain_tile_sources()`** (or similar) if Cartopy and Bokeh both need the same tile definitions without duplication.
- Tests (smoke / optional visual checks) and optional `bokeh` optional dependency in packaging.

## Executed work
- Created branch **`feature/bokeh-maps`**.
- Added and iterated **`docs/plans/bokeh-maps-plan.md`** (terrain audit, API parity table, REDPy removal policy, phased milestones).
- Updated **`docs/branch/BRANCH_SUMMARY_feature_bokeh-maps.md`** and regenerated **`BRANCH_TIMELINE.md`** to record planning-phase progress.

## Back-and-forth / iteration notes
- User requested branch summary and timeline **updated whenever meaningful progress lands** (planning counts until code exists).
- Grid/tick polish for Bokeh maps explicitly deferred (“for meow”) in the written plan.

## Problems + resolutions
- None yet.

## Validation
- Plan aligns with existing **`Map`** / **`CrossSection`** and **`map_tiles.add_arcgis_terrain`** implementation in-tree.

## Final changelog-style outcome
- Pending: merge to main will note introduction of Bokeh map API and any new deps or entry points.
