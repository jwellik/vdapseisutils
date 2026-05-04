# Branch Summary: feature/plotly-interactive-views

## Metadata
- **Branch**: feature/plotly-interactive-views
- **Status**: closed
- **Opened on**: 2026-04-18
- **Closed on**: 2026-05-04
- **Merged into**: main
- **Merge strategy**: merge commit
- **Last updated**: 2026-05-04

## Accomplishments
- Landed matplotlib-focused improvements: `plot_peak_value` on `ClipboardClass` / `SwarmClipboard`, quieter default grids on Map / CrossSection / TimeSeries / event-rate plots, `TimeSeries.plot_eventrate` grid documentation, terrain tile kwargs cleanup, `VelocityModel1D`, `VolcanoFigure` extensions, catalog `plot_eventrate_from_times(..., grid=False)` default, and README usage for peak overlays.
- Added repo branch-doc automation (`scripts/branch_docs.py`), template refresh, gallery figure list note, and `.gitignore` entries for local tooling paths.
- Explored an optional Plotly cross-section stack (`vdapseisutils.core.maps.plotly`, tests, example); that code was **removed before merge** so `main` ships without Plotly figure implementations or a `[plotly]` extra.

## Planned work
- None (branch closed; Plotly figure work was abandoned rather than merged).

## Executed work
- Implemented and then deleted Plotly cross-section modules, tests, standalone example, plan/parity markdown, `pyproject` `[plotly]` extra, and README Plotly section so the merge to `main` contains only the retained matplotlib and packaging/doc tooling changes above.

## Back-and-forth / iteration notes
- Scope narrowed earlier from Clipboard/Swarm Plotly to Map/CrossSection only; final decision was to drop Plotly implementations entirely and keep matplotlib-only deliverables.

## Problems + resolutions
- **Problem:** Optional Plotly dependency and parallel API surface added maintenance and CI surface without a committed product direction.
- **Resolution:** Remove Plotly package subtree and related docs/tests before merging the rest of the branch work to `main`.

## Validation
- `python -m pytest` on `tests/` after removing Plotly-only tests (full suite per local environment).

## Final changelog-style outcome
- Merged to `main`: map/volcano/time-series/grid defaults, `VelocityModel1D`, clipboard peak raster overlay, branch documentation automation, and related README / catalog plotting tweaks—**no** in-tree Plotly figures or `[plotly]` extra.
