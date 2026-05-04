# Branch Summary: feature/plotly-interactive-views

## Metadata
- **Branch**: feature/plotly-interactive-views
- **Status**: open
- **Opened on**: 2026-04-18
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-05-04

## Accomplishments
- Branch `feature/plotly-interactive-views` hosts optional Plotly-backed **Map** and **CrossSection** work plus related matplotlib map/volcano improvements (not Plotly Swarm clipboard UIs).
- Documented branch-only policy and phased prompts in `docs/plotly_map_crosssection_plan.md`; Plotly ports of SwarmMPL **Clipboard** / **SwarmClipboard** were removed from scope (matplotlib remains canonical there).
- Added optional `[plotly]` extra and `vdapseisutils.core.maps.plotly` with `CrossSectionPlotly`, data layer, tests, `examples/cross_section_plotly_standalone.py`, and `docs/plotly_cross_section_PARITY.md`.

## Planned work
- Map Plotly Parts 1–3 per `docs/plotly_map_crosssection_plan.md` (six phased prompts total for CrossSection + Map).

## Executed work
- Branch created; plan committed with branch-only policy and prompt updates.
- CrossSection Plotly Part 1: `pyproject` `[plotly]` extra, `cross_section_data` + `empty_cross_section_figure`, README note, tests.
- CrossSection Plotly Part 2: `cross_section_plotly.CrossSectionPlotly` with profile scatter, heatmap/scatter/catalog paths, tick styling, HTML export, tests, standalone example; lazy export in `plotly` package `__init__`.
- CrossSection Plotly Part 3: parity documentation, expanded standalone example, colorbar and magnitude-legend behavior, branch summary maintenance.
- Matplotlib-side improvements merged on the same branch: grid defaults, `TimeSeries.plot_eventrate` grid docs, terrain kwargs on map tiles, peak-value raster overlay on `ClipboardClass` / `SwarmClipboard`, `VelocityModel1D`, volcano figure extensions, branch docs tooling.

## Back-and-forth / iteration notes
- Narrowed Plotly scope to maps cross-section only; no in-repo Plotly Swarm clipboard implementation was started beyond the former plan text.

## Problems + resolutions
- Branch summary metadata had stray lines from an earlier template sync; restored a valid metadata block on 2026-05-04.

## Validation
- Run `pytest` (including `tests/test_cross_section_plotly_*.py` when Plotly is installed).

## Final changelog-style outcome
- 
