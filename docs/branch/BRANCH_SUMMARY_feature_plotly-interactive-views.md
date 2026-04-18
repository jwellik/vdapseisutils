# Branch Summary: feature/plotly-interactive-views

## Metadata
- **Branch**: feature/plotly-interactive-views
- **Status**: open
- **Opened on**: 2026-04-18
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
P26-04-18

## Accomplishments
- Created branch `feature/plotly-interactive-views` for Plotly-backed Map, CrossSection, and Clipboard work.
- Documented mandatory branch policy and aligned copy-paste prompts in `docs/plotly_map_crosssection_clipboard_plan.md`.
- Added optional `[plotly]` extra and a matplotlib-free cross-section data layer under `vdapseisutils.core.maps.plotly` (Part 1 foundation).
- CrossSection Plotly Part 2: `CrossSectionPlotly` core figure with `go.Heatmap` / `go.Scatter`, axis titles (distance vs depth km), styling from `CROSSSECTION_DEFAULTS` and `TICK_DEFAULTS`, `save_html` / `show`, and `examples/cross_section_plotly_standalone.py`.
- CrossSection Plotly Part 3: time colorbar + `MagLegend`-style `add_magnitude_legend` / `plot_catalog(show_magnitude_legend=...)`, heatmap count colorbar, `docs/plotly_cross_section_PARITY.md`, `VolcanoFigure` integration note, and tests.

## Planned work
- CrossSection → Map → Clipboard Plotly ports per `docs/plotly_map_crosssection_clipboard_plan.md` (nine phased prompts).

## Executed work
- Branch created; plan committed with branch-only policy and prompt updates.
- Part 1: `pyproject` `[plotly]` extra, `cross_section_data` + `empty_cross_section_figure`, README note, tests.
- Part 2: `cross_section_plotly.CrossSectionPlotly` with profile scatter, heatmap/scatter/catalog paths, tick and spine styling, HTML export, tests, standalone Plotly example; lazy export in `plotly` package `__init__`.
- Part 3: parity documentation, expanded standalone example (synthetic scatter + demo catalog), colorbar and magnitude-legend behavior, branch summary fixes.

## Back-and-forth / iteration notes
- 

## Problems + resolutions
- 

## Validation
- `pytest` full suite (65 passed, local).

## Final changelog-style outcome
- 
