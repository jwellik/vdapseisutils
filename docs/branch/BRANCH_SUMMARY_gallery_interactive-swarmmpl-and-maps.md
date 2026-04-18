# Branch Summary: gallery/interactive-swarmmpl-and-maps

## Metadata
- **Branch**: gallery/interactive-swarmmpl-and-maps
- **Status**: closed
- **Opened on**: 2026-04-17
- **Closed on**: 2026-04-18
- **Merged into**: -
- **Merge strategy**: abandoned (branch deleted locally; not merged)
- **Last updated**: 2026-04-18

## Accomplishments
- Created the `gallery/interactive-swarmmpl-and-maps` branch for interactive gallery exploration.
- Captured the initial scope around `swarmmpl` Helicorder, Clipboard, and map examples.
- Documented a recommended interactive plotting stack and fallback options before implementation.
- Implemented an experimental Plotly + Panel interactive Helicorder path with datasource, `wave_id`, time-range, and timestamp-overlay controls.
- Added an offline gallery app script, a demo timestamp fixture, and focused tests for the new interactive slice.
- Added SDS directory support for the interactive Helicorder and made the default app window one day long.
- Refactored the app toward a viewer model with locked helicorder axes, explicit line/day navigation, cached SDS window loading, and verbose load/render logging.

## Platform recommendations
- **Recommended starting point: Plotly + Panel.** Plotly is a strong fit for zoomable time-series, heatmaps/spectrogram-like panels, hover inspection, and interactive map layers. Panel keeps the workflow Python-first and lighter-weight than a full web app while still supporting widgets, layouts, and local/served apps.
- **Use Dash if this becomes an app product.** Dash is a good choice if the goal shifts from interactive figures to a polished browser application with more routing, callbacks, and deployment structure, but it is probably heavier than needed for first-pass gallery exploration.
- **Keep Bokeh/HoloViews as a secondary option.** They are powerful for linked brushing and scientific dashboards, but they add more framework surface area and are less common than Plotly for quick interactive gallery examples.
- **Avoid notebook-only approaches as the primary target.** `ipywidgets` can be useful for exploration, but it is a weaker fit if the examples should live as shareable gallery artifacts outside notebooks.
- **Map-specific note:** if terrain tiles and leaflet-style layers become the main need, evaluate `folium` or `leafmap` for map examples, but start by seeing whether Plotly maps cover the required interactions so the branch can stay on one main stack.

## Planned work
- Generalize the interactive swarm path from the first Helicorder prototype into Clipboard-style multi-panel views.
- Evaluate whether map examples can share the same interactive stack or need a map-specific fallback.
- Decide whether to formalize a stable public interactive namespace after the prototype API settles.

## Executed work
- Opened the branch and created the required branch-tracking documentation.
- Wrote initial recommendations to guide the interactive plotting approach before implementation begins.
- Added `vdapseisutils.plot.swarm.interactive` as an experimental module for the first interactive Helicorder prototype.
- Added datasource loading paths for the packaged fixture, optional extended fixture, local waveform files, and SDS directories.
- Added `wave_id` selection, time-window controls, and catalog or generic timestamp-file overlay loading in the Panel app wrapper.
- Added on-demand SDS waveform fetching by `wave_id` and selected time window rather than requiring a preloaded stream.
- Changed the interactive app defaults to use a one-day time window and configured the gallery script to point at the Spurr SDS path by default.
- Reworked helicorder scaling to use a `one_bar_range`-style per-line percentile estimate and a separate clip threshold, which better matches the static `swarmmpl` logic.
- Replaced free plot zoom with viewer-style controls: previous/next line, previous/next day, latest-day jump, minutes shown per line, and line-view offset.
- Added an in-memory trace cache keyed by datasource, `wave_id`, start time, and duration so revisiting windows can be served from cache.
- Added terminal logging for SDS metadata scans, latest-day detection, cache hits/misses, file-backed waveform loads, and render-time scaling choices.
- Added `gallery/scripts/interactive_swarm_helicorder.py` as the first interactive gallery entrypoint.
- Added `data/fixtures/helicorder_demo_timestamps.csv` as an offline timestamp overlay example.
- Added focused tests covering datasource loading, timestamp overlay parsing, Plotly figure creation, and app construction.
- Declared `plotly` and `panel` in project dependencies.

## Back-and-forth / iteration notes
- The work expanded from branch setup into a concrete first implementation slice after choosing `Plotly + Panel`.
- The interactive Helicorder scope was refined to treat datasource choice, `wave_id`, time periods, and timestamp overlays as core user interactions instead of follow-on polish.
- Existing Matplotlib swarm APIs were left untouched; the interactive path lives in a separate experimental module.
- The datasource design was adjusted a second time so SDS could behave as a query-based source, with waveform retrieval driven by `wave_id` plus a one-day window.
- The first free-pan Plotly interaction model was not a good fit for helicorder semantics, so the app was shifted toward explicit viewer controls rather than generic chart zooming.

## Problems + resolutions
- **Problem:** The repo had no existing Plotly or Panel integration, so the first interactive slice needed both package metadata changes and a clean module boundary.
- **Resolution:** Add a new experimental module under `vdapseisutils.plot.swarm.interactive` and keep all existing Matplotlib-first imports and tests unchanged.
- **Problem:** The first interactive view needed to support real data-selection workflows without coupling rendering code to a single datasource.
- **Resolution:** Split the work into a datasource-loading layer, timestamp overlay parsing, and a lightweight Plotly helicorder builder consumed by a Panel app wrapper.
- **Problem:** SDS directories cannot be treated like a single preloaded waveform file because the selected `wave_id` and time window determine what should be fetched.
- **Resolution:** Add SDS-aware metadata loading plus on-demand waveform retrieval through the existing ObsPy client facade, and make the app default to one day of data per view.
- **Problem:** Free x/y Plotly interaction made the helicorder hard to read and did not match how analysts navigate line-wrapped waveform displays.
- **Resolution:** Lock free plot zoom, add line/day navigation controls, and treat horizontal zoom as a per-line viewer setting rather than an unconstrained chart transform.
- **Problem:** The first interactive scaling pass used the clip threshold as the display scale, which flattened the waveform rows.
- **Resolution:** Separate display scaling from clipping and compute a helicorder-style `one_bar_range` using interval-based percentiles.

## Validation
- Verified that the new branch summary file exists with required metadata and required sections.
- Verified that the branch timeline includes this branch as an open entry.
- Ran `python -m pytest tests/test_swarm_interactive.py` and confirmed the new interactive Helicorder tests pass.
- Ran `python -m pytest tests/test_swarm_canonical_api.py` and confirmed the existing Matplotlib swarm API smoke tests still pass.
- Checked edited files with the linter and found no new linter errors.
- Added and passed focused tests covering SDS metadata listing, SDS waveform querying, and one-day default time-window behavior.
- Added and passed focused tests covering locked helicorder axes, SDS load logging, and cache reuse inside the interactive viewer.

## Final changelog-style outcome
- Established a new branch for interactive `swarmmpl` and map exploration, selected `Plotly + Panel`, and delivered a more viewer-oriented experimental Helicorder with SDS-backed one-day windows, improved helicorder scaling, explicit navigation controls, cache-backed data loading, verbose terminal logs, and preserved Matplotlib swarm APIs.
- **Abandoned:** The branch was deleted locally without merge; interactive prototype files were not kept on `main`.
