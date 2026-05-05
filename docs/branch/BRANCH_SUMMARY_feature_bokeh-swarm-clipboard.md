# Branch Summary: feature/bokeh-swarm-clipboard

## Metadata
- **Branch**: feature/bokeh-swarm-clipboard
- **Status**: open
- **Opened on**: 2026-05-05
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
- **Last updated**: 2026-05-05

## Accomplishments
- Implemented **Phase 3** overlays and navigation on **`SwarmClipboardBk`**: **`axvline`** (metadata targeting and **`axes`** indices), **`plot_peak_value`** raster behind waveform axes, **`plot_trace`** / **`plot_horizontals`**, **`plot_catalog`**, **`scroll_traces`**, plus **`set_alim`** / **`set_flim`** for y-limits.
- Completed **Phase 4**: gallery notebook **`gallery/SwarmMPL/Clipboard_tutorial_bokeh.ipynb`** (Examples 1–3 aligned with **`Clipboard_Tutorial_A.ipynb`**), README **“Bokeh clipboard”** subsection, extended **`tests/test_swarm_clipboard_bokeh.py`**, and marked Phases 3–4 complete in **`docs/plans/bokeh-swarm-clipboard-plan.md`**.
- Fixed **`plot_peak_value`** horizontal **`ColorBar`** placement for Bokeh 3 (**`location='bottom_center'`**).

## Planned work
- Phase 5 Helicorder Bokeh milestone (separate scope per plan).

## Executed work
- Added **`_resolve_target_panels`** / **`_figures_for_axes`** helpers and **`_PanelRecord`** wiring for **`wg`** stacked panels.
- Notebook documents **`SwarmClipboardBk`** usage with **`output_notebook`**, **`show`**, and **`save`** HTML artifacts under **`gallery/SwarmMPL/`**.

## Back-and-forth / iteration notes
- Tutorial A labels **“Example 3”** twice (Examples 2 and 3 in narrative order); the Bokeh notebook follows the **three worked examples** pattern from the plan (simple clip → filtered markers → Augustine relative + scroll).

## Problems + resolutions
- **`ColorBar`** rejected **`location='below'`** on this Bokeh version — use **`bottom_center`** when **`orientation='horizontal'`**.
- **`Span`** annotations attach to **`figure.center`**, not glyph **`renderers`** — tests count spans accordingly.

## Validation
- **`python3 -m pytest tests/test_swarm_clipboard_bokeh.py`** passes (15 tests).

## Final changelog-style outcome
- **`feature/bokeh-swarm-clipboard`**: Phase 3–4 deliverables landed — overlay/navigation API on **`SwarmClipboardBk`**, gallery notebook + README + tests + plan checklist updates.
