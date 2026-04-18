# CrossSection (Plotly) vs matplotlib `CrossSection`

`vdapseisutils.core.maps.plotly.cross_section_plotly.CrossSectionPlotly` mirrors the public workflow of `vdapseisutils.core.maps.cross_section.CrossSection` for typical volcano-map style use (profile, scatter, catalog, inventory, heatmap, titles). The matplotlib class remains the default API exported from `vdapseisutils` until maintainers opt to expose Plotly in `__init__.py`.

## What matches

- **Constructor inputs:** `points` / `origin` + `azimuth` + `radius_km`, `depth_extent`, `map_extent`, `resolution`, `max_n`, `label`, `width`, `maglegend`, `verbose` (matplotlib-only `fig` / `dpi` / `figsize` are ignored; use Plotly `layout_width` / `layout_height` or `width_px` / `height_px`).
- **Endpoints and geometry:** `A1`, `A2`, `properties`, and the shared data bundle from `build_cross_section_data` / `prep_catalog_for_cross_section`.
- **Catalog / inventory:** `plot_catalog`, `plot_inventory`, `plot_volcano`, `plot_peak` use the same defaults (`PLOT_*_DEFAULTS`) and `prep_catalog_data_mpl` + `MagLegend.mag2s` for `s="magnitude"`.
- **Magnitude vs time:** Default `plot_catalog` uses magnitude-sized markers and time-based coloring. Plotly shows a **continuous colorbar** titled “Time” when `c="time"`; discrete **M…** marker sizes are available via `add_magnitude_legend()` or `plot_catalog(..., show_magnitude_legend=True)` (default when `s="magnitude"`).
- **Heatmap:** Same calling patterns and binning idea as matplotlib; implementation uses `go.Heatmap`.
- **Titles:** `set_title`, `set_subtitle`, `set_titles`, `set_catalog_subtitle`, `save_html`, `show`.

## What differs (by design)

- **Renderer:** Plotly `Figure` only (no `Axes`). Axis titles: explicit **Distance along profile (km)** on *x*; **Depth (km)** on the **right** *y* (matplotlib leaves *x* unlabeled and appends `km` on the last tick).
- **Topography:** Drawn as a line trace when elevation data exist; no matplotlib spine tricks.

## Unsupported or not replicated

- **Custom spine bounds** on the matplotlib axes (e.g. `_add_profile` tying the top spine to surface elevation) — Plotly has no equivalent; the axis rectangle is always a box.
- **Pixel-perfect** match to matplotlib tick text, font metrics, or `MagLegend.display` layout — Plotly uses native colorbars and legend groups instead.
- **`VolcanoFigure` integration** — still matplotlib-only; use `CrossSectionPlotly` in standalone scripts or notebooks until a future layout wrapper exists.

## Workarounds

- **Spine / surface alignment:** Rely on the topography line and fixed `depth_extent`; adjust `depth_extent` or annotations if you need more margin near the surface.
- **Magnitude key without clutter:** Pass `show_magnitude_legend=False` to `plot_catalog` and call `add_magnitude_legend()` once when you want the size scale, or hide the main trace from the legend with `plot_catalog(..., showlegend=False)` on the catalog trace if needed.
