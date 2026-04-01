# Maps and volcano layout (API v1)

This page summarizes the **map stack** types named in `.local/api-v1-coord/API_V1_CANONICAL.md` §4–§5 and §11: how they fit together, where to import them from, and how they relate to compute helpers.

## Types and roles

- **`Map`** — Cartopy-backed map figure helper: Mercator axes, extent from an origin and radial distance (or an explicit `map_extent`), gridlines, and additive methods such as `plot_catalog`, terrain tiles, and scale bar. The geographic axes remain plain Cartopy / Matplotlib objects so `ax.scatter`, `ax.plot`, and `ax.imshow` keep working.
- **`CrossSection`** — Vertical cross-section axes along a great-circle line (from two `(lat, lon)` points or from `origin`, `azimuth`, and `radius_km`). Optional topography along the line uses online elevation services when available; offline runs may omit the profile if the download fails.
- **`TimeSeries`** — Time–depth or time–magnitude panel used next to the map in volcano layouts. The x-axis is formatted for seismic catalog times (ObsPy date helpers).
- **`MagLegend`** — Maps magnitude to scatter marker sizes (`mag2s`) and can draw a small magnitude scale (`display`). Catalog plotting paths reuse this for consistent symbol sizing.
- **`VolcanoFigure`** — A **composed** `matplotlib.figure.Figure` that wires **`Map`**, two **`CrossSection`** instances (`xs1_obj`, `xs2_obj`), **`TimeSeries`** (`ts_obj`), and a legend subfigure. It is not a separate map engine; it is the standard multi-panel layout for volcano monitoring-style plots. Domain methods such as `scatter` and `plot_catalog` fan out to the panels that apply.

## Import paths

Canonical implementations live under **`vdapseisutils.core.maps`** (split modules). The module **`vdapseisutils.core.maps.maps`** is a **shim** that re-exports the same symbols for older import paths.

```python
from vdapseisutils.core.maps import Map, VolcanoFigure, CrossSection, TimeSeries, MagLegend
# Equivalent for legacy code:
from vdapseisutils.core.maps.maps import Map, VolcanoFigure, CrossSection, TimeSeries, MagLegend
```

The top-level package **`vdapseisutils`** also re-exports **`VolcanoFigure`**, **`Map`**, **`CrossSection`**, **`TimeSeries`**, and **`MagLegend`** from `vdapseisutils/__init__.py` (see that file for the current list).

## Pyplot constructors (`register_pyplot`)

After registering helpers on Matplotlib’s pyplot module, you can construct the same types as `plt.eqmap(...)` and `plt.volcano(...)`:

```python
import matplotlib.pyplot as plt
import vdapseisutils.plot.mpl as vsmpl

vsmpl.register_pyplot()
fig_map = plt.eqmap(...)
fig_volc = plt.volcano(...)
```

See `vdapseisutils.plot.mpl` and README “Pyplot helpers” for details.

## Catalog compute (backend-neutral)

For tables of hypocenter fields without Matplotlib objects, use **`vdapseisutils.compute.catalog.prepare_catalog_points`** (and related helpers described in the package docstring). Plotting code may consume those results or ObsPy catalogs; **`prep_catalog_data_mpl`** in `core/maps` remains a thin matplotlib-oriented wrapper around the same preparation steps where needed.

## Runnable examples

From the repository root (with the package on `PYTHONPATH` or installed editable), each script uses the Agg backend, draws once, and exits without writing files:

- `python examples/map_minimal.py`
- `python examples/volcano_figure_layout.py`
- `python examples/cross_section_standalone.py`

Or: `uv run python examples/map_minimal.py` if you use uv-managed environments.

## Further reading

- Canonical spec: `.local/api-v1-coord/API_V1_CANONICAL.md` (§0 preservation, §4–§5 types, §10 tests, §11 docs).
- Gallery-style tutorials may live under `gallery/`; these `examples/` scripts are minimal offline smoke workflows.
