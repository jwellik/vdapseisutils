"""
Plotly cross-section figure with a matplotlib-like method surface.

Each plotting / layout method returns the underlying :class:`plotly.graph_objects.Figure`
so callers can chain ``fig.show()`` or ``fig.write_html(...)``.

Requires the optional ``[plotly]`` dependency.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from vdapseisutils.core.maps.defaults_constants import (
    CROSSSECTION_DEFAULTS,
    HEATMAP_DEFAULTS,
    PLOT_CATALOG_DEFAULTS,
    PLOT_INVENTORY_DEFAULTS,
    PLOT_PEAK_DEFAULTS,
    PLOT_VOLCANO_DEFAULTS,
    SUBTITLE_DEFAULTS,
    TITLE_DEFAULTS,
)
from vdapseisutils.core.maps.legends import MagLegend
from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
from vdapseisutils.utils.geoutils import project2line

from .cross_section_data import (
    CrossSectionData,
    _along_line_km_1d,
    _z_to_axes_km,
    build_cross_section_data,
    prep_catalog_for_cross_section,
    prep_inventory_for_cross_section,
    project_latlon_to_cross_section,
)

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - guarded at runtime
    go = None  # type: ignore[misc, assignment]


def _require_plotly() -> Any:
    if go is None:
        raise ImportError(
            "Plotly is not installed. Install with: pip install 'vdapseisutils[plotly]'"
        )
    return go


def _mpl_marker_to_plotly(marker: str | None) -> str | None:
    if marker is None:
        return None
    table = {
        "o": "circle",
        "^": "triangle-up",
        "v": "triangle-down",
        "s": "square",
        "D": "diamond",
        ".": "circle",
        ",": "circle",
        "x": "x",
        "+": "cross",
    }
    return table.get(marker, marker)


def _mpl_color_to_plotly(c: Any) -> Any:
    """Map single-letter matplotlib colors to names Plotly accepts."""
    if isinstance(c, str) and len(c) == 1:
        return {
            "k": "black",
            "w": "white",
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
        }.get(c, c)
    return c


def _colorscale_from_mpl_cmap(name: str | None) -> tuple[str, bool] | None:
    if not name:
        return None
    n = str(name).lower().replace("_r", "")
    rev = str(name).lower().endswith("_r")
    plotly_name = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
        "turbo": "Turbo",
    }.get(n, "Viridis")
    return plotly_name, rev


def _series_to_numeric_colors(c: Any) -> Any:
    """Convert pandas datetime / categorical to plottable numeric for colorscales."""
    try:
        import pandas as pd

        if hasattr(c, "dtype"):
            if pd.api.types.is_datetime64_any_dtype(c):
                return pd.to_numeric(c.view("int64")) / 1e9
    except Exception:
        pass
    return c


def _scatter_kwargs_to_plotly_marker(
    *,
    s: Any,
    c: Any,
    color: Any,
    cmap: Any,
    alpha: float | None,
    edgecolors: Any,
    linewidths: Any,
    marker: Any,
) -> dict[str, Any]:
    """Map common matplotlib ``scatter`` kwargs to Plotly ``marker`` dict."""
    m: dict[str, Any] = {}
    if alpha is not None:
        m["opacity"] = float(alpha)

    use_c = color if color is not None else c
    if use_c is not None and not isinstance(use_c, str):
        arr = np.asarray(_series_to_numeric_colors(use_c))
        if arr.dtype == object:
            arr = np.asarray(use_c, dtype=float)
        m["color"] = arr
        m["showscale"] = True
        cs = _colorscale_from_mpl_cmap(cmap if cmap is not None else "viridis")
        if cs:
            name, rev = cs
            m["colorscale"] = name
            if rev:
                m["reversescale"] = True
    elif use_c is not None:
        m["color"] = _mpl_color_to_plotly(use_c)

    if s is not None:
        if np.ndim(s) == 0:
            m["size"] = float(s)
        else:
            m["size"] = np.asarray(s, dtype=float)
            m["sizemode"] = "diameter"

    sym = _mpl_marker_to_plotly(marker) if isinstance(marker, str) else None
    if sym:
        m["symbol"] = sym

    line: dict[str, Any] = {}
    if edgecolors is not None:
        line["color"] = _mpl_color_to_plotly(edgecolors)
    if linewidths is not None:
        lw = linewidths
        line["width"] = float(lw) if np.ndim(lw) == 0 else np.asarray(lw, dtype=float)
    if line:
        m["line"] = line

    return m


class CrossSectionPlotly:
    """
    Interactive cross-section on a :class:`plotly.graph_objects.Figure`.

    High-level kwargs align with :class:`~vdapseisutils.core.maps.cross_section.CrossSection`.
    Plotting methods return the same ``Figure`` instance (not ``self``).
    """

    name = "cross-section"

    def __init__(
        self,
        figure=None,
        points=None,
        origin=None,
        radius_km: float = 25.0,
        azimuth: float = 270,
        map_extent=None,
        depth_extent: tuple[float, float] = (-50.0, 4.0),
        resolution="auto",
        max_n: int = 100,
        label: str = "A",
        width: float | None = None,
        maglegend: MagLegend | None = None,
        verbose: bool = False,
        profile_source: str = "opentopo",
        layout_width: int | None = None,
        layout_height: int | None = None,
        **kwargs: Any,
    ) -> None:
        _require_plotly()
        if figure is None:
            figure = go.Figure()
        self.figure = figure
        if maglegend is None:
            maglegend = MagLegend()

        ignore = {"dpi", "figsize"}
        for k in ignore:
            kwargs.pop(k, None)

        self._data: CrossSectionData = build_cross_section_data(
            points=points,
            origin=origin,
            radius_km=radius_km,
            azimuth=azimuth,
            map_extent=map_extent,
            depth_extent=depth_extent,
            resolution=resolution,
            max_n=max_n,
            label=label,
            width=width,
            verbose=verbose,
            profile_source=profile_source,
        )
        self.properties = self._data.properties
        self.A1 = self._data.A1
        self.A2 = self._data.A2
        self.profile = self._data.profile
        self._maglegend = maglegend

        lw = layout_width if layout_width is not None else kwargs.pop("width_px", None)
        lh = layout_height if layout_height is not None else kwargs.pop("height_px", None)
        w = int(lw) if lw is not None else 700
        h = int(lh) if lh is not None else 420

        self.figure.update_layout(
            width=w,
            height=h,
            margin=dict(l=50, r=80, t=40, b=50),
            template="plotly_white",
            showlegend=True,
        )
        self._apply_axis_layout()
        if self._data.has_profile and len(self._data.profile_distance_km):
            self.plot(
                x=self._data.profile_distance_km,
                z=self._data.profile_elevation_km,
                z_dir="elev",
                z_unit="km",
                mode="lines",
                line=dict(
                    color=_mpl_color_to_plotly(CROSSSECTION_DEFAULTS["profile_color"]),
                    width=CROSSSECTION_DEFAULTS["profile_linewidth"],
                ),
                name="topography",
                showlegend=True,
            )
        self._add_section_end_labels()

    def _apply_axis_layout(self):
        xmin, xmax = self._data.horiz_extent_km
        ymin, ymax = self._data.depth_extent
        self.figure.update_xaxes(
            range=[xmin, xmax],
            title=dict(text=""),
            showgrid=True,
            zeroline=False,
        )
        self.figure.update_yaxes(
            range=[ymin, ymax],
            title=dict(text="Depth (km)", standoff=12),
            side="right",
            showgrid=True,
            zeroline=False,
        )
        return self.figure

    def set_depth_extent(self, depth_extent: tuple[float, float] | None = None) -> go.Figure:
        if depth_extent is None:
            depth_extent = self._data.depth_extent
        self.figure.update_yaxes(range=list(depth_extent))
        return self.figure

    def set_horiz_extent(self, extent: tuple[float, float] | None = None) -> go.Figure:
        if extent is None:
            extent = self._data.horiz_extent_km
        self.figure.update_xaxes(range=list(extent))
        return self.figure

    def _add_section_end_labels(self) -> None:
        x0, x1 = self._data.horiz_extent_km
        y0, y1 = self._data.depth_extent
        pad_x = (x1 - x0) * 0.03
        pad_y = (y1 - y0) * 0.03
        ya = y0 + pad_y
        self.figure.add_annotation(
            x=x0 + pad_x,
            y=ya,
            text=self._data.label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="black"),
        )
        self.figure.add_annotation(
            x=x1 - pad_x,
            y=ya,
            text=f"{self._data.label}'",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color="black"),
        )

    def plot(
        self,
        lat=None,
        lon=None,
        z=None,
        x=None,
        z_dir: str = "depth",
        z_unit: str = "m",
        **kwargs: Any,
    ) -> go.Figure:
        """Line plot in section coordinates; pass ``x`` + ``z`` **or** ``lat`` + ``lon`` + ``z``."""
        if x is None:
            if lat is None or lon is None:
                raise ValueError("Either (lat, lon) or x must be provided.")
            x_km, z_km = project_latlon_to_cross_section(
                lat, lon, z, self.A1, self.A2, z_dir=z_dir, z_unit=z_unit
            )
        else:
            x_km = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
            z_in = z if z is not None else np.zeros_like(x_km, dtype=float)
            z_km = _z_to_axes_km(z_in, z_dir=z_dir, z_unit=z_unit)

        mode = kwargs.pop("mode", "lines")
        name = kwargs.pop("name", None)
        showlegend = kwargs.pop("showlegend", True)
        line = kwargs.pop("line", None)
        color = kwargs.pop("color", None)
        linewidth = kwargs.pop("linewidth", None)
        linestyle = kwargs.pop("linestyle", None)
        alpha = kwargs.pop("alpha", None)
        label = kwargs.pop("label", None)

        line_d: dict[str, Any] = {}
        if line is not None:
            line_d.update(line)
        if color is not None:
            line_d.setdefault("color", _mpl_color_to_plotly(color))
        if linewidth is not None:
            line_d.setdefault("width", linewidth)
        if linestyle is not None:
            # rough mpl dash mapping
            dash_map = {"-": None, "--": "dash", ":": "dot", "-.": "dashdot"}
            line_d.setdefault("dash", dash_map.get(linestyle, linestyle))
        if alpha is not None:
            line_d.setdefault("opacity", alpha)

        self.figure.add_trace(
            go.Scatter(
                x=x_km,
                y=z_km,
                mode=mode,
                name=name or label,
                line=line_d if line_d else None,
                showlegend=showlegend,
                **kwargs,
            )
        )
        self.set_depth_extent()
        self.set_horiz_extent()
        return self.figure

    def scatter(
        self,
        lat=None,
        lon=None,
        z=None,
        x=None,
        z_dir: str = "depth",
        z_unit: str = "m",
        **kwargs: Any,
    ) -> go.Figure:
        """Scatter in section coordinates; pass ``x`` + ``z`` **or** ``lat`` + ``lon`` + ``z``."""
        if x is None:
            if lat is None or lon is None:
                raise ValueError("Either (lat, lon) or x must be provided.")
            x_km, z_km = project_latlon_to_cross_section(
                lat, lon, z, self.A1, self.A2, z_dir=z_dir, z_unit=z_unit
            )
        else:
            xz = z if z is not None else np.zeros_like(np.atleast_1d(x), dtype=float)
            x_km = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
            z_km = _z_to_axes_km(xz, z_dir=z_dir, z_unit=z_unit)

        s = kwargs.pop("s", 20)
        c = kwargs.pop("c", None)
        color = kwargs.pop("color", None)
        cmap = kwargs.pop("cmap", None)
        alpha = kwargs.pop("alpha", None)
        edgecolors = kwargs.pop("edgecolors", None)
        linewidths = kwargs.pop("linewidths", None)
        marker = kwargs.pop("marker", None)
        name = kwargs.pop("name", None)
        label = kwargs.pop("label", None)
        showlegend = kwargs.pop("showlegend", True)
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)

        mk = _scatter_kwargs_to_plotly_marker(
            s=s,
            c=c,
            color=color,
            cmap=cmap,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
            marker=marker,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x_km,
                y=z_km,
                mode="markers",
                name=name or label,
                marker=mk,
                showlegend=showlegend,
                **kwargs,
            )
        )
        self.set_depth_extent()
        self.set_horiz_extent()
        return self.figure

    def plot_catalog(
        self,
        catalog: Any,
        s=PLOT_CATALOG_DEFAULTS["s"],
        c=PLOT_CATALOG_DEFAULTS["c"],
        color=PLOT_CATALOG_DEFAULTS["color"],
        cmap=PLOT_CATALOG_DEFAULTS["cmap"],
        alpha=PLOT_CATALOG_DEFAULTS["alpha"],
        time_format: str = "matplotlib",
        **kwargs: Any,
    ) -> go.Figure:
        prep = prep_catalog_for_cross_section(
            catalog,
            self.A1,
            self.A2,
            time_format=time_format,
            maglegend=self._maglegend,
        )
        catdata = prep["catdata"]
        if s == "magnitude":
            s = catdata["size"]
        if color is not None:
            c = color
        elif c == "time":
            c = catdata["time"]

        return self.scatter(
            lat=catdata["lat"],
            lon=catdata["lon"],
            z=catdata["depth"],
            z_dir="elev",
            z_unit="km",
            s=s,
            c=c,
            cmap=cmap,
            alpha=alpha,
            name=kwargs.pop("name", "catalog"),
            **kwargs,
        )

    def plot_inventory(
        self,
        inventory: Any,
        s=PLOT_INVENTORY_DEFAULTS["s"],
        c=PLOT_INVENTORY_DEFAULTS["c"],
        alpha=PLOT_INVENTORY_DEFAULTS["alpha"],
        **kwargs: Any,
    ) -> go.Figure:
        try:
            prep = prep_inventory_for_cross_section(inventory, self.A1, self.A2)
            if prep is None:
                print("No valid station coordinates found in inventory for cross-section")
                return self.figure
            plot_kwargs = {**PLOT_INVENTORY_DEFAULTS, **kwargs}
            plot_kwargs.update({"s": s, "c": c, "alpha": alpha})
            return self.scatter(
                lat=prep["lat"],
                lon=prep["lon"],
                z=prep["elevation_m"],
                z_dir="elev",
                z_unit="m",
                name=plot_kwargs.pop("name", "inventory"),
                **plot_kwargs,
            )
        except Exception as e:
            print(f"Error plotting inventory on cross-section: {e}")
            print("Continuing without cross-section inventory plot...")
            return self.figure

    def plot_volcano(self, lat: float, lon: float, elev: float, **kwargs: Any) -> go.Figure:
        plot_kwargs = {**PLOT_VOLCANO_DEFAULTS, **kwargs}
        return self.scatter(
            lat=lat,
            lon=lon,
            z=elev,
            z_dir="elev",
            z_unit="m",
            name=plot_kwargs.pop("name", "volcano"),
            **plot_kwargs,
        )

    def plot_peak(self, lat: float, lon: float, elev: float, **kwargs: Any) -> go.Figure:
        plot_kwargs = {**PLOT_PEAK_DEFAULTS, **kwargs}
        return self.scatter(
            lat=lat,
            lon=lon,
            z=elev,
            z_dir="elev",
            z_unit="m",
            name=plot_kwargs.pop("name", "peak"),
            **plot_kwargs,
        )

    def plot_heatmap(
        self,
        *args: Any,
        grid_size=HEATMAP_DEFAULTS["grid_size"],
        cmap=HEATMAP_DEFAULTS["cmap"],
        alpha=HEATMAP_DEFAULTS["alpha"],
        vmin=HEATMAP_DEFAULTS["vmin"],
        vmax=HEATMAP_DEFAULTS["vmax"],
        **kwargs: Any,
    ) -> go.Figure:
        """Same calling patterns as matplotlib ``CrossSection.plot_heatmap``."""
        try:
            if len(args) == 1 and hasattr(args[0], "events"):
                catalog = args[0]
                catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
                lat = catdata["lat"].values
                lon = catdata["lon"].values
                depth_km = np.asarray(catdata["depth"].values, dtype=float)
            elif len(args) >= 2:
                lat = np.asarray(args[0])
                lon = np.asarray(args[1])
                depth = np.asarray(args[2]) if len(args) > 2 else None
                depth_km = np.asarray(depth, dtype=float) / 1000.0
                depth_km = -depth_km
            else:
                raise ValueError(
                    "Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)"
                )

            if len(lat) == 0 or len(lon) == 0:
                warnings.warn("Empty coordinate arrays", stacklevel=2)
                return self.figure

            x = _along_line_km_1d(lat, lon, self.A1, self.A2)
            if len(x) == 0 or np.any(np.isnan(x)):
                warnings.warn(
                    "Failed to project coordinates to cross-section line", stacklevel=2
                )
                return self.figure

            x_min, x_max = float(np.min(x)), float(np.max(x))
            depth_min, depth_max = float(np.min(depth_km)), float(np.max(depth_km))

            grid_size_km = float(grid_size) * 111.0
            if grid_size_km < 0.1:
                grid_size_km = 0.1
            data_range_x = x_max - x_min
            data_range_depth = depth_max - depth_min
            min_grid_size = max(0.1, min(data_range_x, data_range_depth) * 0.1)
            if grid_size_km < min_grid_size:
                grid_size_km = min_grid_size

            x_pad = (x_max - x_min) * 0.1
            depth_pad = (depth_max - depth_min) * 0.1

            if x_max <= x_min or depth_max <= depth_min:
                warnings.warn("Invalid coordinate ranges for heatmap", stacklevel=2)
                return self.figure

            x_grid = np.arange(x_min - x_pad, x_max + x_pad, grid_size_km)
            depth_grid = np.arange(depth_min - depth_pad, depth_max + depth_pad, grid_size_km)

            if len(x_grid) < 2 or len(depth_grid) < 2:
                warnings.warn("Grid too small for heatmap", stacklevel=2)
                return self.figure

            H, xedges, yedges = np.histogram2d(x, depth_km, bins=[x_grid, depth_grid])

            if H.size == 0 or np.all(H == 0):
                warnings.warn("No data points in the specified region", stacklevel=2)
                return self.figure

            x_centers = (xedges[:-1] + xedges[1:]) / 2
            depth_centers = (yedges[:-1] + yedges[1:]) / 2

            cs_name, rev = _colorscale_from_mpl_cmap(cmap) or ("Plasma", False)
            zmin = None if vmin is None else float(vmin)
            zmax = None if vmax is None else float(vmax)

            self.figure.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=depth_centers,
                    z=H.T,
                    opacity=float(alpha) if alpha is not None else 0.7,
                    colorscale=cs_name,
                    reversescale=bool(rev),
                    zmin=zmin,
                    zmax=zmax,
                    name="heatmap",
                    **kwargs,
                )
            )
            self.set_depth_extent()
            self.set_horiz_extent()
            return self.figure
        except Exception as e:
            warnings.warn(f"Heatmap skipped: {e}", stacklevel=2)
            return self.figure

    def set_title(self, title_text: str, **kwargs: Any) -> go.Figure:
        params = {**TITLE_DEFAULTS, **kwargs}
        self.figure.update_layout(
            title=dict(
                text=title_text,
                font=dict(
                    size=14 if params.get("fontsize") == "large" else 12,
                    color=params.get("color", "black"),
                    family="Arial Black" if params.get("fontweight") == "bold" else None,
                ),
                x=params.get("x", 0.5),
                xanchor=params.get("ha", "center"),
                y=params.get("y", 0.98),
                yanchor=params.get("va", "top"),
            )
        )
        return self.figure

    def set_subtitle(self, subtitle_text: str, **kwargs: Any) -> go.Figure:
        params = {**SUBTITLE_DEFAULTS, **kwargs}
        self.figure.add_annotation(
            xref="paper",
            yref="paper",
            x=params.get("x", 0.5),
            y=params.get("y", 0.90),
            text=subtitle_text,
            showarrow=False,
            font=dict(
                size=11 if params.get("fontsize") == "medium" else 10,
                color=params.get("color", "black"),
            ),
            xanchor="center",
        )
        return self.figure

    def set_titles(
        self,
        title_text: str | None = None,
        subtitle_text: str | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        title_kwargs = {
            k.replace("title_", ""): v for k, v in kwargs.items() if k.startswith("title_")
        }
        subtitle_kwargs = {
            k.replace("subtitle_", ""): v
            for k, v in kwargs.items()
            if k.startswith("subtitle_")
        }
        if title_text:
            self.set_title(title_text, **title_kwargs)
        if subtitle_text:
            self.set_subtitle(subtitle_text, **subtitle_kwargs)
        return self.figure

    def set_catalog_subtitle(self, catalog: Any, **kwargs: Any) -> go.Figure:
        from vdapseisutils.obspy_ext.catalog import VCatalog

        if not isinstance(catalog, VCatalog):
            vcatalog = VCatalog(catalog)
        else:
            vcatalog = catalog
        summary_str = vcatalog.short_summary_str()
        return self.set_subtitle(summary_str, **kwargs)
