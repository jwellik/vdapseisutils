"""
Bokeh :class:`CrossSection` with MPL-compatible constructor and core plot methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure as bk_figure

from vdapseisutils.core.maps.defaults import default_volcano
from vdapseisutils.utils.geoutils import backazimuth, project2line, sight_point_pyproj


def _parse_figure_dimensions(kwargs: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    """Resolve ``figsize`` / ``dpi`` / ``width`` / ``height`` similarly to matplotlib."""
    kw = dict(kwargs)
    figsize = kw.pop("figsize", None)
    dpi = kw.pop("dpi", 100)
    width = kw.pop("width", None)
    height = kw.pop("height", None)
    if width is None and figsize is not None:
        w_in, h_in = figsize
        width = int(w_in * dpi)
        height = int(h_in * dpi)
    if width is None:
        width = 700
    if height is None:
        height = 350
    return width, height, kw


def _compute_depth_km(z, z_dir="depth", z_unit="m"):
    """Convert depth/elevation inputs into cross-section y values in km."""
    if z is None:
        return None
    depth = np.asarray(z)
    if z_unit.lower() == "km":
        z_unit_conv = 1.0
    elif z_unit.lower() == "m":
        z_unit_conv = 1.0 / 1000.0
    else:
        raise ValueError(f"Invalid z_unit '{z_unit}'. Options: 'km' or 'm'.")

    if z_dir.lower() == "depth":
        z_dir_conv = -1.0
    elif z_dir.lower() == "elev":
        z_dir_conv = 1.0
    else:
        raise ValueError(f"Invalid z_dir '{z_dir}'. Options: 'depth' or 'elev'.")
    return depth * z_unit_conv * z_dir_conv


class CrossSection:
    """
    Bokeh-backed cross section using the same constructor surface as MPL CrossSection.

    This scaffold focuses on constructor parity plus ``plot`` and ``scatter``.
    """

    name = "cross-section"

    _FIGURE_KW = frozenset(
        {
            "title",
            "tools",
            "toolbar_location",
            "active_scroll",
            "active_drag",
            "active_tap",
            "output_backend",
        }
    )

    def __init__(
        self,
        fig=None,
        points=[(46.198776, -122.261317), (46.197484, -122.122234)],
        origin=None,
        radius_km=25.0,
        azimuth=270,
        map_extent=None,
        depth_extent=(-50.0, 4.0),
        resolution="auto",
        max_n=100,
        label="A",
        width=None,
        maglegend=None,
        verbose=False,
        **kwargs,
    ):
        _ = resolution
        _ = max_n
        _ = maglegend
        width_px, height_px, plot_kwargs = _parse_figure_dimensions(kwargs)

        self.properties: dict[str, Any] = {}
        if origin:
            self.properties["origin"] = origin
            self.properties["azimuth"] = azimuth
            self.properties["radius"] = radius_km * 1000.0
            self.properties["points"] = [np.nan, np.nan]
            self.properties["points"][0] = sight_point_pyproj(
                origin, azimuth, self.properties["radius"]
            )
            self.properties["points"][1] = sight_point_pyproj(
                origin, np.mod(azimuth + 180, 360), self.properties["radius"]
            )
            self.properties["length"] = self.properties["radius"] * 2.0
        else:
            if len(points) != 2:
                raise ValueError(
                    "ERROR: Points must be a list of 2 tuples of lat,lon coordinates."
                )
            self.properties["points"] = points
            self.properties["origin"] = None
            self.properties["azimuth"], self.properties["length"] = backazimuth(
                points[0], points[1]
            )
            self.properties["radius"] = None

        self.properties["map_extent"] = map_extent
        self.properties["depth_extent"] = depth_extent
        self.properties["depth_range"] = depth_extent[1] - depth_extent[0]
        self.properties["label"] = label
        self.properties["full_label"] = f"{label}-{label}'"
        self.properties["width"] = width
        self.properties["orientation"] = "horizontal"

        self.verbose = verbose
        self.A1 = self.properties["points"][0]
        self.A2 = self.properties["points"][1]

        if fig is None:
            x_end_km = self._default_horiz_extent()[1]
            fig_kwargs = {k: v for k, v in plot_kwargs.items() if k in self._FIGURE_KW}
            self.figure = bk_figure(
                x_range=(0.0, x_end_km),
                y_range=depth_extent,
                width=width_px,
                height=height_px,
                **fig_kwargs,
            )
        else:
            self.figure = fig
            self.figure.y_range.start = depth_extent[0]
            self.figure.y_range.end = depth_extent[1]

        self.figure.grid.visible = False
        self.figure.xaxis.axis_label = ""
        self.figure.yaxis.axis_label = "Depth (km)"
        self.figure.yaxis.axis_label_text_font_style = "normal"

    def _default_horiz_extent(self):
        if self.properties["radius"] is not None:
            return (0.0, self.properties["radius"] * 2.0 / 1000.0)
        if self.properties.get("length") is not None:
            return (0.0, self.properties["length"] / 1000.0)
        return (0.0, 50.0)

    def info(self):
        """Print cross-section properties."""
        print("::: BOKEH CROSS SECTION :::")
        print(self.properties)
        print()

    def set_depth_extent(self, depth_extent=None):
        """Set y-axis depth extent in km."""
        if depth_extent is None:
            depth_extent = self.properties["depth_extent"]
        else:
            self.properties["depth_extent"] = depth_extent
        self.figure.y_range.start = float(depth_extent[0])
        self.figure.y_range.end = float(depth_extent[1])
        return self

    def set_horiz_extent(self, extent=None):
        """Set x-axis extent in km along profile."""
        if extent is None:
            extent = self._default_horiz_extent()
        self.figure.x_range.start = float(extent[0])
        self.figure.x_range.end = float(extent[1])
        return self

    def _compute_x_from_latlon(self, lat=None, lon=None, x=None):
        if x is not None:
            return np.asarray(x)
        if lat is None or lon is None:
            raise ValueError("Either (lat, lon) or x must be provided.")
        return np.asarray(project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km"))

    def plot(self, lat=None, lon=None, z=None, x=None, z_dir="depth", z_unit="m", **kwargs):
        """Plot line data on the cross section (MPL-compatible signature)."""
        x_vals = self._compute_x_from_latlon(lat=lat, lon=lon, x=x)
        if z is None:
            z = np.zeros_like(x_vals)
        depth = _compute_depth_km(z, z_dir=z_dir, z_unit=z_unit)
        return self.figure.line(x=x_vals, y=depth, **kwargs)

    def scatter(
        self,
        lat=None,
        lon=None,
        z=None,
        x=None,
        z_dir="depth",
        z_unit="m",
        hover_tooltips=None,
        hover_formatters=None,
        hover_text=None,
        **kwargs,
    ):
        """Scatter data on the cross section (MPL-compatible signature)."""
        x_vals = self._compute_x_from_latlon(lat=lat, lon=lon, x=x)
        if z is None:
            z = np.zeros_like(x_vals)
        depth = _compute_depth_km(z, z_dir=z_dir, z_unit=z_unit)
        scatter_kwargs = dict(kwargs)
        if "c" in scatter_kwargs and "color" not in scatter_kwargs:
            scatter_kwargs["color"] = scatter_kwargs.pop("c")
        if "s" in scatter_kwargs and "size" not in scatter_kwargs:
            s = np.asarray(scatter_kwargs.pop("s"), dtype=float)
            scatter_kwargs["size"] = np.sqrt(np.clip(s, a_min=0.0, a_max=None))

        if hover_text is not None:
            x_arr = np.atleast_1d(x_vals)
            y_arr = np.atleast_1d(depth)
            h_arr = np.atleast_1d(hover_text).astype(str)
            if h_arr.size == 1 and x_arr.size > 1:
                h_arr = np.repeat(h_arr, x_arr.size)
            source_data = {"x": x_arr, "y": y_arr, "hover_text": h_arr}
            size_val = scatter_kwargs.get("size")
            if isinstance(size_val, (list, tuple, np.ndarray)):
                source_data["size"] = np.atleast_1d(size_val)
                scatter_kwargs["size"] = "size"
            source = ColumnDataSource(source_data)
            renderer = self.figure.scatter(x="x", y="y", source=source, **scatter_kwargs)
            if hover_tooltips is None:
                hover_tooltips = [("label", "@hover_text")]
        else:
            renderer = self.figure.scatter(x=x_vals, y=depth, **scatter_kwargs)

        if hover_tooltips:
            hover = HoverTool(
                renderers=[renderer],
                tooltips=hover_tooltips,
                formatters=hover_formatters or {},
            )
            self.figure.add_tools(hover)
        return renderer


__all__ = ["CrossSection"]
