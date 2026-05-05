"""
Bokeh :class:`CrossSection` with MPL-compatible constructor and core plot methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, Title
from bokeh.plotting import figure as bk_figure

from vdapseisutils.core.maps import elev_profile
from vdapseisutils.core.maps.defaults import (
    PLOT_CATALOG_DEFAULTS,
    PLOT_INVENTORY_DEFAULTS,
    SUBTITLE_DEFAULTS,
    TITLE_DEFAULTS,
)
from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
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


def _mpl_fontsize_to_pt(size: Any) -> str:
    if isinstance(size, (int, float)):
        return f"{float(size)}pt"
    table = {
        "xx-small": "7pt",
        "x-small": "8pt",
        "smaller": "8pt",
        "small": "9pt",
        "medium": "12pt",
        "large": "14pt",
        "x-large": "16pt",
        "xx-large": "18pt",
        "larger": "18pt",
    }
    return table.get(str(size).lower(), "12pt")


def _stringify_hover_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
        self.profile = elev_profile.TopographicProfile(
            [self.A1, self.A2], resolution=resolution, max_n=max_n
        )

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

        self._add_profile()
        self.set_horiz_extent()
        self.set_depth_extent()

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

    def _add_profile(self):
        """Draw topographic profile if elevation data are available."""
        if self.profile is None or len(self.profile.distance) == 0:
            return None
        hd = np.asarray(self.profile.distance) / 1000.0
        elev_km = np.asarray(self.profile.elevation) / 1000.0
        return self.figure.line(
            x=hd,
            y=elev_km,
            line_color="black",
            line_width=1.5,
            line_alpha=0.9,
        )

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

    def plot_catalog(
        self,
        catalog,
        s=PLOT_CATALOG_DEFAULTS["s"],
        c=PLOT_CATALOG_DEFAULTS["c"],
        color=PLOT_CATALOG_DEFAULTS["color"],
        cmap=PLOT_CATALOG_DEFAULTS["cmap"],
        alpha=PLOT_CATALOG_DEFAULTS["alpha"],
        **kwargs,
    ):
        """Plot ObsPy catalog projected along the cross-section line."""
        _ = cmap
        hover_tooltips = kwargs.pop("hover_tooltips", None)
        hover_formatters = kwargs.pop("hover_formatters", None)
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
        if s == "magnitude":
            s = catdata["size"]
        if color is not None:
            c = color
        elif c == "time":
            c = catdata["time"]
        x = np.atleast_1d(
            project2line(catdata["lat"], catdata["lon"], P1=self.A1, P2=self.A2, unit="km")
        )
        y = np.atleast_1d(np.asarray(catdata["depth"]))
        size = np.sqrt(np.clip(np.asarray(s, dtype=float), a_min=0.0, a_max=None))
        size = np.atleast_1d(size)
        if size.size == 1 and x.size > 1:
            size = np.repeat(size, x.size)
        source = ColumnDataSource(
            {
                "x": np.asarray(x),
                "y": y,
                "size": size,
                "time_str": np.asarray([str(t) for t in catdata["time"]]),
                "mag_str": np.asarray([_stringify_hover_value(v) for v in catdata["mag"]]),
                "depth_str": np.asarray([_stringify_hover_value(v) for v in catdata["depth"]]),
                "lat_str": np.asarray([_stringify_hover_value(v) for v in catdata["lat"]]),
                "lon_str": np.asarray([_stringify_hover_value(v) for v in catdata["lon"]]),
            }
        )
        glyph_kwargs = dict(kwargs)
        glyph_kwargs.setdefault("alpha", alpha)
        if isinstance(c, str):
            renderer = self.figure.scatter(
                x="x", y="y", size="size", source=source, color=c, **glyph_kwargs
            )
        else:
            renderer = self.figure.scatter(
                x="x", y="y", size="size", source=source, color="royalblue", **glyph_kwargs
            )
        if hover_tooltips is None:
            hover_tooltips = [
                ("time", "@time_str"),
                ("mag", "@mag_str"),
                ("depth (km)", "@depth_str"),
                ("lat", "@lat_str"),
                ("lon", "@lon_str"),
            ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=hover_tooltips,
            formatters=hover_formatters or {},
        )
        self.figure.add_tools(hover)
        return renderer

    def plot_inventory(
        self,
        inventory,
        s=PLOT_INVENTORY_DEFAULTS["s"],
        c=PLOT_INVENTORY_DEFAULTS["c"],
        alpha=PLOT_INVENTORY_DEFAULTS["alpha"],
        **kwargs,
    ):
        """Plot station inventory projected onto the cross-section."""
        hover_tooltips = kwargs.pop("hover_tooltips", None)
        hover_formatters = kwargs.pop("hover_formatters", None)
        station_lats = []
        station_lons = []
        station_elevs = []
        networks = []
        stations = []
        for network in inventory:
            for station in network:
                if hasattr(station, "latitude") and hasattr(station, "longitude"):
                    station_lats.append(station.latitude)
                    station_lons.append(station.longitude)
                    station_elevs.append(getattr(station, "elevation", 0.0))
                    networks.append(getattr(network, "code", "") or "")
                    stations.append(getattr(station, "code", "") or "")
        if not station_lats:
            return None
        x = np.atleast_1d(
            project2line(station_lats, station_lons, P1=self.A1, P2=self.A2, unit="km")
        )
        y = np.atleast_1d(np.asarray(station_elevs, dtype=float) / 1000.0)
        size = np.sqrt(np.clip(np.asarray(s, dtype=float), a_min=0.0, a_max=None))
        size = np.atleast_1d(size)
        if size.size == 1 and x.size > 1:
            size = np.repeat(size, x.size)
        source = ColumnDataSource(
            {
                "x": np.asarray(x),
                "y": y,
                "size": np.asarray(size),
                "station_id": np.asarray(
                    [f"{n}.{sta}" if n else sta for n, sta in zip(networks, stations)],
                    dtype=str,
                ),
                "lat_str": np.asarray([_stringify_hover_value(v) for v in station_lats]),
                "lon_str": np.asarray([_stringify_hover_value(v) for v in station_lons]),
                "elev_str": np.asarray([_stringify_hover_value(v) for v in station_elevs]),
            }
        )
        renderer = self.figure.scatter(
            x="x",
            y="y",
            size="size",
            source=source,
            color=c,
            alpha=alpha,
            marker="inverted_triangle",
            **kwargs,
        )
        if hover_tooltips is None:
            hover_tooltips = [
                ("station", "@station_id"),
                ("lat", "@lat_str"),
                ("lon", "@lon_str"),
                ("elev (m)", "@elev_str"),
            ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=hover_tooltips,
            formatters=hover_formatters or {},
        )
        self.figure.add_tools(hover)
        return renderer

    def set_title(self, title_text: str, **kwargs: Any):
        title_params = {**TITLE_DEFAULTS, **kwargs}
        self.figure.title = Title(
            text=title_text,
            text_font_size=_mpl_fontsize_to_pt(title_params["fontsize"]),
            text_font_style="bold" if title_params.get("fontweight") == "bold" else "normal",
            text_color=title_params.get("color", "black"),
            align="center",
        )
        return self

    def set_subtitle(self, subtitle_text: str, **kwargs: Any):
        subtitle_params = {**SUBTITLE_DEFAULTS, **kwargs}
        subt = Title(
            text=subtitle_text,
            text_font_size=_mpl_fontsize_to_pt(subtitle_params["fontsize"]),
            text_font_style="bold" if subtitle_params.get("fontweight") == "bold" else "normal",
            text_color=subtitle_params.get("color", "black"),
            align="center",
            standoff=2,
        )
        self.figure.add_layout(subt, "above")
        return self


__all__ = ["CrossSection"]
