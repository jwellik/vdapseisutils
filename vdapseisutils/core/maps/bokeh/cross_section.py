"""
Bokeh :class:`CrossSection` with MPL-compatible constructor and core plot methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, Label, LinearColorMapper, Title
from bokeh.palettes import Viridis256
from bokeh.plotting import figure as bk_figure
from bokeh.transform import linear_cmap

from vdapseisutils.core.maps import elev_profile
from vdapseisutils.core.maps.bokeh.heatmap_common import (
    heatmap_bin_step,
    palette_for_heatmap_cmap,
    remove_layout_annotation,
)
from vdapseisutils.core.maps.defaults import (
    HEATMAP_DEFAULTS,
    PLOT_CATALOG_DEFAULTS,
    PLOT_INVENTORY_DEFAULTS,
    PLOT_PEAK_DEFAULTS,
    PLOT_VOLCANO_DEFAULTS,
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


def _normalize_mpl_color(color):
    """Convert matplotlib single-letter colors to CSS names for Bokeh."""
    if not isinstance(color, str):
        return color
    cmap = {
        "b": "blue",
        "g": "green",
        "r": "red",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
    }
    return cmap.get(color, color)


def _bokeh_marker(marker: str | None) -> str | None:
    """Map common matplotlib marker letters to Bokeh marker names."""
    if marker is None:
        return None
    m = {
        "v": "inverted_triangle",
        "^": "triangle",
        "s": "square",
        "o": "circle",
        "D": "diamond",
        "d": "diamond",
    }
    return m.get(str(marker), str(marker))


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
        debug_corner_labels=False,
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
        self._debug_corner_labels = bool(debug_corner_labels)
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
        self._add_corner_labels()
        self._heatmap_colorbar: ColorBar | None = None

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

    def _add_corner_labels(self):
        """Add fixed A / A' corner labels that stay put while panning/zooming."""
        label = str(self.properties.get("label", "A"))
        fig_w = int(getattr(self.figure, "width", 700) or 700)
        toolbar_loc = str(getattr(self.figure, "toolbar_location", "right") or "right")
        # Reserve screen-space where the toolbar lives so labels remain visible.
        toolbar_reserve_px = 42
        left_pad = 8 + (toolbar_reserve_px if toolbar_loc == "left" else 0)
        right_pad = 8 + (toolbar_reserve_px if toolbar_loc == "right" else 0)
        # Hybrid inset: proportional to width but never less than 56 px.
        right_inset_px = max(56, int(fig_w * 0.06))
        x_right = max(left_pad + 16, fig_w - right_pad - right_inset_px)
        common = dict(
            y=8,
            y_units="screen",
            x_units="screen",
            text_baseline="bottom",
            text_font_style="bold",
            text_color="black",
            background_fill_color="white",
            background_fill_alpha=0.75,
            border_line_alpha=0.0,
        )
        self.figure.add_layout(
            Label(
                x=left_pad,
                text=label,
                text_align="left",
                **common,
            )
        )
        self.figure.add_layout(
            Label(
                x=x_right,
                text=f"{label}'",
                text_align="right",
                **common,
            )
        )
        if self._debug_corner_labels:
            debug_common = dict(
                y=26,
                y_units="screen",
                x_units="screen",
                text_baseline="bottom",
                text_font_style="bold",
                text_color="red",
                background_fill_color="white",
                background_fill_alpha=0.9,
                border_line_alpha=0.25,
                border_line_color="red",
            )
            self.figure.add_layout(
                Label(
                    x=left_pad,
                    text=f"DBG A x={left_pad}px",
                    text_align="left",
                    **debug_common,
                )
            )
            self.figure.add_layout(
                Label(
                    x=x_right,
                    text=f"DBG A' x={x_right}px (w={fig_w}, inset={right_inset_px}, tb={toolbar_loc})",
                    text_align="right",
                    **debug_common,
                )
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
        x_vals = np.atleast_1d(self._compute_x_from_latlon(lat=lat, lon=lon, x=x))
        if z is None:
            z = np.zeros_like(x_vals)
        depth = np.atleast_1d(_compute_depth_km(z, z_dir=z_dir, z_unit=z_unit))
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
        x_vals = np.atleast_1d(self._compute_x_from_latlon(lat=lat, lon=lon, x=x))
        if z is None:
            z = np.zeros_like(x_vals)
        depth = np.atleast_1d(_compute_depth_km(z, z_dir=z_dir, z_unit=z_unit))
        scatter_kwargs = dict(kwargs)
        c_alias = scatter_kwargs.pop("c", None)
        if "color" not in scatter_kwargs and c_alias is not None:
            scatter_kwargs["color"] = c_alias
        s_alias = scatter_kwargs.pop("s", None)
        if "size" not in scatter_kwargs and s_alias is not None:
            s = np.asarray(s_alias, dtype=float)
            scatter_kwargs["size"] = np.sqrt(np.clip(s, a_min=0.0, a_max=None))
        if "edgecolors" in scatter_kwargs and "line_color" not in scatter_kwargs:
            scatter_kwargs["line_color"] = scatter_kwargs.pop("edgecolors")
        if "linewidths" in scatter_kwargs and "line_width" not in scatter_kwargs:
            scatter_kwargs["line_width"] = scatter_kwargs.pop("linewidths")
        if "color" in scatter_kwargs:
            scatter_kwargs["color"] = _normalize_mpl_color(scatter_kwargs["color"])
        if "line_color" in scatter_kwargs:
            scatter_kwargs["line_color"] = _normalize_mpl_color(scatter_kwargs["line_color"])
        mk = scatter_kwargs.pop("marker", None)
        if mk:
            scatter_kwargs["marker"] = _bokeh_marker(mk) or mk

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
        plot_kwargs = dict(kwargs)
        if "c" in plot_kwargs and color is None:
            c = plot_kwargs.pop("c")
        if "s" in plot_kwargs:
            s = plot_kwargs.pop("s")
        if "edgecolors" in plot_kwargs and "line_color" not in plot_kwargs:
            plot_kwargs["line_color"] = plot_kwargs.pop("edgecolors")
        if "linewidths" in plot_kwargs and "line_width" not in plot_kwargs:
            plot_kwargs["line_width"] = plot_kwargs.pop("linewidths")
        if "marker" in plot_kwargs:
            mk = plot_kwargs.get("marker")
            plot_kwargs["marker"] = _bokeh_marker(mk) or mk
        if isinstance(c, str):
            c = _normalize_mpl_color(c)
        if "line_color" in plot_kwargs:
            plot_kwargs["line_color"] = _normalize_mpl_color(plot_kwargs["line_color"])
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
        plot_kwargs.setdefault("alpha", alpha)
        c_array = np.asarray(c) if hasattr(c, "__len__") and not isinstance(c, str) else None
        if isinstance(c, str):
            renderer = self.figure.scatter(
                x="x", y="y", size="size", source=source, color=c, **plot_kwargs
            )
        elif c_array is not None and c_array.size == len(x):
            source.data["cval"] = c_array
            mapper = linear_cmap("cval", Viridis256, low=float(np.nanmin(c_array)), high=float(np.nanmax(c_array)))
            renderer = self.figure.scatter(
                x="x",
                y="y",
                size="size",
                source=source,
                fill_color=mapper,
                line_color=mapper,
                **plot_kwargs,
            )
        else:
            renderer = self.figure.scatter(
                x="x",
                y="y",
                size="size",
                source=source,
                color=_normalize_mpl_color(c),
                **plot_kwargs,
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
        plot_kwargs = dict(kwargs)
        if "edgecolors" in plot_kwargs and "line_color" not in plot_kwargs:
            plot_kwargs["line_color"] = plot_kwargs.pop("edgecolors")
        if "linewidths" in plot_kwargs and "line_width" not in plot_kwargs:
            plot_kwargs["line_width"] = plot_kwargs.pop("linewidths")
        mk = plot_kwargs.pop("marker", "inverted_triangle")
        renderer = self.figure.scatter(
            x="x",
            y="y",
            size="size",
            source=source,
            color=_normalize_mpl_color(c),
            alpha=alpha,
            marker=_bokeh_marker(mk) or mk,
            **plot_kwargs,
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

    def set_titles(self, title_text: str | None = None, subtitle_text: str | None = None, **kwargs):
        """Set title and subtitle with MPL-style prefixed kwargs."""
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
        return self

    def set_catalog_subtitle(self, catalog, **kwargs):
        """Subtitle from catalog summary (same as matplotlib CrossSection)."""
        from vdapseisutils.obspy_ext.catalog import VCatalog

        if not isinstance(catalog, VCatalog):
            vcatalog = VCatalog(catalog)
        else:
            vcatalog = catalog
        summary_str = vcatalog.short_summary_str()
        return self.set_subtitle(summary_str, **kwargs)

    def plot_volcano(self, lat, lon, elev=0, **kwargs):
        """Plot volcano location on the cross section."""
        plot_kwargs = {**PLOT_VOLCANO_DEFAULTS, **kwargs}
        return self.scatter(lat=lat, lon=lon, z=elev, z_dir="elev", z_unit="m", **plot_kwargs)

    def plot_peak(self, lat, lon, elev=0, **kwargs):
        """Plot peak location on the cross section."""
        plot_kwargs = {**PLOT_PEAK_DEFAULTS, **kwargs}
        return self.scatter(lat=lat, lon=lon, z=elev, z_dir="elev", z_unit="m", **plot_kwargs)

    def plot_heatmap(
        self,
        *args,
        grid_size=HEATMAP_DEFAULTS["grid_size"],
        cmap=HEATMAP_DEFAULTS["cmap"],
        alpha=HEATMAP_DEFAULTS["alpha"],
        vmin=HEATMAP_DEFAULTS["vmin"],
        vmax=HEATMAP_DEFAULTS["vmax"],
        **kwargs,
    ):
        """Plot event-density heatmap on cross-section (catalog or lat/lon/depth arrays)."""
        if len(args) == 1 and hasattr(args[0], "events"):
            catdata = prep_catalog_data_mpl(args[0], time_format="matplotlib")
            lat = np.asarray(catdata["lat"])
            lon = np.asarray(catdata["lon"])
            depth_km = np.asarray(catdata["depth"], dtype=float)
        elif len(args) >= 2:
            lat = np.asarray(args[0])
            lon = np.asarray(args[1])
            depth = np.asarray(args[2]) if len(args) > 2 else None
            if depth is None:
                raise ValueError("depth is required when calling plot_heatmap(lat, lon, depth, ...).")
            depth_km = -np.asarray(depth, dtype=float) / 1000.0
        else:
            raise ValueError("Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, depth, ...)")

        if lat.size == 0 or lon.size == 0:
            return None

        x = np.asarray(project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km"), dtype=float)
        if x.size == 0 or np.all(~np.isfinite(x)):
            return None

        finite_mask = np.isfinite(x) & np.isfinite(depth_km)
        if not np.any(finite_mask):
            return None
        x = x[finite_mask]
        depth_km = depth_km[finite_mask]

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        y_min, y_max = float(np.nanmin(depth_km)), float(np.nanmax(depth_km))
        if x_max <= x_min or y_max <= y_min:
            return None

        data_range_x = x_max - x_min
        data_range_y = y_max - y_min
        extent_short = min(data_range_x, data_range_y)
        colorbar = kwargs.pop("colorbar", True)
        colorbar_title = kwargs.pop("colorbar_title", "Event count")
        colorbar_location = kwargs.pop("colorbar_location", "right")
        max_heatmap_bins = int(kwargs.pop("max_heatmap_bins", 120))
        requested_km = max(float(grid_size) * 111.0, 1e-3)
        grid_size_km = heatmap_bin_step(
            extent_short, requested_km, floor=1e-3, max_bins=max_heatmap_bins
        )

        x_pad = data_range_x * 0.1
        y_pad = data_range_y * 0.1
        x_grid = np.arange(x_min - x_pad, x_max + x_pad + grid_size_km, grid_size_km)
        y_grid = np.arange(y_min - y_pad, y_max + y_pad + grid_size_km, grid_size_km)
        if x_grid.size < 2 or y_grid.size < 2:
            return None

        H, xedges, yedges = np.histogram2d(x, depth_km, bins=[x_grid, y_grid])
        if H.size == 0 or np.all(H == 0):
            return None

        xs = []
        ys = []
        vals = []
        for ix in range(H.shape[0]):
            for iy in range(H.shape[1]):
                val = float(H[ix, iy])
                if val <= 0:
                    continue
                xs.append(
                    [
                        float(xedges[ix]),
                        float(xedges[ix + 1]),
                        float(xedges[ix + 1]),
                        float(xedges[ix]),
                    ]
                )
                ys.append(
                    [
                        float(yedges[iy]),
                        float(yedges[iy]),
                        float(yedges[iy + 1]),
                        float(yedges[iy + 1]),
                    ]
                )
                vals.append(val)

        if not vals:
            return None

        source = ColumnDataSource({"xs": xs, "ys": ys, "count": vals})
        palette = kwargs.pop("palette", palette_for_heatmap_cmap(cmap))
        low = float(vmin) if vmin is not None else float(np.nanmin(vals))
        high = float(vmax) if vmax is not None else float(np.nanmax(vals))
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        if high <= low:
            high = low + 1.0
        color_mapper = LinearColorMapper(palette=palette, low=low, high=high)
        line_alpha = kwargs.pop("line_alpha", 0.0)
        renderer = self.figure.patches(
            xs="xs",
            ys="ys",
            source=source,
            fill_color={"field": "count", "transform": color_mapper},
            fill_alpha=alpha,
            line_alpha=line_alpha,
            **kwargs,
        )
        if colorbar:
            remove_layout_annotation(self.figure, self._heatmap_colorbar)
            bar = ColorBar(
                color_mapper=color_mapper,
                title=colorbar_title,
                margin=10,
                padding=2,
            )
            self.figure.add_layout(bar, colorbar_location)
            self._heatmap_colorbar = bar
        return renderer


__all__ = ["CrossSection"]
