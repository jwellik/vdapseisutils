"""
Bokeh :class:`Map` — constructor and terrain aligned with ``core.maps.map.Map``.

Author: Jay Wellik
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from bokeh.models import ColumnDataSource, WMTSTileSource
from bokeh.models.annotations import Label
from bokeh.plotting import figure as bk_figure
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from pyproj import Transformer

from vdapseisutils.core.maps.defaults import (
    PLOT_CATALOG_DEFAULTS,
    PLOT_INVENTORY_DEFAULTS,
    PLOT_PEAK_DEFAULTS,
    PLOT_VOLCANO_DEFAULTS,
    default_volcano,
)
from vdapseisutils.core.maps.map_tiles import (
    ARCGIS_WORLD_HILLSHADE_URL,
    ATTRIBUTION_CARTO_POSITRON_NO_LABELS,
    ATTRIBUTION_ESRI_HILLSHADE,
    CARTO_LIGHT_NOLABELS_URL,
    _calculate_auto_zoom_arcgis,
)
from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
from vdapseisutils.utils.geoutils import radial_extent2map_extent

# PROJ4 strings avoid EPSG lookups so imports work when ``proj.db`` is missing or
# misconfigured (CRSError: no database context specified).
_WGS84_LONG_LAT = "+proj=longlat +datum=WGS84 +no_defs"
_WEB_MERCATOR = (
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +k=1.0 "
    "+x_0=0.0 +y_0=0 +units=m +no_defs"
)


@lru_cache(maxsize=1)
def _wgs84_to_web_mercator_transformer() -> Transformer:
    return Transformer.from_crs(
        _WGS84_LONG_LAT, _WEB_MERCATOR, always_xy=True
    )


def _extent_lonlat_to_mercator_ranges(
    extent: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Convert ``[min_lon, max_lon, min_lat, max_lat]`` to Web Mercator x/y ranges."""
    min_lon, max_lon, min_lat, max_lat = extent
    xs: list[float] = []
    ys: list[float] = []
    tr = _wgs84_to_web_mercator_transformer()
    for lon, lat in (
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
    ):
        x, y = tr.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return (min(xs), max(xs)), (min(ys), max(ys))


def _lonlat_to_mercator(lon, lat):
    """Convert lon/lat scalars or arrays to Web Mercator x/y."""
    tr = _wgs84_to_web_mercator_transformer()
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    return tr.transform(lon_arr, lat_arr)


def _mpl_fmt_to_bokeh(fmt: str) -> dict[str, Any]:
    """Translate a minimal matplotlib format string to bokeh style kwargs."""
    style: dict[str, Any] = {"line_dash": "solid", "marker": None, "color": None}
    color_map = {
        "b": "blue",
        "g": "green",
        "r": "red",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
    }
    marker_choices = ["^", "o", "s", "d", "x", "+", "*", "v", "<", ">"]
    for marker in marker_choices:
        if marker in fmt:
            style["marker"] = marker
            break
    if "--" in fmt:
        style["line_dash"] = "dashed"
    elif "-." in fmt:
        style["line_dash"] = "dashdot"
    elif ":" in fmt:
        style["line_dash"] = "dotted"
    for key, val in color_map.items():
        if key in fmt:
            style["color"] = val
            break
    return style


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


def _wmts_url_for_bokeh(url: str) -> str:
    """Bokeh ``WMTSTileSource`` expects ``{X}``, ``{Y}``, ``{Z}`` placeholders."""
    return url.replace("{z}", "{Z}").replace("{x}", "{X}").replace("{y}", "{Y}")


def _parse_figure_dimensions(kwargs: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    """Resolve ``figsize`` / ``dpi`` / ``width`` / ``height`` like a matplotlib figure."""
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
        width = 600
    if height is None:
        height = 600
    return width, height, kw


class Map:
    """
    Bokeh-backed geographic map using Web Mercator tiles and axes.

    Constructor arguments match :class:`vdapseisutils.core.maps.map.Map`; pass
    ``figsize`` or ``width`` / ``height`` for pixel dimensions (``figsize`` uses
    ``dpi``, default 100, to convert inches to pixels).

    Example::

        from vdapseisutils.core.maps.bokeh import Map

        m = Map(origin=(59.36, -153.43), radial_extent_km=30)
        m.add_terrain()
        show(m.figure)
    """

    name = "map"

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
        origin=(default_volcano["lat"], default_volcano["lon"]),
        radial_extent_km: float = 50.0,
        map_extent=None,
        **kwargs: Any,
    ) -> None:
        width, height, plot_kwargs = _parse_figure_dimensions(kwargs)

        self.properties: dict[str, Any] = {}
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(
            origin[0], origin[1], radial_extent_km
        )
        if map_extent is not None:
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent

        if fig is None:
            x_range, y_range = _extent_lonlat_to_mercator_ranges(
                self.properties["map_extent"]
            )
            fig_kwargs = {k: v for k, v in plot_kwargs.items() if k in self._FIGURE_KW}
            self.figure = bk_figure(
                x_range=x_range,
                y_range=y_range,
                x_axis_type="mercator",
                y_axis_type="mercator",
                width=width,
                height=height,
                **fig_kwargs,
            )
        else:
            self.figure = fig

    def info(self) -> None:
        """Print map properties (same idea as the matplotlib :class:`~vdapseisutils.core.maps.map.Map`)."""
        print("::: BOKEH MAP :::")
        print(self.properties)
        print()

    def add_terrain(
        self,
        zoom="auto",
        cache=False,
        verbose=False,
        ssl_verify=False,
        **kwargs,
    ) -> None:
        """
        Add default terrain: Esri World Hillshade plus Carto light nolabels overlay.

        Matches :meth:`vdapseisutils.core.maps.map.Map.add_terrain` / ``add_arcgis_terrain``.
        ``cache`` and ``ssl_verify`` apply to the Cartopy matplotlib path only; Bokeh tiles
        are loaded by the browser and ignore these flags for now.

        Parameters
        ----------
        zoom : int or ``'auto'``
            Tile zoom level or automatic choice from ``radial_extent_km`` when set.
        cache, verbose, ssl_verify
            Accepted for API compatibility with the matplotlib map.
        **kwargs
            Swallowed for compatibility (e.g. ``style`` forwarded from wrappers).
        """
        _ = cache
        _ = ssl_verify
        _ = kwargs.pop("style", None)
        if kwargs:
            pass

        radial = self.properties.get("radial_extent_km")
        if zoom == "auto":
            zoom_level = (
                _calculate_auto_zoom_arcgis(radial) if radial is not None else 10
            )
        else:
            zoom_level = int(zoom)

        if verbose:
            print(f"Bokeh add_terrain: zoom_level={zoom_level}")

        base = WMTSTileSource(
            url=_wmts_url_for_bokeh(ARCGIS_WORLD_HILLSHADE_URL),
            attribution=ATTRIBUTION_ESRI_HILLSHADE,
        )
        overlay = WMTSTileSource(
            url=_wmts_url_for_bokeh(CARTO_LIGHT_NOLABELS_URL),
            attribution=ATTRIBUTION_CARTO_POSITRON_NO_LABELS,
        )
        self.figure.add_tile(base)
        self.figure.add_tile(overlay, alpha=0.5)

    def plot(self, lat, lon, *args, transform=None, **kwargs):
        """Plot line data on the map (matplotlib-like signature)."""
        _ = transform
        style = {}
        if args and isinstance(args[0], str):
            style = _mpl_fmt_to_bokeh(args[0])
        x, y = _lonlat_to_mercator(lon, lat)
        line_kwargs = dict(kwargs)
        if style.get("color") and "line_color" not in line_kwargs and "color" not in line_kwargs:
            line_kwargs["line_color"] = _normalize_mpl_color(style["color"])
        if style.get("line_dash"):
            line_kwargs.setdefault("line_dash", style["line_dash"])
        line_renderer = self.figure.line(x=x, y=y, **line_kwargs)
        renderers = [line_renderer]
        if style.get("marker"):
            marker_kwargs = {}
            if style.get("color"):
                marker_kwargs["fill_color"] = _normalize_mpl_color(style["color"])
                marker_kwargs["line_color"] = _normalize_mpl_color(style["color"])
            marker_kwargs.update({k: v for k, v in kwargs.items() if k in ("size", "alpha")})
            renderers.append(
                self.figure.scatter(x=x, y=y, marker=style["marker"], **marker_kwargs)
            )
        return renderers

    def scatter(self, lat, lon, size, color, transform=None, **kwargs):
        """Plot scatter data on the map (matplotlib-like signature)."""
        _ = transform
        x, y = _lonlat_to_mercator(lon, lat)
        scatter_kwargs = dict(kwargs)
        if "c" in scatter_kwargs and color is None:
            color = scatter_kwargs.pop("c")
        if "s" in scatter_kwargs and size is None:
            size = scatter_kwargs.pop("s")
        if "edgecolors" in scatter_kwargs and "line_color" not in scatter_kwargs:
            scatter_kwargs["line_color"] = scatter_kwargs.pop("edgecolors")
        if "linewidths" in scatter_kwargs and "line_width" not in scatter_kwargs:
            scatter_kwargs["line_width"] = scatter_kwargs.pop("linewidths")
        if color is not None and "color" not in scatter_kwargs:
            scatter_kwargs["color"] = _normalize_mpl_color(color)
        if size is not None and "size" not in scatter_kwargs:
            scatter_kwargs["size"] = size
        return self.figure.scatter(x=x, y=y, **scatter_kwargs)

    def plot_catalog(
        self,
        catalog,
        s=PLOT_CATALOG_DEFAULTS["s"],
        c=PLOT_CATALOG_DEFAULTS["c"],
        color=PLOT_CATALOG_DEFAULTS["color"],
        cmap=PLOT_CATALOG_DEFAULTS["cmap"],
        alpha=PLOT_CATALOG_DEFAULTS["alpha"],
        transform=None,
        **kwargs,
    ):
        """Plot earthquake catalog on the map."""
        _ = transform
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
        if s == "magnitude":
            s = catdata["size"]
        if color is not None:
            c = color
        elif c == "time":
            c = catdata["time"]

        x, y = _lonlat_to_mercator(catdata["lon"].values, catdata["lat"].values)
        plot_kwargs = dict(kwargs)
        plot_kwargs.setdefault("alpha", alpha)

        # Numeric color arrays use linear color mapping in Bokeh.
        c_array = np.asarray(c) if hasattr(c, "__len__") and not isinstance(c, str) else None
        if c_array is not None and c_array.size == len(x):
            c_min = float(np.nanmin(c_array))
            c_max = float(np.nanmax(c_array))
            mapper = linear_cmap("cval", Viridis256, low=c_min, high=c_max)
            source = ColumnDataSource({"x": x, "y": y, "cval": c_array, "size": s})
            return self.figure.scatter(
                x="x",
                y="y",
                size="size",
                source=source,
                fill_color=mapper,
                line_color=mapper,
                **plot_kwargs,
            )

        return self.figure.scatter(
            x=x, y=y, size=s, color=_normalize_mpl_color(c), **plot_kwargs
        )

    def plot_inventory(
        self,
        inventory,
        s=PLOT_INVENTORY_DEFAULTS["s"],
        c=PLOT_INVENTORY_DEFAULTS["c"],
        alpha=PLOT_INVENTORY_DEFAULTS["alpha"],
        transform=None,
        **kwargs,
    ):
        """Plot seismic station inventory on the map."""
        _ = transform
        station_lats = []
        station_lons = []
        for network in inventory:
            for station in network:
                if hasattr(station, "latitude") and hasattr(station, "longitude"):
                    station_lats.append(station.latitude)
                    station_lons.append(station.longitude)
        if not station_lats:
            print("No valid station coordinates found in inventory")
            return None
        plot_kwargs = {**PLOT_INVENTORY_DEFAULTS, **kwargs}
        plot_kwargs.update({"size": s, "color": c, "alpha": alpha})
        plot_kwargs.pop("s", None)
        plot_kwargs.pop("c", None)
        if "edgecolors" in plot_kwargs:
            plot_kwargs["line_color"] = plot_kwargs.pop("edgecolors")
        x, y = _lonlat_to_mercator(station_lons, station_lats)
        return self.figure.scatter(x=x, y=y, **plot_kwargs)

    def plot_volcano(self, lat, lon, elev=0, transform=None, **kwargs):
        """Plot volcano location on the map."""
        _ = elev
        _ = transform
        plot_kwargs = {**PLOT_VOLCANO_DEFAULTS, **kwargs}
        return self.scatter(lat, lon, plot_kwargs.pop("s", 64), plot_kwargs.pop("c", "orangered"), **plot_kwargs)

    def plot_peak(self, lat, lon, elev=0, transform=None, **kwargs):
        """Plot peak location on the map."""
        _ = elev
        _ = transform
        plot_kwargs = {**PLOT_PEAK_DEFAULTS, **kwargs}
        return self.scatter(lat, lon, plot_kwargs.pop("s", 64), plot_kwargs.pop("c", "floralwhite"), **plot_kwargs)

    def plot_line(
        self,
        p1,
        p2,
        color="k",
        linewidth=1,
        label=None,
        va="center",
        ha="center",
        transform=None,
        **kwargs,
    ):
        """Plot a line between two points with optional endpoint labels."""
        _ = transform
        x, y = _lonlat_to_mercator([p1[1], p2[1]], [p1[0], p2[0]])
        renderer = self.figure.line(
            x=x, y=y, line_color=_normalize_mpl_color(color), line_width=linewidth, **kwargs
        )
        if label:
            self.figure.add_layout(Label(x=x[0], y=y[0], text=label, text_align=ha))
            self.figure.add_layout(Label(x=x[1], y=y[1], text=f"{label}'", text_align=ha))
        return renderer
