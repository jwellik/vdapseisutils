"""
Bokeh :class:`Map` — constructor and terrain aligned with ``core.maps.map.Map``.

Author: Jay Wellik
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from bokeh.layouts import row
from bokeh.models import (
    BoxZoomTool,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    MetricLength,
    ScaleBar,
    Title,
    WMTSTileSource,
)
from bokeh.models.annotations import Label
from bokeh.plotting import figure as bk_figure
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from pyproj import Transformer

from vdapseisutils.core.maps.bokeh.heatmap_common import (
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
    WORLD_LOCATION_MAP_DEFAULTS,
    default_volcano,
)
from vdapseisutils.core.maps.heatmap_utils import heatmap_bin_step, histogram_bin_edges
from vdapseisutils.core.maps.map_tiles import (
    ARCGIS_WORLD_HILLSHADE_URL,
    ATTRIBUTION_CARTO_POSITRON_NO_LABELS,
    ATTRIBUTION_ESRI_HILLSHADE,
    CARTO_LIGHT_NOLABELS_URL,
    GOOGLE_MAPS_TILE_ATTRIBUTION,
    _calculate_auto_zoom_arcgis,
    _calculate_auto_zoom_google,
    google_maps_xyz_url_template,
)
from vdapseisutils.core.maps.utils import choose_scale_bar_length, prep_catalog_data_mpl
from vdapseisutils.utils.geoutils import backazimuth, radial_extent2map_extent, sight_point_pyproj

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


def _mpl_fontsize_to_pt(size: Any) -> str:
    """Map matplotlib-style font sizes to Bokeh ``pt`` strings."""
    if isinstance(size, (int, float)):
        return f"{float(size)}pt"
    key = str(size).lower()
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
    return table.get(key, "14pt")


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


def _mpl_scatter_size_to_bokeh(size: Any) -> Any:
    """
    Convert matplotlib-style scatter area ``s`` (pt^2) into Bokeh marker ``size``.

    Matplotlib's ``scatter`` interprets ``s`` as marker area, while Bokeh expects a
    marker diameter-like size in screen units. Taking the square root keeps Bokeh
    glyphs visually closer to their matplotlib counterparts.
    """
    if size is None:
        return None
    arr = np.asarray(size, dtype=float)
    arr = np.sqrt(np.clip(arr, a_min=0.0, a_max=None))
    if np.ndim(arr) == 0:
        return float(arr)
    return arr


def _wmts_url_for_bokeh(url: str) -> str:
    """Bokeh ``WMTSTileSource`` expects ``{X}``, ``{Y}``, ``{Z}`` placeholders."""
    return url.replace("{z}", "{Z}").replace("{x}", "{X}").replace("{y}", "{Y}")


def _position_to_bokeh_location(position: str) -> str:
    """Translate MPL-ish strings like ``'lower right'`` to Bokeh locations."""
    mapping = {
        "lower right": "bottom_right",
        "lower left": "bottom_left",
        "lower center": "bottom_center",
        "upper right": "top_right",
        "upper left": "top_left",
        "upper center": "top_center",
        "center right": "center_right",
        "center left": "center_left",
        "center": "center",
    }
    return mapping.get(position, position.replace(" ", "_"))


def _stringify_hover_value(value: Any) -> str:
    """Format values for hover display without failing on missing/null inputs."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
            "match_aspect",
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
        match_aspect = bool(plot_kwargs.pop("match_aspect", True))

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
            fig_kwargs.setdefault("match_aspect", match_aspect)
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
            self.figure.match_aspect = match_aspect

        for tool in getattr(self.figure.toolbar, "tools", []):
            if isinstance(tool, BoxZoomTool):
                tool.match_aspect = match_aspect

        self._world_figure = None
        self._world_position: str | None = None
        self._scale_bar = None
        self._heatmap_colorbar: ColorBar | None = None

    @property
    def layout(self):
        """Bokeh layout: ``row`` of main map + world inset when used; else ``figure``."""
        if self._world_figure is not None and self._world_position is not None:
            if self._world_position == "upper right":
                return row(self.figure, self._world_figure, sizing_mode="stretch_width")
            return row(self._world_figure, self.figure, sizing_mode="stretch_width")
        return self.figure

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

    def _google_zoom_level(self, zoom: Any) -> int:
        radial = self.properties.get("radial_extent_km")
        if zoom == "auto":
            return (
                int(_calculate_auto_zoom_google(float(radial)))
                if radial is not None
                else 10
            )
        return int(zoom)

    def _add_google_tile_layer(self, style: str, zoom="auto", verbose=False, **kwargs: Any) -> None:
        """Attach one Google XYZ tile layer (Web Mercator); zoom hint affects logs only."""
        kwargs.pop("style", None)
        zoom_level = self._google_zoom_level(zoom)
        if verbose:
            print(f"Bokeh Google tiles ({style}): zoom_level={zoom_level}")
        url = _wmts_url_for_bokeh(google_maps_xyz_url_template(style))
        self.figure.add_tile(
            WMTSTileSource(url=url, attribution=GOOGLE_MAPS_TILE_ATTRIBUTION),
        )

    def add_google_terrain(
        self,
        zoom="auto",
        cache=False,
        verbose=False,
        ssl_verify=False,
        **kwargs,
    ) -> None:
        """Google terrain tiles (XYZ); matches matplotlib ``Map.add_google_terrain`` signature."""
        _ = cache
        _ = ssl_verify
        self._add_google_tile_layer("terrain", zoom=zoom, verbose=verbose, **kwargs)

    def add_google_street(
        self,
        zoom="auto",
        cache=False,
        verbose=False,
        ssl_verify=False,
        **kwargs,
    ) -> None:
        """Google street map tiles; matches matplotlib ``Map.add_google_street`` signature."""
        _ = cache
        _ = ssl_verify
        self._add_google_tile_layer("street", zoom=zoom, verbose=verbose, **kwargs)

    def add_google_satellite(
        self,
        zoom="auto",
        cache=False,
        verbose=False,
        ssl_verify=False,
        **kwargs,
    ) -> None:
        """Google satellite imagery tiles; matches matplotlib ``Map.add_google_satellite`` signature."""
        _ = cache
        _ = ssl_verify
        self._add_google_tile_layer("satellite", zoom=zoom, verbose=verbose, **kwargs)

    def set_title(self, title_text: str, **kwargs: Any):
        """Set the main plot title (MPL-compatible kwargs merged with ``TITLE_DEFAULTS``)."""
        title_params = {**TITLE_DEFAULTS, **kwargs}
        fs = _mpl_fontsize_to_pt(title_params["fontsize"])
        weight = title_params.get("fontweight", "bold")
        if weight == "bold":
            font_style = "bold"
        else:
            font_style = "normal"
        self.figure.title = Title(
            text=title_text,
            text_font_size=fs,
            text_font_style=font_style,
            text_color=title_params.get("color", "black"),
            align="center",
        )
        return self

    def set_subtitle(self, subtitle_text: str, **kwargs: Any):
        """Add a second title row above the map (Bokeh ``Title`` in ``above``)."""
        subtitle_params = {**SUBTITLE_DEFAULTS, **kwargs}
        fs = _mpl_fontsize_to_pt(subtitle_params["fontsize"])
        weight = subtitle_params.get("fontweight", "normal")
        font_style = "bold" if weight == "bold" else "normal"
        subt = Title(
            text=subtitle_text,
            text_font_size=fs,
            text_font_style=font_style,
            text_color=subtitle_params.get("color", "black"),
            align="center",
            standoff=2,
        )
        self.figure.add_layout(subt, "above")
        return self

    def set_catalog_subtitle(self, catalog, **kwargs: Any):
        """Subtitle from catalog summary (same as matplotlib ``Map``)."""
        from vdapseisutils.obspy_ext.catalog import VCatalog

        if not isinstance(catalog, VCatalog):
            vcatalog = VCatalog(catalog)
        else:
            vcatalog = catalog
        summary_str = vcatalog.short_summary_str()
        return self.set_subtitle(summary_str, **kwargs)

    def add_scalebar(
        self,
        scale_length_km="auto",
        position="lower right",
        color="black",
        fontsize=10,
        pad=0.5,
        frameon=False,
        **_ignored,
    ):
        """Add a corner-anchored scale bar, falling back to a manual Mercator bar if needed."""
        _ = pad
        extent = self.properties["map_extent"]
        map_lon_l, map_lon_r = extent[0], extent[1]
        map_mid_lat = (extent[2] + extent[3]) / 2.0
        _, d_m = backazimuth((map_mid_lat, map_lon_l), (map_mid_lat, map_lon_r))
        map_width_km = d_m / 1000.0

        bokeh_location = _position_to_bokeh_location(position)
        try:
            if self._scale_bar is not None:
                self.figure.center = [
                    item for item in self.figure.center if item is not self._scale_bar
                ]
            sb_kwargs = dict(
                range=self.figure.x_range,
                unit="m",
                dimensional=MetricLength(),
                orientation="horizontal",
                location=bokeh_location,
                label_location="above",
                label_align="center",
                bar_line_width=3,
                bar_line_color=_normalize_mpl_color(color),
                label_text_color=_normalize_mpl_color(color),
                label_text_font_size=f"{fontsize}pt",
                background_fill_alpha=0.0 if not frameon else 0.8,
                border_line_alpha=0.0 if not frameon else 1.0,
            )
            if scale_length_km == "auto":
                sb_kwargs["length_sizing"] = "adaptive"
                sb_kwargs["label"] = "@{value} @{unit}"
            else:
                if scale_length_km < 1:
                    scale_length_m = int(scale_length_km * 1000)
                    scale_label = f"{scale_length_m} m"
                elif scale_length_km == int(scale_length_km):
                    scale_label = f"{int(scale_length_km)} km"
                else:
                    scale_label = f"{scale_length_km} km"
                sb_kwargs["bar_length"] = float(scale_length_km) * 1000.0
                sb_kwargs["label"] = scale_label
            scale_bar = ScaleBar(
                **sb_kwargs,
            )
            self.figure.add_layout(scale_bar)
            self._scale_bar = scale_bar
            return self
        except Exception:
            pass

        if scale_length_km == "auto":
            scale_length_km = choose_scale_bar_length(map_width_km, 0.25)
        if scale_length_km < 1:
            scale_length_m = int(scale_length_km * 1000)
            scale_label = f"{scale_length_m} m"
        elif scale_length_km == int(scale_length_km):
            scale_label = f"{int(scale_length_km)} km"
        else:
            scale_label = f"{scale_length_km} km"

        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        lon_c = (map_lon_l + map_lon_r) / 2.0
        lon_e, lat_e, _ = geod.fwd(lon_c, map_mid_lat, 90.0, float(scale_length_km) * 1000.0)
        x_start, _ = _lonlat_to_mercator(lon_c, map_mid_lat)
        x_end, _ = _lonlat_to_mercator(lon_e, lat_e)
        bar_dx = abs(x_end - x_start)

        xr = self.figure.x_range
        yr = self.figure.y_range
        span_x = float(xr.end) - float(xr.start)
        span_y = float(yr.end) - float(yr.start)
        pad_x = pad * span_x * 0.08 if pad <= 1 else pad
        pad_y = pad * span_y * 0.08 if pad <= 1 else pad

        if "left" in position:
            x_left = float(xr.start) + pad_x
        elif position == "lower center":
            x_left = float(xr.start) + span_x / 2.0 - bar_dx / 2.0
        else:
            x_left = float(xr.end) - pad_x - bar_dx

        y_line = float(yr.start) + pad_y
        self.figure.segment(
            x0=[x_left],
            y0=[y_line],
            x1=[x_left + bar_dx],
            y1=[y_line],
            line_color=_normalize_mpl_color(color),
            line_width=3,
        )
        cx = x_left + bar_dx / 2.0
        off_y = span_y * 0.012
        self.figure.add_layout(
            Label(
                x=cx,
                y=y_line + off_y,
                text=scale_label,
                text_align="center",
                text_baseline="bottom",
                text_font_size=f"{fontsize}pt",
                text_color=_normalize_mpl_color(color),
            )
        )
        return self

    def _add_hover_tool(
        self,
        renderer,
        tooltips=None,
        formatters=None,
    ):
        """Attach a renderer-specific hover tool when tooltips are provided."""
        if not tooltips:
            return renderer
        hover = HoverTool(
            renderers=[renderer],
            tooltips=tooltips,
            formatters=formatters or {},
        )
        self.figure.add_tools(hover)
        return renderer

    def add_world_location_map(
        self,
        size=0.18,
        position="upper left",
        **kwargs,
    ):
        """
        Small world locator map beside the main map (``layout`` becomes a ``row``).

        Uses Web Mercator tiles plus a marker at the main map centre / ``origin``.
        """
        style_params = {**WORLD_LOCATION_MAP_DEFAULTS, **kwargs}

        map_extent = self.properties["map_extent"]
        center_lon = (map_extent[0] + map_extent[1]) / 2.0
        center_lat = (map_extent[2] + map_extent[3]) / 2.0
        origin = self.properties.get("origin")
        if origin is not None:
            main_lat, main_lon = origin
        else:
            main_lat, main_lon = center_lat, center_lon

        w_main = getattr(self.figure, "width", 600) or 600
        h_main = getattr(self.figure, "height", 400) or 400
        inset_w = max(int(w_main * float(size)), 120)
        inset_h = max(int(h_main * float(size)), 120)

        world_extent = [-180.0, 180.0, -65.0, 65.0]
        xr_w, yr_w = _extent_lonlat_to_mercator_ranges(world_extent)

        wf = bk_figure(
            x_range=xr_w,
            y_range=yr_w,
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=inset_w,
            height=inset_h,
            toolbar_location=None,
            outline_line_color="gray",
            background_fill_color=style_params.get("ocean_color", "lightgrey"),
            border_fill_color=style_params.get("ocean_color", "lightgrey"),
        )
        wf.axis.visible = False
        wf.grid.visible = False

        carto = WMTSTileSource(
            url=_wmts_url_for_bokeh(CARTO_LIGHT_NOLABELS_URL),
            attribution="",
        )
        wf.add_tile(carto, alpha=style_params.get("ocean_alpha", 0.8))

        mx, my = _lonlat_to_mercator(main_lon, main_lat)
        marker = style_params.get("marker_style", "square")
        wf.scatter(
            x=[mx],
            y=[my],
            marker=_bokeh_marker(marker),
            size=int(style_params.get("marker_size", 6)) + 4,
            fill_color=style_params.get("marker_color", "black"),
            line_color="black",
        )

        self._world_figure = wf
        self._world_position = position
        return wf

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
        hover_text = scatter_kwargs.pop("hover_text", None)
        hover_tooltips = scatter_kwargs.pop("hover_tooltips", None)
        hover_formatters = scatter_kwargs.pop("hover_formatters", None)
        c_alias = scatter_kwargs.pop("c", None)
        s_alias = scatter_kwargs.pop("s", None)
        if color is None and c_alias is not None:
            color = c_alias
        if size is None and s_alias is not None:
            size = s_alias
        if "edgecolors" in scatter_kwargs and "line_color" not in scatter_kwargs:
            scatter_kwargs["line_color"] = scatter_kwargs.pop("edgecolors")
        if "linewidths" in scatter_kwargs and "line_width" not in scatter_kwargs:
            scatter_kwargs["line_width"] = scatter_kwargs.pop("linewidths")
        if color is not None and "color" not in scatter_kwargs:
            scatter_kwargs["color"] = _normalize_mpl_color(color)
        if size is not None and "size" not in scatter_kwargs:
            scatter_kwargs["size"] = _mpl_scatter_size_to_bokeh(size)
        if "line_color" in scatter_kwargs:
            scatter_kwargs["line_color"] = _normalize_mpl_color(scatter_kwargs["line_color"])
        if hover_text is not None and "source" not in scatter_kwargs:
            x_arr = np.atleast_1d(x)
            y_arr = np.atleast_1d(y)
            hover_arr = np.atleast_1d(hover_text).astype(str)
            if hover_arr.size == 1 and x_arr.size > 1:
                hover_arr = np.repeat(hover_arr, x_arr.size)
            source_data = {"x": x_arr, "y": y_arr, "hover_text": hover_arr}
            size_val = scatter_kwargs.get("size")
            if isinstance(size_val, (list, tuple, np.ndarray)):
                size_arr = np.atleast_1d(np.asarray(size_val, dtype=float))
                if size_arr.size == 1 and x_arr.size > 1:
                    size_arr = np.repeat(size_arr, x_arr.size)
                source_data["size"] = size_arr
                scatter_kwargs["size"] = "size"
            source = ColumnDataSource(source_data)
            scatter_kwargs["source"] = source
            x = "x"
            y = "y"
            if hover_tooltips is None:
                hover_tooltips = [("label", "@hover_text")]
        renderer = self.figure.scatter(x=x, y=y, **scatter_kwargs)
        return self._add_hover_tool(renderer, hover_tooltips, hover_formatters)

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
        """Plot a map density heatmap (catalog or lat/lon arrays), mirroring MPL usage."""
        try:
            if len(args) == 1 and hasattr(args[0], "events"):
                catdata = prep_catalog_data_mpl(args[0], time_format="matplotlib")
                lat = np.asarray(catdata["lat"].values, dtype=float)
                lon = np.asarray(catdata["lon"].values, dtype=float)
            elif len(args) >= 2:
                lat = np.asarray(args[0], dtype=float)
                lon = np.asarray(args[1], dtype=float)
            else:
                raise ValueError(
                    "Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)"
                )

            if lat.size == 0 or lon.size == 0:
                return None

            finite_mask = np.isfinite(lat) & np.isfinite(lon)
            if not np.any(finite_mask):
                return None
            lat = lat[finite_mask]
            lon = lon[finite_mask]

            lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
            lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
            if lon_max <= lon_min or lat_max <= lat_min:
                return None

            data_range_lon = lon_max - lon_min
            data_range_lat = lat_max - lat_min
            extent_short = min(data_range_lon, data_range_lat)
            colorbar = kwargs.pop("colorbar", True)
            colorbar_title = kwargs.pop("colorbar_title", "Event count")
            colorbar_location = kwargs.pop("colorbar_location", "right")
            max_heatmap_bins = int(kwargs.pop("max_heatmap_bins", 120))
            grid_size_deg = heatmap_bin_step(
                extent_short, float(grid_size), floor=1e-5, max_bins=max_heatmap_bins
            )

            lon_pad = data_range_lon * 0.1
            lat_pad = data_range_lat * 0.1
            lon_grid = histogram_bin_edges(
                lon_min - lon_pad, lon_max + lon_pad, grid_size_deg
            )
            lat_grid = histogram_bin_edges(
                lat_min - lat_pad, lat_max + lat_pad, grid_size_deg
            )
            if lon_grid.size < 2 or lat_grid.size < 2:
                return None

            hist, lon_edges, lat_edges = np.histogram2d(lon, lat, bins=[lon_grid, lat_grid])
            if hist.size == 0 or np.all(hist == 0):
                return None

            left: list[float] = []
            right: list[float] = []
            bottom: list[float] = []
            top: list[float] = []
            counts: list[float] = []
            for i_lon in range(hist.shape[0]):
                for i_lat in range(hist.shape[1]):
                    val = float(hist[i_lon, i_lat])
                    if val <= 0:
                        continue
                    x0, y0 = _lonlat_to_mercator(lon_edges[i_lon], lat_edges[i_lat])
                    x1, y1 = _lonlat_to_mercator(
                        lon_edges[i_lon + 1], lat_edges[i_lat + 1]
                    )
                    left.append(float(min(x0, x1)))
                    right.append(float(max(x0, x1)))
                    bottom.append(float(min(y0, y1)))
                    top.append(float(max(y0, y1)))
                    counts.append(val)

            if not counts:
                return None

            source = ColumnDataSource(
                {
                    "left": left,
                    "right": right,
                    "bottom": bottom,
                    "top": top,
                    "count": counts,
                }
            )
            palette = palette_for_heatmap_cmap(cmap)
            low = float(vmin) if vmin is not None else float(np.nanmin(counts))
            high = float(vmax) if vmax is not None else float(np.nanmax(counts))
            if not np.isfinite(low) or not np.isfinite(high):
                return None
            if high <= low:
                high = low + 1.0
            color_mapper = LinearColorMapper(palette=palette, low=low, high=high)
            line_alpha = kwargs.pop("line_alpha", 0.0)
            renderer = self.figure.quad(
                left="left",
                right="right",
                bottom="bottom",
                top="top",
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
        except Exception:
            return None

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
        _ = cmap
        _ = transform
        hover_tooltips = kwargs.pop("hover_tooltips", None)
        hover_formatters = kwargs.pop("hover_formatters", None)
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
        size_bokeh = _mpl_scatter_size_to_bokeh(s)

        # Numeric color arrays use linear color mapping in Bokeh.
        c_array = np.asarray(c) if hasattr(c, "__len__") and not isinstance(c, str) else None
        source_data = {
            "x": x,
            "y": y,
            "size": size_bokeh,
            "time_str": np.asarray([str(t) for t in catdata["time"]]),
            "mag_str": np.asarray([_stringify_hover_value(v) for v in catdata["mag"]]),
            "depth_str": np.asarray([_stringify_hover_value(v) for v in catdata["depth"]]),
            "lat_str": np.asarray([_stringify_hover_value(v) for v in catdata["lat"]]),
            "lon_str": np.asarray([_stringify_hover_value(v) for v in catdata["lon"]]),
        }
        if hover_tooltips is None:
            hover_tooltips = [
                ("time", "@time_str"),
                ("mag", "@mag_str"),
                ("depth (km)", "@depth_str"),
                ("lat", "@lat_str"),
                ("lon", "@lon_str"),
            ]
        if c_array is not None and c_array.size == len(x):
            c_min = float(np.nanmin(c_array))
            c_max = float(np.nanmax(c_array))
            mapper = linear_cmap("cval", Viridis256, low=c_min, high=c_max)
            source_data["cval"] = c_array
            source = ColumnDataSource(source_data)
            renderer = self.figure.scatter(
                x="x",
                y="y",
                size="size",
                source=source,
                fill_color=mapper,
                line_color=mapper,
                **plot_kwargs,
            )
            return self._add_hover_tool(renderer, hover_tooltips, hover_formatters)

        source = ColumnDataSource(source_data)
        renderer = self.figure.scatter(
            x="x",
            y="y",
            size="size",
            color=_normalize_mpl_color(c),
            source=source,
            **plot_kwargs,
        )
        return self._add_hover_tool(renderer, hover_tooltips, hover_formatters)

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
        hover_tooltips = kwargs.pop("hover_tooltips", None)
        hover_formatters = kwargs.pop("hover_formatters", None)
        station_lats = []
        station_lons = []
        network_codes = []
        station_codes = []
        elevations = []
        for network in inventory:
            for station in network:
                if hasattr(station, "latitude") and hasattr(station, "longitude"):
                    station_lats.append(station.latitude)
                    station_lons.append(station.longitude)
                    network_codes.append(getattr(network, "code", "") or "")
                    station_codes.append(getattr(station, "code", "") or "")
                    elevations.append(getattr(station, "elevation", None))
        if not station_lats:
            print("No valid station coordinates found in inventory")
            return None
        plot_kwargs = {**PLOT_INVENTORY_DEFAULTS, **kwargs}
        plot_kwargs.update(
            {"size": _mpl_scatter_size_to_bokeh(s), "color": c, "alpha": alpha}
        )
        plot_kwargs.pop("s", None)
        plot_kwargs.pop("c", None)
        if "edgecolors" in plot_kwargs:
            plot_kwargs["line_color"] = plot_kwargs.pop("edgecolors")
        mk = plot_kwargs.pop("marker", None)
        if mk:
            plot_kwargs["marker"] = _bokeh_marker(mk) or mk
        x, y = _lonlat_to_mercator(station_lons, station_lats)
        source = ColumnDataSource(
            {
                "x": x,
                "y": y,
                "network": np.asarray(network_codes, dtype=str),
                "station": np.asarray(station_codes, dtype=str),
                "station_id": np.asarray(
                    [f"{n}.{s}" if n else s for n, s in zip(network_codes, station_codes)],
                    dtype=str,
                ),
                "lat_str": np.asarray([_stringify_hover_value(v) for v in station_lats]),
                "lon_str": np.asarray([_stringify_hover_value(v) for v in station_lons]),
                "elev_str": np.asarray([_stringify_hover_value(v) for v in elevations]),
            }
        )
        if hover_tooltips is None:
            hover_tooltips = [
                ("station", "@station_id"),
                ("lat", "@lat_str"),
                ("lon", "@lon_str"),
                ("elev (m)", "@elev_str"),
            ]
        renderer = self.figure.scatter(x="x", y="y", source=source, **plot_kwargs)
        return self._add_hover_tool(renderer, hover_tooltips, hover_formatters)

    def plot_volcano(self, lat, lon, elev=0, transform=None, **kwargs):
        """Plot volcano location on the map."""
        _ = elev
        _ = transform
        plot_kwargs = {**PLOT_VOLCANO_DEFAULTS, **kwargs}
        return self.scatter(
            lat,
            lon,
            plot_kwargs.pop("s", 64),
            plot_kwargs.pop("c", "orangered"),
            **plot_kwargs,
        )

    def plot_peak(self, lat, lon, elev=0, transform=None, **kwargs):
        """Plot peak location on the map."""
        _ = elev
        _ = transform
        plot_kwargs = {**PLOT_PEAK_DEFAULTS, **kwargs}
        return self.scatter(
            lat,
            lon,
            plot_kwargs.pop("s", 64),
            plot_kwargs.pop("c", "floralwhite"),
            **plot_kwargs,
        )

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

    def plot_cross_section(
        self,
        points=None,
        origin=None,
        radius_km=40.0,
        azimuth=270.0,
        label="A",
        color="k",
        linewidth=1.5,
        va="center",
        ha="center",
        transform=None,
        **kwargs,
    ):
        """
        Plot a cross-section/transect line and endpoint labels (A and A').

        Provide either ``points=[(lat1, lon1), (lat2, lon2)]`` or ``origin`` with
        ``radius_km`` + ``azimuth``.
        """
        _ = va
        _ = transform
        if points is None:
            if origin is None:
                raise ValueError("Provide either points or origin.")
            radius_m = float(radius_km) * 1000.0
            p1 = sight_point_pyproj(origin, azimuth, radius_m)
            p2 = sight_point_pyproj(origin, np.mod(azimuth + 180.0, 360.0), radius_m)
        else:
            if len(points) != 2:
                raise ValueError("points must contain exactly two (lat, lon) tuples.")
            p1, p2 = points
        self.plot_line(
            p1,
            p2,
            color=color,
            linewidth=linewidth,
            label=label,
            ha=ha,
            **kwargs,
        )
        return p1, p2

    def plot_transect(self, *args, **kwargs):
        """Alias for :meth:`plot_cross_section`."""
        return self.plot_cross_section(*args, **kwargs)
