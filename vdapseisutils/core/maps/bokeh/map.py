"""
Bokeh :class:`Map` — constructor and terrain aligned with ``core.maps.map.Map``.

Author: Jay Wellik
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from bokeh.models import WMTSTileSource
from bokeh.plotting import figure as bk_figure
from pyproj import Transformer

from vdapseisutils.core.maps.defaults import default_volcano
from vdapseisutils.core.maps.map_tiles import (
    ARCGIS_WORLD_HILLSHADE_URL,
    ATTRIBUTION_CARTO_POSITRON_NO_LABELS,
    ATTRIBUTION_ESRI_HILLSHADE,
    CARTO_LIGHT_NOLABELS_URL,
    _calculate_auto_zoom_arcgis,
)
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
