"""
Matplotlib-free cross-section **data** for a future Plotly view (Part 1 of 3).

**Scope:** geometry, :class:`~vdapseisutils.core.maps.elev_profile.TopographicProfile`,
axis extents in km, and catalog/inventory tables projected into cross-section coordinates.
No ``Figure`` / ``Axes`` and no Plotly traces yet.

Downstream ``CrossSection`` (matplotlib) public surface relied on by
:class:`~vdapseisutils.core.maps.volcano_figure.VolcanoFigure` and
``examples/cross_section_standalone.py`` (for Plotly parity later):

- **Constructor kwargs:** ``fig``, ``points``, ``origin``, ``radius_km``, ``azimuth``,
  ``map_extent``, ``depth_extent``, ``resolution``, ``max_n``, ``label``, ``width``,
  ``maglegend``, ``verbose``, plus ``dpi`` / ``figsize`` (figure only).
- **Attributes:** ``figure``, ``ax``, ``properties`` (dict), ``A1`` / ``A2`` (endpoints),
  ``profile`` (:class:`~vdapseisutils.core.maps.elev_profile.TopographicProfile`),
  ``name`` (``\"cross-section\"``).
- **Methods used in layouts / examples:** ``plot``, ``scatter``, ``plot_catalog``,
  ``plot_inventory``, ``plot_volcano``, ``plot_peak``, ``plot_heatmap``, ``set_title``,
  ``set_subtitle``, ``set_titles``, ``set_catalog_subtitle``, ``set_depth_extent``,
  ``set_horiz_extent``.

VolcanoFigure specifically reads ``xs*_obj.properties[\"points\"]`` and
``xs*_obj.properties[\"label\"]`` for map section lines, and calls ``plot_catalog``,
``plot_inventory``, ``plot_volcano``, ``plot_peak``, ``scatter``, and ``plot`` on each
``CrossSection`` instance.

Author: Jay Wellik (Plotly port scaffolding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from vdapseisutils.core.maps import elev_profile
from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
from vdapseisutils.utils.geoutils import backazimuth, project2line, sight_point_pyproj


def _z_to_axes_km(
    z: np.ndarray | Sequence[float],
    *,
    z_dir: str,
    z_unit: str,
) -> np.ndarray:
    """Match :meth:`~vdapseisutils.core.maps.cross_section.CrossSection.scatter` depth conversion."""
    depth = np.asarray(z, dtype=float)
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


@dataclass(frozen=True)
class CrossSectionData:
    """Immutable bundle of cross-section inputs shared with matplotlib ``CrossSection``."""

    properties: dict[str, Any]
    A1: tuple[float, float]
    A2: tuple[float, float]
    label: str
    depth_extent: tuple[float, float]
    depth_range: float
    profile: elev_profile.TopographicProfile
    profile_distance_km: np.ndarray
    profile_elevation_km: np.ndarray
    horiz_extent_km: tuple[float, float]
    has_profile: bool = field(compare=False)


def build_cross_section_data(
    *,
    points: Sequence[tuple[float, float]] | None = None,
    origin: tuple[float, float] | None = None,
    radius_km: float = 25.0,
    azimuth: float = 270,
    map_extent: Sequence[float] | None = None,
    depth_extent: tuple[float, float] = (-50.0, 4.0),
    resolution: str | float = "auto",
    max_n: int = 100,
    label: str = "A",
    width: float | None = None,
    verbose: bool = False,
    profile_source: str = "opentopo",
) -> CrossSectionData:
    """
    Compute cross-section geometry and topography profile without Matplotlib.

    Mirrors the logic in :class:`~vdapseisutils.core.maps.cross_section.CrossSection`
    ``__init__`` (figure/axes excluded). Horizontal axis is distance in km from ``A1``
    toward ``A2`` (same convention as ``project2line(..., unit='km')``).
    """
    properties: dict[str, Any] = {}

    if origin is not None:
        properties["origin"] = origin
        properties["azimuth"] = azimuth
        properties["radius"] = radius_km * 1000.0
        p0 = sight_point_pyproj(origin, azimuth, properties["radius"])
        p1 = sight_point_pyproj(origin, np.mod(azimuth + 180, 360), properties["radius"])
        properties["points"] = [p0, p1]
        horiz_km = 2.0 * radius_km
    else:
        if points is None or len(points) != 2:
            raise ValueError(
                "ERROR: Points must be a list of 2 tuples of lat,lon coordinates."
            )
        properties["points"] = [points[0], points[1]]
        properties["origin"] = None
        properties["azimuth"], length_m = backazimuth(points[0], points[1])
        properties["length"] = length_m
        properties["radius"] = None
        horiz_km = float(length_m) / 1000.0

    properties["map_extent"] = map_extent
    properties["depth_extent"] = depth_extent
    properties["depth_range"] = depth_extent[1] - depth_extent[0]
    properties["label"] = label
    properties["full_label"] = "{a}-{a}'".format(a=properties["label"])
    properties["width"] = width
    properties["orientation"] = "horizontal"
    properties["verbose"] = verbose

    A1 = tuple(properties["points"][0])  # type: ignore[arg-type]
    A2 = tuple(properties["points"][1])  # type: ignore[arg-type]

    profile = elev_profile.TopographicProfile(
        [A1, A2], source=profile_source, resolution=resolution, max_n=max_n
    )
    if np.any(profile.elevation):
        dist_km = np.asarray(profile.distance, dtype=float) / 1000.0
        elev_km = np.asarray(profile.elevation, dtype=float) / 1000.0
        has_profile = True
    else:
        dist_km = np.array([], dtype=float)
        elev_km = np.array([], dtype=float)
        has_profile = False

    horiz_extent_km = (0.0, horiz_km)

    return CrossSectionData(
        properties=properties,
        A1=A1,
        A2=A2,
        label=label,
        depth_extent=tuple(depth_extent),
        depth_range=float(properties["depth_range"]),
        profile=profile,
        profile_distance_km=dist_km,
        profile_elevation_km=elev_km,
        horiz_extent_km=horiz_extent_km,
        has_profile=has_profile,
    )


def _along_line_km_1d(
    lat: np.ndarray | Sequence[float],
    lon: np.ndarray | Sequence[float],
    A1: tuple[float, float],
    A2: tuple[float, float],
) -> np.ndarray:
    """``project2line`` returns a scalar for one point; always return a 1D float array."""
    raw = project2line(lat, lon, P1=A1, P2=A2, unit="km")
    return np.atleast_1d(np.asarray(raw, dtype=float)).ravel()


def prep_catalog_for_cross_section(
    catalog: Any,
    A1: tuple[float, float],
    A2: tuple[float, float],
    *,
    time_format: str = "matplotlib",
    maglegend: Any = None,
) -> dict[str, Any]:
    """
    Run :func:`~vdapseisutils.core.maps.utils.prep_catalog_data_mpl` and add cross-section columns.

    Returns a dict including ``catdata`` (pandas DataFrame) and ``x_km``, ``z_axes_km``
    using the same rules as ``CrossSection.plot_catalog`` / ``scatter`` (elev, km).
    """
    catdata = prep_catalog_data_mpl(
        catalog, time_format=time_format, maglegend=maglegend
    )
    x_km = _along_line_km_1d(catdata["lat"], catdata["lon"], A1, A2)
    z_axes_km = _z_to_axes_km(
        catdata["depth"].values, z_dir="elev", z_unit="km"
    )
    return {
        "catdata": catdata,
        "x_km": np.asarray(x_km, dtype=float),
        "z_axes_km": z_axes_km,
    }


def inventory_station_arrays(inventory: Any) -> dict[str, list[float] | np.ndarray]:
    """
    Extract station lat/lon/elevation (m) from an ObsPy Inventory (same loop as
    :meth:`~vdapseisutils.core.maps.cross_section.CrossSection.plot_inventory`).
    """
    station_lats: list[float] = []
    station_lons: list[float] = []
    station_elevs: list[float] = []

    for network in inventory:
        for station in network:
            if hasattr(station, "latitude") and hasattr(station, "longitude"):
                station_lats.append(float(station.latitude))
                station_lons.append(float(station.longitude))
                station_elevs.append(float(getattr(station, "elevation", 0.0) or 0.0))

    return {
        "lat": station_lats,
        "lon": station_lons,
        "elevation_m": station_elevs,
    }


def prep_inventory_for_cross_section(
    inventory: Any,
    A1: tuple[float, float],
    A2: tuple[float, float],
) -> dict[str, Any] | None:
    """
    Project inventory stations onto the cross-section line (elev, m → axes km).

    Returns ``None`` if no station coordinates were found (same as empty plot path).
    """
    raw = inventory_station_arrays(inventory)
    if not raw["lat"] or not raw["lon"]:
        return None
    lat = np.asarray(raw["lat"], dtype=float)
    lon = np.asarray(raw["lon"], dtype=float)
    x_km = _along_line_km_1d(lat, lon, A1, A2)
    z_axes_km = _z_to_axes_km(raw["elevation_m"], z_dir="elev", z_unit="m")
    return {
        "x_km": np.asarray(x_km, dtype=float),
        "z_axes_km": z_axes_km,
        "lat": lat,
        "lon": lon,
        "elevation_m": np.asarray(raw["elevation_m"], dtype=float),
    }


def project_latlon_to_cross_section(
    lat: np.ndarray | Sequence[float],
    lon: np.ndarray | Sequence[float],
    z: np.ndarray | Sequence[float] | None,
    A1: tuple[float, float],
    A2: tuple[float, float],
    *,
    z_dir: str = "depth",
    z_unit: str = "m",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project arbitrary lat/lon/(z) to cross-section coordinates (km, km).

    Matches :meth:`~vdapseisutils.core.maps.cross_section.CrossSection.scatter`.
    """
    lat_a = np.asarray(lat, dtype=float)
    lon_a = np.asarray(lon, dtype=float)
    if z is None:
        z = np.zeros_like(lat_a, dtype=float)
    x_km = _along_line_km_1d(lat_a, lon_a, A1, A2)
    z_axes_km = _z_to_axes_km(z, z_dir=z_dir, z_unit=z_unit)
    return x_km, np.atleast_1d(z_axes_km).ravel()
