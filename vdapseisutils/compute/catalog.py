"""
Backend-neutral catalog point preparation for plotting and dashboards.

See ``prepare_catalog_points`` and ``.local/api-v1-coord/API_V1_CANONICAL.md`` §7.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import Catalog

from vdapseisutils.obspy_ext.catalog.origin import get_primary_origin

TimeEncoding = Literal["utcdatetime", "utc", "mpl_date", "datetime", "datetime64"]
OriginPolicy = Literal["preferred_or_first", "first", "last"]


def _time_format_kw_to_encoding(time_format: str) -> str:
    """Map legacy ``catalog2txyzm`` / ``prep_catalog_data_mpl`` names to encodings."""
    mapping = {
        "UTCDateTime": "utcdatetime",
        "matplotlib": "mpl_date",
        "datetime": "datetime",
    }
    return mapping.get(time_format, time_format)


def _encode_time(utct: UTCDateTime, time_encoding: str):
    if time_encoding in ("utcdatetime", "utc"):
        return utct
    if time_encoding == "mpl_date":
        return float(utct.matplotlib_date)
    if time_encoding == "datetime":
        return utct.datetime
    if time_encoding == "datetime64":
        return np.datetime64(utct.isoformat())
    return utct.strftime(time_encoding)


def _pick_origin(event, origin_policy: OriginPolicy):
    if origin_policy == "preferred_or_first":
        return get_primary_origin(event)
    if origin_policy == "first":
        return event.origins[0] if event.origins else None
    if origin_policy == "last":
        return event.origins[-1] if event.origins else None
    raise ValueError(f"Unknown origin_policy: {origin_policy!r}")


def prepare_catalog_points(
    catalog: Catalog,
    *,
    time_encoding: str | TimeEncoding = "utcdatetime",
    origin_policy: OriginPolicy = "preferred_or_first",
    depth_unit: str = "km",
    z_dir: str = "depth",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a tabular, matplotlib-free summary of catalog hypocenters.

    Parameters
    ----------
    catalog : obspy.core.event.Catalog
    time_encoding : str
        One of ``"utcdatetime"`` / ``"utc"`` (ObsPy :class:`~obspy.core.utcdatetime.UTCDateTime`),
        ``"mpl_date"`` (float, matplotlib day number), ``"datetime"`` (timezone-aware
        :class:`~datetime.datetime`), ``"datetime64"``, or any :func:`~datetime.datetime.strftime`
        pattern string (e.g. ``"%Y-%m-%d"``).
    origin_policy : str
        ``"preferred_or_first"`` — :func:`~vdapseisutils.obspy_ext.catalog.origin.get_primary_origin`;
        ``"first"`` / ``"last"`` — first or last entry in ``event.origins`` (legacy alignment).
    depth_unit : {"km", "m"}
        Units of the returned ``depth`` column (after ObsPy m → km or m conversion).
    z_dir : {"depth", "elev"}
        Same convention as :func:`~vdapseisutils.utils.obspyutils.catalogutils.catalog2txyzm`.
    verbose : bool
        If True, print a short message when an event is skipped.

    Returns
    -------
    pandas.DataFrame
        Columns: ``time``, ``lat``, ``lon``, ``depth`` (km or m, positive down in depth mode),
        ``mag``. Rows follow catalog event order; events without a usable origin are omitted.
    """
    if depth_unit == "m":
        dconvert = 1
    elif depth_unit == "km":
        dconvert = 1000
    else:
        if verbose:
            print("'depth_unit' not understood. Default value 'km' is used.")
        dconvert = 1000

    if z_dir == "depth":
        z_dir_convert = 1
    elif z_dir == "elev":
        z_dir_convert = -1
    else:
        if verbose:
            print("'z_dir' not understood. Default value 'depth' is used.")
        z_dir_convert = 1

    rows = []
    for event in catalog:
        try:
            origin = _pick_origin(event, origin_policy)
            if origin is None:
                continue
            lat = origin.latitude
            lon = origin.longitude
            dep = origin.depth
            if dep is None:
                dep = float("nan")
            else:
                dep = dep / dconvert * z_dir_convert
            if event.magnitudes:
                m = event.magnitudes[-1]
                mag = m.mag if m.mag is not None else -1
            else:
                mag = -1
            t = _encode_time(origin.time, time_encoding)
            rows.append({"time": t, "lat": lat, "lon": lon, "depth": dep, "mag": mag})
        except Exception as err:  # noqa: BLE001 — mirror catalog2txyzm resilience
            print(f"Skipping event due to error: {err}")

    return pd.DataFrame(rows)


def prepare_catalog_points_from_time_format(
    catalog: Catalog,
    *,
    time_format: str = "UTCDateTime",
    origin_policy: OriginPolicy = "preferred_or_first",
    depth_unit: str = "km",
    z_dir: str = "depth",
    verbose: bool = False,
) -> pd.DataFrame:
    """Like :func:`prepare_catalog_points` but accepts legacy ``time_format`` kwargs."""
    enc = _time_format_kw_to_encoding(time_format)
    return prepare_catalog_points(
        catalog,
        time_encoding=enc,
        origin_policy=origin_policy,
        depth_unit=depth_unit,
        z_dir=z_dir,
        verbose=verbose,
    )
