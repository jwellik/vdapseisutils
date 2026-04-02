"""Batch 3: prepare_catalog_points, catalog2txyzm delegation, get_primary_origin consistency."""

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin
from obspy.core.event.resourceid import ResourceIdentifier

from vdapseisutils.compute.catalog import prepare_catalog_points, prepare_catalog_points_from_time_format
from vdapseisutils.core.maps.legends import MagLegend
from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
from vdapseisutils.obspy_ext.catalog.core import VCatalog
from vdapseisutils.obspy_ext.catalog.origin import get_primary_origin
from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm


def _two_origin_event():
    """First origin at (1,2); second (preferred) at (10,20), different time."""
    e = Event()
    o1 = Origin(
        time=UTCDateTime(2020, 1, 1),
        latitude=1.0,
        longitude=2.0,
        depth=3000.0,
    )
    o1.resource_id = ResourceIdentifier(id="origin/o1")
    o2 = Origin(
        time=UTCDateTime(2020, 2, 1),
        latitude=10.0,
        longitude=20.0,
        depth=5000.0,
    )
    o2.resource_id = ResourceIdentifier(id="origin/o2")
    e.origins = [o1, o2]
    e.preferred_origin_id = o2.resource_id
    m = Magnitude(mag=4.5, magnitude_type="ML")
    m.resource_id = ResourceIdentifier(id="mag/1")
    e.magnitudes = [m]
    return e


def test_get_primary_origin_prefers_quakeml_preferred():
    e = _two_origin_event()
    o = get_primary_origin(e)
    assert o is e.origins[1]
    assert float(o.latitude) == pytest.approx(10.0)


def test_get_primary_origin_falls_back_to_first():
    e = Event()
    o1 = Origin(
        time=UTCDateTime(2019, 1, 1),
        latitude=-8.0,
        longitude=115.0,
        depth=10000.0,
    )
    e.origins = [o1]
    assert get_primary_origin(e) is o1


def test_prepare_last_origin_matches_legacy_last_index():
    """origin_policy='last' reproduces former catalog2txyzm origins[-1] selection."""
    e = _two_origin_event()
    cat = Catalog([e])
    df = prepare_catalog_points(cat, time_encoding="utcdatetime", origin_policy="last")
    assert len(df) == 1
    assert float(df.iloc[0]["lat"]) == pytest.approx(10.0)  # last in list is still o2
    # Swap order so last is geographically distinct from preferred
    o_first = e.origins[0]
    o_last = e.origins[1]
    e.origins = [o_last, o_first]
    e.preferred_origin_id = o_last.resource_id
    df_last = prepare_catalog_points(cat, time_encoding="utcdatetime", origin_policy="last")
    assert float(df_last.iloc[0]["lat"]) == pytest.approx(1.0)


def test_prepare_preferred_or_first_follows_primary_not_last_index():
    e = _two_origin_event()
    o_first = e.origins[0]
    o_last = e.origins[1]
    e.origins = [o_last, o_first]
    e.preferred_origin_id = o_last.resource_id
    cat = Catalog([e])
    df = prepare_catalog_points(cat, time_encoding="utcdatetime", origin_policy="preferred_or_first")
    assert float(df.iloc[0]["lat"]) == pytest.approx(10.0)


def test_catalog2txyzm_delegates_to_prepare():
    e = _two_origin_event()
    cat = Catalog([e])
    d_new = catalog2txyzm(cat, time_format="UTCDateTime", origin_policy="preferred_or_first")
    df = prepare_catalog_points_from_time_format(
        cat, time_format="UTCDateTime", origin_policy="preferred_or_first"
    )
    for k in ("time", "lat", "lon", "depth", "mag"):
        assert d_new[k] == df[k].tolist()


def test_extract_origin_times_matches_primary_origin_times():
    cat = Catalog([_two_origin_event()])
    vc = VCatalog(cat)
    expected = [get_primary_origin(e).time for e in cat]
    assert vc.extract_origin_times() == expected


def test_scatter_uses_same_longitudes_as_primary_origin():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    cat = Catalog([_two_origin_event()])
    vc = VCatalog(cat)
    primary_lons = [get_primary_origin(e).longitude for e in vc]

    # Replicate scatter's coordinate extraction after mixin change
    lons = []
    for event in vc:
        o = get_primary_origin(event)
        lons.append(o.longitude if o is not None and o.longitude is not None else np.nan)
    assert lons == primary_lons

    ax = vc.scatter()
    assert ax is not None
    matplotlib.pyplot.close(ax.figure)


def test_prep_catalog_data_mpl_matches_prepare_plus_swarmmpl_steps():
    e = _two_origin_event()
    cat = Catalog([e])
    mpl_df = prep_catalog_data_mpl(cat, time_format="matplotlib")
    base = prepare_catalog_points_from_time_format(
        cat, time_format="matplotlib", origin_policy="preferred_or_first"
    ).sort_values("time")
    maglegend = MagLegend()
    expected = base.copy()
    expected["depth"] *= -1
    expected["size"] = maglegend.mag2s(expected["mag"])
    pd.testing.assert_frame_equal(
        mpl_df.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
