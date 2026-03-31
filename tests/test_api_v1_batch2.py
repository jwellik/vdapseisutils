"""Tests for API v1 Batch 2 (compute scaffold, maps shim, plot_trace label, lazy pickqc)."""

import ast
from pathlib import Path

import numpy as np
import pytest
from obspy import Trace, UTCDateTime


def test_maps_shim_same_objects_as_split_modules():
    from vdapseisutils.core.maps import map as map_mod
    from vdapseisutils.core.maps import maps as maps_shim

    assert maps_shim.Map is map_mod.Map
    assert maps_shim.add_hillshade_pygmt is map_mod.add_hillshade_pygmt

    from vdapseisutils.core.maps import cross_section, legends, map_tiles, time_series, utils, volcano_figure

    assert maps_shim.prep_catalog_data_mpl is utils.prep_catalog_data_mpl
    assert maps_shim.ShadedReliefESRI is map_tiles.ShadedReliefESRI

    assert maps_shim.CrossSection is cross_section.CrossSection
    assert maps_shim.TimeSeries is time_series.TimeSeries
    assert maps_shim.VolcanoFigure is volcano_figure.VolcanoFigure
    assert maps_shim.MagLegend is legends.MagLegend
    assert maps_shim.ColorBar is legends.ColorBar
    assert maps_shim.get_scale_length is utils.get_scale_length
    assert maps_shim.choose_scale_bar_length is utils.choose_scale_bar_length


def test_compute_package_importable():
    import vdapseisutils.compute as c

    assert hasattr(c, "catalog")
    assert hasattr(c, "waveforms")
    assert hasattr(c, "maps")


def test_plot_trace_nslc_label_text():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vdapseisutils.core.swarmmpl.clipboard import plot_trace

    tr = Trace(
        data=np.zeros(200),
        header={
            "network": "XX",
            "station": "ABC",
            "location": "00",
            "channel": "HHZ",
            "starttime": UTCDateTime("2020-01-01"),
            "sampling_rate": 100.0,
            "npts": 200,
        },
    )
    fig = plot_trace(tr, mode="w", tick_type="datetime")
    texts = [t.get_text() for t in fig.axes[-1].texts]
    joined = "\n".join(texts)
    assert "XX.ABC" in joined, joined
    plt.close(fig)


def test_plot_trace_nslc_label_swarmmpl2():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vdapseisutils.core.swarmmpl2.clipboard import plot_trace

    tr = Trace(
        data=np.zeros(200),
        header={
            "network": "YY",
            "station": "STA",
            "location": "01",
            "channel": "EHZ",
            "starttime": UTCDateTime("2021-06-01"),
            "sampling_rate": 50.0,
            "npts": 200,
        },
    )
    fig = plot_trace(tr, mode="w", tick_type="datetime")
    joined = "\n".join(t.get_text() for t in fig.axes[-1].texts)
    assert "YY.STA" in joined
    plt.close(fig)


def test_pickqc_has_no_module_level_pyplot_import():
    """pickqc must not import pyplot at module import time (pandas may still load mpl)."""
    import vdapseisutils.utils.obspyutils.catalog.pickqc as pq

    tree = ast.parse(Path(pq.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "matplotlib.pyplot"
        elif isinstance(node, ast.ImportFrom):
            if node.module == "matplotlib.pyplot":
                pytest.fail("module-level matplotlib.pyplot import in pickqc.py")
            if node.module == "matplotlib":
                for alias in node.names:
                    if alias.name == "pyplot":
                        pytest.fail("module-level matplotlib pyplot import in pickqc.py")
