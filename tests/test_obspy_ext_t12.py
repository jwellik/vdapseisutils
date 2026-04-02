"""T12: obspy_ext public API — VStreamID, VClient/DataSource construction, I/O wrappers, VCatalog identity."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Catalog, UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.core.event import Event
from obspy.core.stream import Stream
from obspy.core.trace import Trace

from vdapseisutils.obspy_ext import (
    DataSource,
    VCatalog,
    VClient,
    VEvent,
    VInventory,
    VStream,
    VStreamID,
    read,
    read_events,
    read_inventory,
    waveID,
)
def test_vstreamid_from_four_components():
    vid = VStreamID("IU", "ANMO", "00", "BHZ")
    assert isinstance(vid, VStreamID)
    assert vid.network_code == "IU"
    assert vid.station_code == "ANMO"
    assert vid.location_code == "00"
    assert vid.channel_code == "BHZ"
    assert vid.nslc() == "IU.ANMO.00.BHZ"


def test_vstreamid_from_nslc_string():
    vid = VStreamID("IU.ANMO.00.BHZ")
    assert vid.nslc() == "IU.ANMO.00.BHZ"


def test_parse_wave_id_tuple():
    from vdapseisutils.obspy_ext import parse_wave_id

    assert parse_wave_id("IU.ANMO.00.BHZ") == ("IU", "ANMO", "00", "BHZ")


def test_waveid_is_deprecated_alias_of_vstreamid():
    assert waveID is VStreamID


def test_vclient_fdsn_uri_constructs_fdsn_client():
    vc = VClient("fdsnws://IRIS")
    assert isinstance(vc.client, FDSNClient)


def test_datasource_alias_same_class_and_uri_construction():
    assert DataSource is VClient
    ds = DataSource("fdsnws://IRIS")
    assert isinstance(ds.client, FDSNClient)


def test_vclient_heuristic_short_fdsn_name():
    vc = VClient("IRIS")
    assert isinstance(vc.client, FDSNClient)


def test_vclient_sds_uri_uses_sds_client(tmp_path):
    # SDS root must exist for SDSClient
    root = tmp_path / "sds"
    root.mkdir()
    vc = VClient(f"sds://{root}")
    assert isinstance(vc.client, SDSClient)


@pytest.fixture()
def mini_mseed_path(tmp_path):
    hdr = {
        "network": "XX",
        "station": "TEST",
        "location": "",
        "channel": "HHZ",
        "sampling_rate": 20.0,
        "starttime": UTCDateTime("2020-01-01T00:00:00"),
    }
    tr = Trace(data=np.zeros(5, dtype=np.int32), header=hdr)
    st = Stream([tr])
    path = tmp_path / "mini.mseed"
    st.write(str(path), format="MSEED")
    return path


@pytest.fixture()
def mini_quakeml_path(tmp_path):
    cat = Catalog(
        events=[
            Event(resource_id="smi:local/event/1"),
            Event(resource_id="smi:local/event/2"),
        ]
    )
    path = tmp_path / "mini.xml"
    cat.write(str(path), format="QUAKEML")
    return path


@pytest.fixture()
def mini_stationxml_path(tmp_path):
    # Minimal StationXML ObsPy can write/read
    from obspy import read_inventory

    inv = read_inventory()
    path = tmp_path / "mini.xml"
    inv.write(str(path), format="STATIONXML")
    return path


def test_read_returns_vstream(mini_mseed_path):
    st = read(str(mini_mseed_path))
    assert isinstance(st, VStream)
    assert len(st) >= 1


def test_read_events_returns_vcatalog(mini_quakeml_path):
    cat = read_events(str(mini_quakeml_path))
    assert isinstance(cat, VCatalog)
    assert len(cat.events) == 2


def test_read_inventory_returns_vinventory(mini_stationxml_path):
    inv = read_inventory(str(mini_stationxml_path))
    assert isinstance(inv, VInventory)


def test_vcatalog_from_small_catalog_events_and_getitem_identity():
    """``catalog.events[i]`` is ``VEvent`` and ``catalog[i]`` is the same object (T7)."""
    plain = Catalog(
        events=[
            Event(resource_id="smi:local/event/a"),
            Event(resource_id="smi:local/event/b"),
        ]
    )
    catalog = VCatalog(plain)
    assert len(catalog) == 2
    for i in range(len(catalog)):
        assert isinstance(catalog.events[i], VEvent)
        assert catalog[i] is catalog.events[i]
