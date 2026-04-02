"""T5: VClient waveform paths use _fetch and return VStream; construction smoke."""

import tempfile

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.core import AttribDict
from obspy.core.stream import Stream
from obspy.core.trace import Trace

from vdapseisutils.obspy_ext.client.vclient import VClient
from vdapseisutils.utils.obspyutils.stream.core import VStream


class _MockWaveformClient:
    """Minimal client implementing get_waveforms for _fetch-backed tests."""

    def get_waveforms(self, network, station, location, channel, starttime, endtime, **kwargs):
        hdr = AttribDict(
            {
                "network": network,
                "station": station,
                "location": location or "",
                "channel": channel,
                "sampling_rate": 20.0,
                "starttime": UTCDateTime(starttime),
            }
        )
        hdr.npts = 10
        tr = Trace(data=np.zeros(10, dtype="int32"), header=hdr)
        return Stream([tr])


def test_vclient_iris_is_fdsn():
    vc = VClient("IRIS")
    assert isinstance(vc.client, FDSNClient)


def test_vclient_sds_directory():
    with tempfile.TemporaryDirectory() as tmp:
        vc = VClient(tmp)
        assert isinstance(vc.client, SDSClient)


def test_get_waveforms_returns_vstream_via_fetch():
    vc = VClient.__new__(VClient)
    vc._client = _MockWaveformClient()
    vc.use_bulk = True

    t1, t2 = UTCDateTime("2020-01-01"), UTCDateTime("2020-01-01T00:05:00")
    st = vc.get_waveforms("IU", "ANMO", "00", "BHZ", t1, t2)
    assert isinstance(st, VStream)
    assert len(st) == 1


def test_get_waveforms_keyword_style_returns_vstream():
    vc = VClient.__new__(VClient)
    vc._client = _MockWaveformClient()
    vc.use_bulk = True

    t1, t2 = UTCDateTime("2020-01-01"), UTCDateTime("2020-01-01T00:05:00")
    st = vc.get_waveforms(
        network="IU",
        station="ANMO",
        location="00",
        channel="BHZ",
        starttime=t1,
        endtime=t2,
    )
    assert isinstance(st, VStream)


def test_get_waveforms_bulk_returns_vstream():
    vc = VClient.__new__(VClient)
    vc._client = _MockWaveformClient()
    vc.use_bulk = True

    t1, t2 = UTCDateTime("2020-01-01"), UTCDateTime("2020-01-01T00:05:00")
    bulk = [
        ("IU", "ANMO", "00", "BHZ", t1, t2),
        ("IU", "COLA", "00", "BHZ", t1, t2),
    ]
    st = vc.get_waveforms_bulk(bulk)
    assert isinstance(st, VStream)
    assert len(st) == 2
