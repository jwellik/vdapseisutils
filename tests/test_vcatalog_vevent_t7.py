"""T7: VCatalog holds canonical VEvent instances; indexing returns ``self.events[i]``."""

from obspy import UTCDateTime
from obspy.core.event import Pick, read_events
from obspy.core.event import WaveformStreamID

from vdapseisutils.obspy_ext.catalog import VCatalog, VEvent


def test_vcatalog_events_are_vevent_and_getitem_is_identity():
    base = read_events()
    cat = VCatalog(base)
    assert all(isinstance(e, VEvent) for e in cat.events)
    for i in range(len(cat)):
        assert cat[i] is cat.events[i]


def test_vcatalog_slice_reuses_same_event_objects():
    base = read_events()
    cat = VCatalog(base)
    sub = cat[0:2]
    assert isinstance(sub, VCatalog)
    assert len(sub.events) == 2
    assert sub[0] is cat[0]
    assert sub[1] is cat[1]


def test_append_normalizes_to_vevent():
    cat = VCatalog()
    plain = read_events()[0]
    cat.append(plain)
    assert len(cat.events) == 1
    assert isinstance(cat.events[0], VEvent)
    assert cat[0] is cat.events[0]


def test_vevent_sort_picks_inplace_mutates_catalog_slot():
    wid = WaveformStreamID(
        network_code="XX",
        station_code="AAA",
        location_code="00",
        channel_code="BHZ",
    )
    plain = read_events()[0]
    t_late = UTCDateTime(2010, 1, 1, 0, 0, 3)
    t_mid = UTCDateTime(2010, 1, 1, 0, 0, 2)
    t_early = UTCDateTime(2010, 1, 1, 0, 0, 1)
    plain.picks = [
        Pick(time=t_late, waveform_id=wid),
        Pick(time=t_early, waveform_id=wid),
        Pick(time=t_mid, waveform_id=wid),
    ]
    cat = VCatalog([plain])
    ev = cat[0]
    assert ev is cat.events[0]
    ev.sort_picks(inplace=True)
    times = [p.time for p in cat.events[0].picks]
    assert times == [t_early, t_mid, t_late]
