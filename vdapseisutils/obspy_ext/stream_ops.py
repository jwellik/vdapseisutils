"""
Stream-level helpers for ObsPy (preprocessing, gaps, NSLC ordering, alignment).

Canonical location for logic previously split across ``utils.obspyutils.streamutils``
and ``utils.obspyutils.stream.utils``.
"""

from __future__ import annotations

import numpy as np
from numpy import dtype, round
from obspy import Stream

from vdapseisutils.obspy_ext.stream_id import VStreamID

winston_gap_value = -(2**31)

__all__ = [
    "winston_gap_value",
    "same_data_type",
    "preprocess",
    "preprocess_stream",
    "removeWinstonGaps",
    "replaceGapValue",
    "clip",
    "sortStreamByNSLClist",
    "createEmptyTrace",
    "idselect",
    "align_streams",
    "round_trace_sampling_rates",
    "SuperStream",
]


def round_trace_sampling_rates(st: Stream) -> Stream:
    """Round each trace's sampling rate to the nearest integer Hz (in place)."""
    for tr in st:
        if tr.stats.sampling_rate != np.round(tr.stats.sampling_rate):
            tr.stats.sampling_rate = float(np.round(tr.stats.sampling_rate))
    return st


def same_data_type(st: Stream) -> Stream:
    """Ensure all traces use int32 samples and integer-like sampling rates."""
    for tr in st:
        if tr.data.dtype.name != "int32":
            tr.data = tr.data.astype("int32")
        if tr.data.dtype != dtype("int32"):
            tr.data = tr.data.astype("int32")
        if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
            tr.stats.sampling_rate = round(tr.stats.sampling_rate)
    return st


def preprocess_stream(
    st: Stream,
    resample=None,
    taper: float = 5.0,
    filter=None,
    trim=None,
) -> Stream:
    """
    Basic preprocessing for short time chunks (in place).

    Parameters mirror the legacy ``preprocess`` helper: ``filter`` is a list
    ``[filter_type, {kwargs}]`` passed to :meth:`obspy.core.stream.Stream.filter`.
    """
    same_data_type(st)
    st.detrend("demean")
    if resample:
        for tr in st:
            if tr.stats["sampling_rate"] != resample:
                tr.resample(resample)
    st.taper(max_percentage=None, max_length=taper)
    if filter:
        st.filter(filter[0], **filter[1])
    if trim:
        st.trim(trim[0], trim[1])
    return st


def preprocess(st: Stream, resample=None, taper: float = 5.0, filter=None, trim=None) -> Stream:
    """Alias of :func:`preprocess_stream` for backward compatibility."""
    return preprocess_stream(st, resample=resample, taper=taper, filter=filter, trim=trim)


def removeWinstonGaps(st: Stream, winston_gap_value: int = winston_gap_value, fill_value=0):
    st2 = st.copy()
    for m in range(len(st2)):
        st2[m].data = np.where(
            st2[m].data == winston_gap_value, fill_value, st2[m].data
        )
    return st2


def replaceGapValue(st: Stream, gap_value=np.nan, fill_value=0):
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == gap_value, fill_value, st[m].data)
    return st


def clip(st: Stream, clip_threshold):
    st2 = st.copy()
    for tr in st2:
        tr.data[np.where(tr.data > clip_threshold)] = clip_threshold
        tr.data[np.where(tr.data < clip_threshold * -1)] = clip_threshold * -1
    return st2


def sortStreamByNSLClist(st_in: Stream, nslc_list, verbose: bool = False):
    from obspy import Stream as _Stream

    nslc_present = [tr.id for tr in st_in]
    st_custom = _Stream()
    for nslc in nslc_list:
        if verbose:
            print(nslc)
        try:
            idx = nslc_present.index(nslc)
            if verbose:
                print(st_in[idx])
            st_custom += st_in[idx]
        except ValueError:
            if verbose:
                print("{} not found in Stream".format(nslc))
    return st_custom


def createEmptyTrace(nslc, t1, t2, sampling_rate=100, dtype="int32"):
    from obspy import Trace

    net, sta, loc, cha = VStreamID(nslc).parts()

    stmp = Trace()
    stmp.stats["network"] = net
    stmp.stats["station"] = sta
    stmp.stats["location"] = loc
    stmp.stats["channel"] = cha
    stmp.stats["sampling_rate"] = sampling_rate
    stmp.stats["starttime"] = t1
    stmp.data = np.zeros(
        int((t2 - t1) * stmp.stats["sampling_rate"]), dtype=dtype
    )

    return stmp


def idselect(st: Stream, ids):
    st2 = Stream()
    for stream_id in ids:
        st2 += st.select(id=stream_id)
    return st2


def align_streams(st_list, shift_len=3.0, pad=True, fill_value=0, main=None):
    from obspy import Stream as _Stream
    from eqcorrscan.utils.stacking import align_traces

    st_select2 = [st.copy() for st in st_list]
    sr = st_list[0][0].stats.sampling_rate

    master = st_select2[main][0] if main is not None else None

    tmp = _Stream()
    for _tmp in st_select2:
        tmp += _tmp
    shifts, corr = align_traces(
        tmp, master=master, shift_len=int(shift_len * sr)
    )

    st_align = [st.copy() for st in st_select2]
    for st, shift_sec in zip(st_align, shifts):
        tstart = st[0].stats.starttime
        tend = st[0].stats.endtime
        st.trim(tstart - shift_sec, tend - shift_sec, pad=pad, fill_value=fill_value)

    return st_align, shifts, corr


class SuperStream(Stream):
    def __init__(self, traces=None, custom_attribute=None):
        super().__init__(traces)

    def info(self):
        print("vdapseisutils SuperStream:")
        print(self)

    def clip(self, clip_threshold):
        return clip(self, clip_threshold)

    def preproces(self):
        return same_data_type(self)

    def ffrsam(self, window_length=60, step=None, freqmin=0.0001, freqmax=1000.0):
        step = step if step else window_length
        if step > window_length:
            raise ValueError("Error: step must be less than or equal to window_length.")

        st = SuperStream(self.copy())
        st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        st = st.merge(fill_value=0, method=1)

        for tr in st:
            rms = []
            tvec = []

            for ws in tr.slide(window_length * 60, step * 60):
                rms.append(np.sqrt(np.mean(np.square(ws.data))))
                tvec.append(ws.stats.starttime)

            tr.data = np.array(rms)
            tr.stats.starttime = tvec[0]
            tr.stats.npts = len(tvec)
            tr.stats.delta = tvec[1] - tvec[0]

        return st
