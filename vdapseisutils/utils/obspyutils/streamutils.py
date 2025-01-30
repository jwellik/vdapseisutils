"""
streamutils
[x] Move all funtions to utils.obspyutils.streamutils
[x] preprocess_data()
"""

import numpy as np
from numpy import round, dtype
from obspy import Stream


winston_gap_value = -2**31


def same_data_type(st):
    """Ensures that all Traces have the same data type"""
    for tr in st:
        # deal with error when sub-traces have different dtypes
        if tr.data.dtype.name != 'int32':
            tr.data = tr.data.astype('int32')
        if tr.data.dtype != dtype('int32'):
            tr.data = tr.data.astype('int32')
        # deal with rare error when sub-traces have different sample rates
        if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
            tr.stats.sampling_rate = round(tr.stats.sampling_rate)
    return st


def preprocess(st, resample=None, taper=5.0, filter=None, trim=None):
    """
    PREPROCESS Basic pre-processing steps for short time chunks of data

    EXAMPLE:
    >>> preprocess(st, resample=25.0, filter=["bandpass", {"freqmin": 1.0, "freqmax": 10.0}])

    :param st: ObsPy Stream object
    :param resample: desired sample rate (Hz)
    :param taper: seconds to taper beginning and end of trace before filtering
    :param filter: list : [<filter_type, {<filter_kwargs>]
    :param trim: (t1, t2) to trim extent of trace
    :return:
    """

    # filter_defaults = {"freqmin": 1.0, "freqmax": 10, "corners":2, "zerophase":True}  # Not used, at the moment

    st = same_data_type(st)
    st.detrend('demean')
    if resample:
        for tr in st:
            if tr.stats['sampling_rate'] != resample:
                tr.resample(resample)
    st.taper(max_percentage=None, max_length=taper)
    if filter:
        st.filter(filter[0], **filter[1])  # applies filter_type and filter_kwargs
    if trim:
        st.trim(trim[0], trim[1])

    return st


def removeWinstonGaps(st, winston_gap_value=winston_gap_value, fill_value=0):
    st2 = st.copy()  # so that input stream is left untouched
    for m in range(len(st2)):
        st2[m].data = np.where(st2[m].data == winston_gap_value, fill_value, st2[m].data) # replace -2**31 (Winston NaN token) w 0
    return st2


def replaceGapValue(st, gap_value=np.nan, fill_value=0 ):
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    return st


def clip(st, clip_threshold):
    # CLIP Clips Stream data to +/- clip_threshold
    st2 = st.copy()
    for tr in st2:
        tr.data[np.where(tr.data > clip_threshold)] = clip_threshold
        tr.data[np.where(tr.data < clip_threshold * -1)] = clip_threshold * -1
    return st2


def sortStreamByNSLClist(st_in, nslc_list, verbose=False):
    from obspy import Stream

    # List of NSLCs in the Input Stream
    NSLC_PRESENT = []
    for tr in st_in:
        NSLC_PRESENT.append(tr.id)

    st_custom = Stream()
    for nslc in nslc_list:
        if verbose: print(nslc)
        try:
            idx = NSLC_PRESENT.index(nslc)
            if verbose: print(st_in[idx])
            st_custom += st_in[idx]
        except:
            if verbose:
                if verbose: print("{} not found in Stream".format(nslc))


    return st_custom


def createEmptyTrace(nslc, t1, t2, sampling_rate=100, dtype='int32'):
    from obspy import Trace
    from vdapseisutils.core.datasource.nslcutils import str2nslc

    net, sta, loc, cha = str2nslc(nslc)

    stmp = Trace()
    stmp.stats['network'] = net
    stmp.stats['station'] = sta
    stmp.stats['location'] = loc
    stmp.stats['channel'] = cha
    stmp.stats['sampling_rate'] = sampling_rate
    stmp.stats['starttime'] = t1
    stmp.data = np.zeros(int((t2 - t1) * stmp.stats['sampling_rate']), dtype=dtype)

    return stmp


def idselect(st, ids):
    st2 = Stream()
    for id in ids:
        st2 += st.select(id=id)
    return st2


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
        """Computes frequency filtered RSAM using RMS

        * Does NOT affect the original Stream

        :param window_length (int) RSAM window length in minutes
        :param step (int) RSAM window interval in minutes (default: same as 'window_length')
        :param freqmin Lower bound of filterband (default 1000 seconds)
        :param freqmax Upper bound of filterband (default 1000 Hertz)
        """

        step = step if step else window_length  # default step to same as 'window_length'
        if step > window_length:
            raise ValueError("Error: step must be less than or equal to window_length.")

        # Prepare Stream
        st = SuperStream(self.copy())  # hard copy of original Stream for operations
        st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        st = st.merge(fill_value=0, method=1)  # merge same NSLCs

        # Filter and compute RMS
        for tr in st:
            # initialize ffrsam vectors
            rms = []
            tvec = []

            for ws in tr.slide(window_length*60, step*60):  # convert minutes to seconds
                # rms.append(np.sqrt(np.nanmean(np.square(ws[0].data))))
                rms.append(np.sqrt(np.mean(np.square(ws.data))))
                tvec.append(ws.stats.starttime)

            # Store data back to Trace
            tr.data = np.array(rms)
            tr.stats.starttime = tvec[0]  # force starttime (tvec[0] is a UTCDateTime); endtime is read only
            tr.stats.npts = len(tvec)
            tr.stats.delta = tvec[1] - tvec[0]  # sets delta and sampling_rate (subtract 2 UTCDateTimes returns seconds)

        return st
