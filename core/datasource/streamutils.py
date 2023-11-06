"""
streamutils
TODO Move all funtions to utils.obspyutils.streamutils
TODO preprocess_data()
"""

import numpy as np
from obspy import Stream

winston_gap_value = -2**31



def preprocess(st, resample=None, taper=5, filter_type=None, filter_kwargs=None, trim=None):
    """
    PREPROCESS Basic pre-processing steps for processing short time chunks of data

    :param st: ObsPy Stream object
    :param resample: desired sample rate (Hz)
    :param taper: seconds to taper beginning and end of trace before filtering
    :param filter_type: string : "bandpass", "highpass" or "lowpass"
    :param filter_kwargs: dict : any kwargs that are passed to ObsPy Stream filter() method
    :param trim: (t1, t2) to trim extent of trace
    :return:
    """

    # filter_defaults = {"freqmin": 1.0, "freqmax": 10, "corners":2, "zerophase":True}  # Not used, at the moment

    st.detrend('demean')
    if resample:
        for tr in st:
            if tr.stats['sampling_rate'] != resample:
                tr.resample(resample)
    st.taper(max_percentage=None, max_length=taper)
    if filter_type:
        st.filter(filter_type, **filter_kwargs)
    if trim:
        st.trim(trim[0], trim[1])

    return st


def removeWinstonGaps(st, winston_gap_value=winston_gap_value, fill_value=0):
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == winston_gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    return st

def replaceGapValue(st, gap_value=np.nan, fill_value=0 ):
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    return st

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
