import numpy as np

winston_gap_value = -2**31

def removeWinstonGaps(st, winston_gap_value=winston_gap_value, fill_value=0 ):
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == winston_gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    #stmp = [np.where(stmp2.data == -2**31, 0, stmp2.data) for stmp2 in stmp] # replace -2**31 (Winston NaN token) w 0 <--- Could this be a 1 liner?
    return st


def replaceGapValue(st, gap_value=np.nan, fill_value=0 ):
    import numpy as np
    
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    return st

           
def sortStreamByNSLClist(st_in, nslc_list):
    """SORTSTREAMBYNSLCLIST
    This method inherently removes Traces not in the list
    NSLCs not present in Stream can be created as empty, if desired
    """
    # TODO: Replace with code from PROJECTS/STALTA_tuner/sortNSLC_test.py

    import numpy as np
    from obspy import Stream, Trace
    
    st_in = st_in.merge().sort()         # Ensure that data are merged and sorted
    print('>>> sortStreamByNLSClist : len(st_in) {} ?= len(nslc_list) {}'.format(len(st_in), len(nslc_list)))
    # print(st_in)
    # print(nslc_list)
    # print('')
    
    nslc_sorted = sorted(nslc_list)                # creates new list of alphabetically sorted NSLCs (Not used?)
    sort_order = np.argsort(nslc_list, axis=-1)    # creates an array of the alphabetical position of each NSLC 

    st = Trace() * len(st_in)                      #
    for idx, tr in enumerate(sort_order):
        st[tr] = st_in[idx]
    
    return st


def createEmptyTrace(nslc, t1, t2, sampling_rate=100, dtype='int32'):
    from obspy import Trace
    from vdapseisutils.waveformutils.nslcutils import str2nslc

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


def cleanMaskedData( st ):
    
    return st