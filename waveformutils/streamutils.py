import numpy as np

winston_gap_value = -2**31

def removeWinstonGaps( st, winston_gap_value=winston_gap_value, fill_value=0 ):   
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == winston_gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    #stmp = [np.where(stmp2.data == -2**31, 0, stmp2.data) for stmp2 in stmp] # replace -2**31 (Winston NaN token) w 0 <--- Could this be a 1 liner?
    return st


def replaceGapValue( st, gap_value=np.nan, fill_value=0 ):
    import numpy as np
    
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    return st

           
def sortStreamByNSLClist( st_in, nslc_list ):

    import numpy as np
    from obspy import Stream, Trace
    
    st_in = st_in.merge().sort()         # Ensure that data are merged and sorted
    # print('>>> sortStreamByNLSClist : len(st_in) {} ?= len(nslc_list) {}'.format(len(st_in), len(nslc_list)))
    # print(st_in)
    # print(nslc_list)
    # print('')
    
    nslc_sorted = sorted(nslc_list)                # creates new list of alphabetically sorted NSLCs (Not used?)
    sort_order = np.argsort(nslc_list, axis=-1)    # creates an array of the alphabetical position of each NSLC 

    st = Trace() * len(st_in)                      #
    for idx, tr in enumerate(sort_order):
        st[tr] = st_in[idx]
    
    return st


def cleanMaskedData( st ):
    
    return st