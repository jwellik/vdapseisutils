from obspy import UTCDateTime

def info():
    print('!!! vdapseisutils.waveformutils.timeutils')


def createTimeChunks_v1(tstart, tend, nsec, verbose=False):
    
    print('!!!!!!! This is a deprecated version. !!!!!!')
    
    
    tstart = UTCDateTime(tstart)
    tend   = UTCDateTime(tend)
    
    starts = []
    ends   = []
    
    if tend-tstart > nsec:
        if verbose: print('>>> Splitting time period into smaller chunks...')
        t1 = tstart
        while t1 < tend:
            t2 = min(t1+nsec,tend)
            if verbose: print('>>> New chunk : {} to {}'.format(t1,t2))
            starts.append(t1); ends.append(t2)
            t1 += nsec
    else:
        starts.append(tstart)
        ends.append(tend)
            
    return starts, ends


def createTimeChunks(tstart, tend, nsec, verbose=False):
    """CREATETIMECHUNKS Splits start/stop pair into smaller segments
    
    Input arguments:
        tstart  : ObsPy UTCDateTime or something understood by UTCDateTime
        tend    : ''
        nsec    : float : Length of each new segment

        verbose : bool
        
    Examples:

    >>> createTimeChunks( '2017/03/17', '2017/03/17 03:00:00', 3600)
    ... >>> New chunk : 2017/03/17 00:00:00 - 2017/03/17 01:00:00
    ... >>> New chunk : 2017/03/17 01:00:00 - 2017/03/17 02:00:00
    ... >>> New chunk : 2017/03/17 02:00:00 - 2017/03/17 03:00:00

    
    """
    
    tstart = UTCDateTime(tstart)
    tend   = UTCDateTime(tend)
    
    starts = []
    ends   = []
    
    if tend-tstart > nsec:
        # Create smaller time chunks
        if verbose: print('>>> Splitting time period into smaller chunks...')
        t1 = tstart
        while t1 < tend:
            t2 = min(t1+nsec,tend)
            if verbose: print('>>> New chunk : {} to {}'.format(t1,t2))
            starts.append(t1); ends.append(t2)
            t1 += nsec
    else:
        # No need to create smaller time chunks
        starts.append(tstart)
        ends.append(tend)
            
    return starts, ends
