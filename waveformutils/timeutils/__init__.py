import numpy as np
import pandas as pd
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

    #>>> createTimeChunks( '2017/03/17', '2017/03/17 03:00:00', 3600)
    #... >>> New chunk : 2017/03/17 00:00:00 - 2017/03/17 01:00:00
    #... >>> New chunk : 2017/03/17 01:00:00 - 2017/03/17 02:00:00
    #... >>> New chunk : 2017/03/17 02:00:00 - 2017/03/17 03:00:00

    
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


def date_range(start, stop, freq="1D", dtype="UTCDateTime"):
    """Creates an array of datetimes of the desired type
    Relies on Pandas date_range() to do the real work
    'freq' can be specified as anything understood by date_range or as an integer of days
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    """

    # This might be helpful:
    # https://stackoverflow.com/questions/12137277/how-can-i-make-a-python-numpy-arange-of-datetimev

    dr_tmp = pd.date_range(start, stop, freq=freq)

    dr = []
    if dtype.lower() == "UTCDateTime".lower():
        [dr.append(UTCDateTime(d)) for d in dr_tmp]
    if dtype.lower() == "pdTimestamp".lower():  # Pandas Timestamp
        [dr.append(pd.Timestamp(UTCDateTime(d))) for d in dr_tmp]
    if dtype.lower() == "matplotlib".lower():  # Matplotlib date
        [dr.append(UTCDateTime(d).matplotlib_date) for d in dr_tmp]
    if dtype.lower() == "datetime".lower():  # Python Datetime object
        [dr.append(UTCDateTime(d).datetime) for d in dr_tmp]
    if dtype.lower() == "timestamp".lower():  # returns UTC timestamp in seconds
        [dr.append(UTCDateTime(d).timestamp) for d in dr_tmp]
    if dtype.lower() == "datetime64".lower():  # returns UTC timestamp in seconds
        [dr.append(np.datetime64(UTCDateTime(d))) for d in dr_tmp]

    return dr


def date_period():
    pass
