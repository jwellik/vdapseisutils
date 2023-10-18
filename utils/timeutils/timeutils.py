import datetime

import numpy as np
import pandas as pd
from obspy import UTCDateTime


def convert_timeformat(input, from_format, to_format):

    output = []

    # Convert from UTCDateTime
    if from_format.lower() == "UTCDateTime".lower():
        if to_format.lower() == "UTCDateTime".lower():
            [output.append(UTCDateTime(d)) for d in input]
        elif to_format.lower() == "pdTimestamp".lower():  # Pandas Timestamp
            [output.append(pd.Timestamp(UTCDateTime(d))) for d in input]
        elif to_format.lower() == "matplotlib".lower():  # Matplotlib date
            [output.append(UTCDateTime(d).matplotlib_date) for d in input]
        elif to_format.lower() == "datetime".lower():  # Python Datetime object
            [output.append(UTCDateTime(d).datetime) for d in input]
        elif to_format.lower() == "timestamp".lower():  # returns UTC timestamp in seconds
            [output.append(UTCDateTime(d).timestamp) for d in input]
        elif to_format.lower() == "datetime64".lower():  # returns UTC timestamp in seconds
            [output.append(np.datetime64(UTCDateTime(d))) for d in input]
        else:
            raise Exception("Output time format ({}) not understood.".format(to_format))

    else:
        raise Exception("Input time format ({}) not understood.".format(from_format))

    return output


def interval_range(start, stop, freq="1D", dtype="UTCDateTime", **kwargs):
    """INTERVAL_RANGE Wrapper for pd.interval_range() that always returns dates as dytpe as list of 2 element tuples

    'freq' can be specified as anything understood by date_range or as an integer of days
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    """

    # Force start, stop to be ObsPy UTCDateTime
    start = UTCDateTime(start)
    stop = UTCDateTime(stop)
    ir_tmp = pd.interval_range(start.datetime, stop.datetime, freq=freq, **kwargs)

    atuples = ir_tmp.to_tuples()
    l = []
    for at in atuples:
        l.append([UTCDateTime(at[0]), UTCDateTime(at[1])])

    ir = convert_timeformat(l, "UTCDateTime", dtype)

    return ir


def date_range(start, stop, freq="1D", dtype="UTCDateTime", **kwargs):
    """DATE_RANGE Wrapper for pd.date_range() that always returns dates as dytpe

    'freq' can be specified as anything understood by date_range or as an integer of days
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    """

    # This might be helpful:
    # https://stackoverflow.com/questions/12137277/how-can-i-make-a-python-numpy-arange-of-datetimev

    start = UTCDateTime(start)
    stop = UTCDateTime(stop)
    dr_tmp = pd.date_range(start.datetime, stop.datetime, freq=freq, **kwargs)

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


def time_range_dev(start, end, freq="1D", dur="freq", buffer=[0, 0], dtype="UTCDateTime", **kwargs):
    """TIME_RANGE Extends Panda's date_range() to return a list of start times and a list of end times

    :param start: Any datetime format understood by ObsPy UTCDateTime
    :param end: Any datetime format understood by ObsPy UTCDateTime
    :param freq: Any string representation of a timedelta or an integer (days) : Spacing between start times
    :param dur: Any string representation of a timedelta or an integer (days) : Spacing between start/end pairs
        default: 'freq' (same as frequency); set to longer than freq for a set of sliding windows
    :params buffer: Any str
    :param dtype: date time object type to be returned
    :param kwargs: Any key word arguments understood by Panda's date_range
    :return: starts, ends
        starts : List of start times
        ends : List of corresponding end times
    """

    # This might be helpful:
    # https://stackoverflow.com/questions/12137277/how-can-i-make-a-python-numpy-arange-of-datetimev

    start = UTCDateTime(start)
    end = UTCDateTime(end)
    dur = freq if dur == "freq" else dur
    starts = pd.date_range(start.datetime, end.datetime, freq=freq, **kwargs)
    ends = starts + pd.Timedelta(dur)
    starts = convert_timeformat(starts, "UTCDateTime", dtype) + buffer[0]
    ends = convert_timeformat(ends, "UTCDateTime", dtype) + buffer[0]

    return starts, ends


def time_range(tstart, tend, freq="1D", dur="freq"):

    tstart = UTCDateTime(tstart)
    tend = UTCDateTime(tend)
    dur = freq if dur == "freq" else dur

    starts = []
    ends = []

    # If the given time range is greater than the time frequency, create smaller chunks
    # Note: substracting two ObsPy UTCDateTime objects returns value in seconds
    if pd.Timedelta(seconds=tend-tstart) > pd.Timedelta(freq):
        # Create smaller time chunks
        t1 = tstart
        while t1 < tend:
            t2 = min(t1 + pd.Timedelta(dur), tend)
            starts.append(t1)
            ends.append(t2)
            t1 += pd.Timedelta(freq)
    else:
        # No need to create smaller time chunks
        starts.append(tstart)
        ends.append(tend)

    return starts, ends
