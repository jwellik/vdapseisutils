import numpy as np
import pandas as pd
from obspy import UTCDateTime
import datetime
from matplotlib import dates as mdates
from pandas import Timestamp as pdTimestamp
from numpy import datetime64 as npdatetime64

# Examples
A = UTCDateTime(1980, 5, 18)  # 1980-05-18T00:00:00.000000Z <obspy.swarmmpl.utcdatetime.UTCDateTime>
B = datetime.datetime(1980, 5, 18)  # datetime.datetime(1980, 5, 18, 0, 0) <datetime.datetime>
C = mdates.datestr2num("1980/05/18")  # 3790 <numpy.float64>
D = pdTimestamp("1980/05/18")  #  Timestamp('1980-05-18 00:00:00') <pandas._libs.tslibs.timestamps.Timestamp>
E = npdatetime64("1980-05-18")  # numpy.datetime64('1980-05-18') <numpy.datetime64>
F = "1980/05/18"  # '1980/05/18' <str>


def convert_timeformat_dep(input, from_format, to_format):

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


def convert_timeformat(input, format="UTCDateTime"):
    """CONVERT_TIMEFORMAT Converts any time object to the object type specified by 'format'

    The input time must be in a format undertsood by or convertible by UTCDateTime (this is most things).
    """

    output = []
    if isinstance(input, list):
        pass
    else:
        input = [input]

    # First, convert strings to UTCDateTime objects
    if isinstance(input[0], str):
        input = [UTCDateTime(t) for t in input]

    # Input is ObsPy UTCDateTime
    if isinstance(input[0], UTCDateTime):
        if format.lower() == "UTCDateTime".lower():
            output = [UTCDateTime(t) for t in input]
        elif format.lower() == "pdTimestamp".lower():  # Pandas Timestamp
            output = [pd.Timestamp(UTCDateTime(t)) for t in input]
        elif format.lower() == "matplotlib".lower():  # Matplotlib date
            output = [d.matplotlib_date for d in input]
        elif format.lower() == "datetime".lower():  # Python Datetime object
            output = [UTCDateTime(t).datetime for t in input]
        elif format.lower() == "timestamp".lower():  # returns UTC timestamp in seconds
            output = [UTCDateTime(t).timestamp for t in input]
        elif format.lower() == "datetime64".lower():  # returns UTC timestamp in seconds
            output = [np.datetime64(UTCDateTime(t)) for t in input]
        else:
            raise Exception("Output time format ({}) not understood.".format(output))

    else:
        raise Exception("Input time format ({}) not understood or supported.".format(type(input)))

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
    starts = convert_timeformat(starts, output=dtype) + buffer[0]
    ends = convert_timeformat(ends, output=dtype) + buffer[0]

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


from zoneinfo import ZoneInfo

observatory_timezones = {
    "TGS": ZoneInfo("Pacific/Tongatapu"),  # UTC+17:00
    # "": ZoneInfo("Pacific/Fiji"),  #
    # "Vanuatu": ZoneInfo(""),  # UTC+11:00
    "PHIVOLCS": ZoneInfo("Asia/Manila"),  # UTC+08:00
    "CVGHM/East": ZoneInfo("Asia/Singapore"),  # Bali and east UTC+08:00
    "CVGHM/West": ZoneInfo("Asia/Jakarta"),  # Sumatra to Java UTC+07:00
    "CVGHM": ZoneInfo("Asia/Jakarta"),  # UTC+07:00
    # "": ZoneInfo(""),  # UTC+04:00  La Reunion
    "INGV": ZoneInfo("Europe/Rome"),  # UTC+02:00
    "Goma": ZoneInfo("Africa/Kigali"),  # UTC+02:00  ? Does Kigali work?
    "Iceland": ZoneInfo("Atlantic/Reykjavik"),  # UTC+00:00
    "OAVV": ZoneInfo("America/Argentina/Buenos_Aires"),  # UTC-03:00
    "SERNAGEOMIN": ZoneInfo("America/Santiago"),  # UTC-03:00, UTC-04:00
    "OVDAS": ZoneInfo("America/Santiago"),  # UTC-03:00, UTC-04:00
    "SGC": ZoneInfo("America/Bogota"),  # UTC-05:00
    "IG": ZoneInfo("America/Lima"),  # UTC-05:00 Ecuador
    "IGP": ZoneInfo("America/Lima"),  # UTC-05:00
    "Mexico": ZoneInfo("America/Mexico_City"),  # UTC-06:00
    "UNAM": ZoneInfo("America/Mexico_City"),  # UTC-06:00
    "INSIVUMEH": ZoneInfo("America/Guatemala"),  # UTC-06:00
    "MARN": ZoneInfo("America/El_Salvador"),  # UTC-06:00
    "OVSICORI": ZoneInfo("America/Costa_Rica"),  # UTC-06:00
    "USGS/Yellowstone": ZoneInfo("America/Denver"),  # UTC-06:00, UTC-07:00
    "USGS/Cascades": ZoneInfo("America/Los_Angeles"),  # UTC-07:00, UTC-08:00
    "USGS/Cascade": ZoneInfo("America/Los_Angeles"),  # UTC-07:00, UTC-08:00
    "USGS/California": ZoneInfo("America/Los_Angeles"),  # UTC-07:00, UTC-08:00
    "USGS/Alaska": ZoneInfo("America/Anchorage"),  # UTC-08:00, UTC-09:00
    "USGS/Hawaii": ZoneInfo("Pacific/Honolulu"),  # UTC-09:00, UTC-10:00
    "USGS/Hawaiian": ZoneInfo("Pacific/Honolulu"),  # UTC-09:00, UTC-10:00
    # Montserrat Volcano Observatory, West Indies SRI
    # Kamchatka Volcanic Eruption Response Team KVERT
    # Piton de la Fournaise (IPGP)
    # Martinique (IPGP)
    # Guadeloupe (IPGP)
    # Vesuvius
    # Etna
    # VAAC/Anchorage
    # VAAC/Buenos_Aires
    # VAAC/Darwim
    # VAAC/London
    # VAAC/Montreal
    # VAAC/Tokyo
    # VAAC/Toulouse
    # # VAAC/DC
    # VAAC/Washington
    # VAAC/Wellington
    # Comoros - Karthala
    # Cape Verde
    # Cameron
    # Japan: Asama, Aso, Kirishima, Kusatsu, Sakurajima, USU, ERI
    # Mexico: Chiapas, Colima, Popo, CENAPRED, UNAM
    # Nicaragua
    # PNG
    # Solomon Islands
    # Spain/Canary_Islands
}


def parse_timezone(input):
    """
    PARSE_TIMEZONE Returns a datetime.timezone object from a variety of inputs, including
     - an integer
     - a string recognized by ZoneInfo (e.g., "Pacific/Honolulu")
     - a datetime.timedelta object
     - a datetime.timezone object (returns itself)
    """

    if isinstance(input, int):
        # Convert integer offset (hours) to a timedelta and then to a timezone
        return datetime.timezone(datetime.timedelta(hours=input))
    elif isinstance(input, datetime.timedelta):
        # Convert timedelta directly to a timezone
        return datetime.timezone(input)
    elif isinstance(input, datetime.timezone):
        # If it's already a timezone object, return it as is
        return input
    elif isinstance(input, str):
        try:
            # Try to parse string with ZoneInfo
            zone = ZoneInfo(input)
            # Get the UTC offset in hours for the provided zone
            utcoffset = zone.utcoffset(datetime.datetime.now())
            if utcoffset is None:
                raise ValueError(f"Invalid timezone string: {input}")
            return datetime.timezone(utcoffset)
        except Exception as e:
            raise ValueError(f"Invalid timezone string: {e}")
    else:
        raise ValueError("Offset must be an int, datetime.timedelta, datetime.timezone, or a valid timezone string")
