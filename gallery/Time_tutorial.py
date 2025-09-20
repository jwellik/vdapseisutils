import datetime
from datetime import time, tzinfo, timedelta, timezone, datetime

from zoneinfo import ZoneInfo
import pytz

from obspy import UTCDateTime
from vdapseisutils.utils.timeutils import parse_timezone


def public_time_packages():

    print("Timezones with datetime and ZoneInfo....")

    CHICAGO = ZoneInfo("America/Chicago")
    CHICAGO  # --> zoneinfo.ZoneInfo(key='America/Chicago')

    dt = datetime(1988, 3, 17, 21, 40, tzinfo=ZoneInfo("America/Chicago"))
    dt.tzname()  # --> 'CST'
    dt.timetz()  # --> datetime.time(21, 40, tzinfo=zoneinfo.ZoneInfo(key='America/Chicago'))
    dt.utcoffset()  # --> datetime.timedelta(days=-1, seconds=64800)
    dt.astimezone(ZoneInfo("UTC"))  # --> datetime.datetime(1988, 3, 18, 3, 40, tzinfo=zoneinfo.ZoneInfo(key='UTC'))
    dt.strftime("%Y/%m/%d %H:%M %Z %z")  # --> '1988/03/17 21:40 CST -0600'
    print(dt)  # --> datetime.datetime(1988, 3, 17, 21, 40, tzinfo=zoneinfo.ZoneInfo(key='America/Chicago'))
    print()


    print("Timezones with datetime.timezone...")
    utcoffset = timedelta(hours=-6)  # --> datetime.timedelta(days=-1, seconds=64800)
    tz = timezone(utcoffset)  # --> datetime.timezone(datetime.timedelta(days=-1, seconds=64800))
    dt = datetime(1988, 3, 17, 21, 40, tzinfo=tz)  # --> datetime.datetime(1988, 3, 17, 21, 40, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=64800)))
    dt.tzname()  # --> 'UTC-06:00'
    dt.timetz()  # --> datetime.time(21, 40, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=64800)))
    dt.astimezone(ZoneInfo("UTC"))  # --> datetime.datetime(1988, 3, 18, 3, 40, tzinfo=zoneinfo.ZoneInfo(key='UTC'))
    dt.strftime("%Y/%m/%d %H:%M %Z %z")  # --> '1988/03/17 21:40 UTC-06:00 -0600'
    print()


    print("Timezones with UTCDateTime...")

    t = UTCDateTime("1988-03-17T21:40-08")  # --> UTCDateTime(1988, 3, 18, 5, 40) timezone is UTC-8
    t.tzname()  # --> None
    t.utcoffset()  # --> None
    t.strftime("%Y/%m/%d %H:%M %Z %z")  # --> '1988/03/18 05:40  '
    print()

    print("Localize a UTCDateTime object...")
    t = UTCDateTime("1988-03-17T21:40")  # --> UTCDateTime(1988, 3, 18, 5, 40) timezone is unaware
    t = t.datetime.replace(tzinfo=ZoneInfo("America/Chicago"))  # Localize time as USA/Central; returns datetime.datetime
    print(UTCDateTime(t))  # prints the UTC timestamp
    print(t.astimezone(ZoneInfo("UTC")))  # Convert time to UTC
    print()

    print("Timezones with pytz...")
    utc = pytz.utc
    utc.zone  # --> 'UTC'
    # utc._offset # --> datetime.timedelta(0)

    eastern = pytz.timezone("US/Central")
    eastern.zone  # --> 'US/Central'
    eastern._utcoffset  # --> datetime.timedelta(days=-1, seconds=65340)


    print("Done.")


def vdap_time_packages():

    print("Let's play w time...")

    # Timezone string
    tz = parse_timezone("Pacific/Honolulu")
    print(tz)  # --> UTC-10:00

    # Integer
    tz = parse_timezone(-10)
    print(tz)  # --> UTC-10:00

    # TimeDelta object
    tz = parse_timezone(timedelta(hours=-10))
    print(tz)  # --> UTC-10:00

    print("Done.")


if __name__ == '__main__':
    public_time_packages()
    vdap_time_packages()
