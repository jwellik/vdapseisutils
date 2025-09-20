from obspy import UTCDateTime
import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
import numpy as np

# Observatory timezone mapping
OBSERVATORY_TIMEZONES = {
    "TGS": ZoneInfo("Pacific/Tongatapu"),  # UTC+17:00
    "PHIVOLCS": ZoneInfo("Asia/Manila"),  # UTC+08:00
    "CVGHM/East": ZoneInfo("Asia/Singapore"),  # Bali and east UTC+08:00
    "CVGHM/West": ZoneInfo("Asia/Jakarta"),  # Sumatra to Java UTC+07:00
    "CVGHM": ZoneInfo("Asia/Jakarta"),  # UTC+07:00
    "INGV": ZoneInfo("Europe/Rome"),  # UTC+02:00
    "Goma": ZoneInfo("Africa/Kigali"),  # UTC+02:00
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
}


class VUTCDateTime(UTCDateTime):
    """
    Extended UTCDateTime class for volcano seismology workflows.
    
    Inherits all functionality from ObsPy's UTCDateTime and adds
    volcano seismology-specific methods and utilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize VUTCDateTime with same parameters as UTCDateTime."""
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_utcdatetime(cls, utc_datetime):
        """Create VUTCDateTime from existing UTCDateTime object."""
        return cls(utc_datetime)
    
    def to_utcdatetime(self):
        """Convert back to standard UTCDateTime."""
        return UTCDateTime(self)
    
    # =================================================================
    # Time Range Utilities
    # =================================================================
    
    @classmethod
    def smart_range(cls, t1=None, t2=None, default_minutes=10, round_to_minutes=None):
        """
        Create a smart time range with intelligent defaults and rounding.
        
        Parameters
        ----------
        t1, t2 : str, UTCDateTime, or None
            Start and end times. If None, uses intelligent defaults.
        default_minutes : int, default 10
            Default duration in minutes when only one time is provided.
        round_to_minutes : int, optional
            Round times to nearest N minutes (e.g., 10 for 10-minute intervals).
            
        Returns
        -------
        tuple : (VUTCDateTime, VUTCDateTime)
            Start and end times as VUTCDateTime objects.
        """
        now = cls.utcnow()
        
        # Handle None values
        t1 = None if t1 == "None" else t1
        t2 = None if t2 == "None" else t2
        
        # Convert to VUTCDateTime if provided
        if t1 is not None:
            t1 = cls(t1)
        if t2 is not None:
            t2 = cls(t2)
        
        # Apply intelligent defaults
        if t1 is None and t2 is None:
            # No times provided - use recent time window
            t2 = now.floor_minutes(round_to_minutes) if round_to_minutes else now
            t1 = t2 - datetime.timedelta(minutes=default_minutes)
        elif t1 is None:
            # Only t2 provided - use default duration before t2
            t2 = t2.floor_minutes(round_to_minutes) if round_to_minutes else t2
            t1 = t2 - datetime.timedelta(minutes=default_minutes)
        elif t2 is None:
            # Only t1 provided - use default duration after t1
            t1 = t1.floor_minutes(round_to_minutes) if round_to_minutes else t1
            t2 = t1 + datetime.timedelta(minutes=default_minutes)
        else:
            # Both times provided - just round if requested
            if round_to_minutes:
                t1 = t1.floor_minutes(round_to_minutes)
                t2 = t2.ceil_minutes(round_to_minutes)
        
        # Ensure t1 < t2
        if t1 > t2:
            t1, t2 = t2, t1
        
        return t1, t2
    
    def floor_minutes(self, minutes):
        """Round down to nearest N-minute interval."""
        if minutes is None:
            return self
        
        # Calculate seconds to subtract
        seconds_to_subtract = (self.second + self.minute * 60) % (minutes * 60)
        return self - seconds_to_subtract
    
    def ceil_minutes(self, minutes):
        """Round up to nearest N-minute interval."""
        if minutes is None:
            return self
        
        floored = self.floor_minutes(minutes)
        if floored < self:
            return floored + datetime.timedelta(minutes=minutes)
        return floored
    
    def round_minutes(self, minutes):
        """Round to nearest N-minute interval."""
        if minutes is None:
            return self
        
        floored = self.floor_minutes(minutes)
        ceiling = floored + datetime.timedelta(minutes=minutes)
        
        # Determine which is closer
        if abs(self - floored) < abs(self - ceiling):
            return floored
        else:
            return ceiling
    
    # =================================================================
    # Timezone Utilities
    # =================================================================
    
    def to_local_time(self, lat, lon, time=None):
        """
        Convert to local time for a given latitude/longitude.
        
        Parameters
        ----------
        lat, lon : float
            Latitude and longitude for timezone detection.
        time : datetime, optional
            Time to use for timezone detection (defaults to self).
            
        Returns
        -------
        datetime
            Local time as timezone-aware datetime object.
        """
        if time is None:
            time = self.datetime
        
        # Find timezone for location
        tf = TimezoneFinder(in_memory=True)
        tz_name = tf.timezone_at(lat=lat, lng=lon)
        
        if tz_name is None:
            raise ValueError(f"Could not determine timezone for lat={lat}, lon={lon}")
        
        # Create timezone-aware datetime
        tz = ZoneInfo(tz_name)
        local_dt = time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        
        return local_dt
    
    def local_strftime(self, lat, lon, format_str="%Y-%m-%d %H:%M:%S %Z"):
        """
        Format time in local timezone for a given location.
        
        Parameters
        ----------
        lat, lon : float
            Latitude and longitude for timezone detection.
        format_str : str
            Format string for strftime.
            
        Returns
        -------
        str
            Formatted local time string.
        """
        local_dt = self.to_local_time(lat, lon)
        return local_dt.strftime(format_str)
    
    def to_observatory_time(self, observatory):
        """
        Convert to local time for a named observatory.
        
        Parameters
        ----------
        observatory : str
            Observatory name (e.g., "USGS/Hawaii", "PHIVOLCS")
            
        Returns
        -------
        datetime
            Local time as timezone-aware datetime object.
        """
        if observatory not in OBSERVATORY_TIMEZONES:
            raise ValueError(f"Unknown observatory: {observatory}. Available: {list(OBSERVATORY_TIMEZONES.keys())}")
        
        tz = OBSERVATORY_TIMEZONES[observatory]
        local_dt = self.datetime.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        return local_dt
    
    def observatory_strftime(self, observatory, format_str="%Y-%m-%d %H:%M:%S %Z"):
        """
        Format time in observatory's local timezone.
        
        Parameters
        ----------
        observatory : str
            Observatory name
        format_str : str
            Format string for strftime.
            
        Returns
        -------
        str
            Formatted local time string.
        """
        local_dt = self.to_observatory_time(observatory)
        return local_dt.strftime(format_str)
    
    @classmethod
    def get_observatory_timezone(cls, observatory):
        """
        Get timezone for a named observatory.
        
        Parameters
        ----------
        observatory : str
            Observatory name
            
        Returns
        -------
        ZoneInfo
            Timezone object for the observatory.
        """
        if observatory not in OBSERVATORY_TIMEZONES:
            raise ValueError(f"Unknown observatory: {observatory}. Available: {list(OBSERVATORY_TIMEZONES.keys())}")
        
        return OBSERVATORY_TIMEZONES[observatory]
    
    @classmethod
    def add_observatory_timezone(cls, name, timezone):
        """
        Add a new observatory timezone mapping.
        
        Parameters
        ----------
        name : str
            Observatory name
        timezone : str or ZoneInfo
            Timezone identifier or ZoneInfo object
        """
        if isinstance(timezone, str):
            timezone = ZoneInfo(timezone)
        OBSERVATORY_TIMEZONES[name] = timezone
    
    # =================================================================
    # Seismology-Specific Methods
    # =================================================================
    
    @property
    def julian_day(self):
        """Get Julian day number."""
        # Simple Julian day calculation
        year = self.year
        month = self.month
        day = self.day
        
        if month <= 2:
            year -= 1
            month += 12
        
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        
        # Add fractional day
        jd += (self.hour + self.minute / 60.0 + self.second / 3600.0) / 24.0
        
        return jd
    
    def phase_arrival_time(self, distance_km, phase="P", velocity_km_s=None):
        """
        Calculate approximate phase arrival time.
        
        Parameters
        ----------
        distance_km : float
            Distance in kilometers.
        phase : str, default "P"
            Phase type ("P", "S", etc.).
        velocity_km_s : float, optional
            Phase velocity in km/s. If None, uses default values.
            
        Returns
        -------
        VUTCDateTime
            Estimated arrival time.
        """
        # Default velocities (very approximate)
        if velocity_km_s is None:
            if phase.upper() == "P":
                velocity_km_s = 6.0  # km/s
            elif phase.upper() == "S":
                velocity_km_s = 3.5  # km/s
            else:
                velocity_km_s = 5.0  # km/s
        
        travel_time = distance_km / velocity_km_s
        return self + travel_time
    
    def event_window(self, pre_event_minutes=5, post_event_minutes=10):
        """
        Create a time window around an event time.
        
        Parameters
        ----------
        pre_event_minutes : int, default 5
            Minutes before event time.
        post_event_minutes : int, default 10
            Minutes after event time.
            
        Returns
        -------
        tuple : (VUTCDateTime, VUTCDateTime)
            Start and end times of the event window.
        """
        start_time = self - datetime.timedelta(minutes=pre_event_minutes)
        end_time = self + datetime.timedelta(minutes=post_event_minutes)
        return start_time, end_time
    
    def data_availability_window(self, buffer_minutes=5):
        """
        Create a time window for data availability checking.
        
        Parameters
        ----------
        buffer_minutes : int, default 5
            Buffer time in minutes.
            
        Returns
        -------
        tuple : (VUTCDateTime, VUTCDateTime)
            Start and end times for data availability check.
        """
        return self.event_window(pre_event_minutes=buffer_minutes, 
                                post_event_minutes=buffer_minutes)
    
    # =================================================================
    # Utility Methods
    # =================================================================
    
    def __repr__(self):
        """String representation."""
        return f"VUTCDateTime({self.isoformat()})"
    
    def __str__(self):
        """String representation."""
        return self.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # =================================================================
    # Format Conversion Properties
    # =================================================================
    
    @property
    def pandas_timestamp(self):
        """Convert to pandas Timestamp."""
        import pandas as pd
        return pd.Timestamp(self)
    
    @property
    def matplotlib_date(self):
        """Convert to matplotlib date number."""
        return self.matplotlib_date
    
    @property
    def datetime_obj(self):
        """Convert to Python datetime object."""
        return self.datetime
    
    @property
    def timestamp_seconds(self):
        """Convert to Unix timestamp in seconds."""
        return self.timestamp
    
    @property
    def numpy_datetime64(self):
        """Convert to numpy datetime64."""
        import numpy as np
        return np.datetime64(self)
    
    @property
    def nonlinloc_str(self):
        """Format for NonLinLoc: YYYY MM DD HH MM SS.SSSSSS"""
        return self.strftime("%Y %m %d %H %M %S.%f")[:-3]  # Remove microseconds, keep milliseconds
    
    @property
    def hypoinverse_str(self):
        """Format for HypoInverse: YYYYMMDDHHMMSS.SS"""
        return self.strftime("%Y%m%d%H%M%S.%f")[:-4]  # Keep 2 decimal places
    
    @property
    def compact_str(self):
        """Compact format: YYYYMMDD_HHMMSS"""
        return self.strftime("%Y%m%d_%H%M%S")
    
    @property
    def iso_str(self):
        """ISO format: YYYY-MM-DDTHH:MM:SS.SSSSSSZ"""
        return self.isoformat()
    
    # =================================================================
    # Format Conversion Methods
    # =================================================================
    
    def to_format(self, format_type):
        """
        Convert to specified format.
        
        Parameters
        ----------
        format_type : str
            Format type: "pandas", "matplotlib", "datetime", "timestamp", 
                        "numpy", "nonlinloc", "hypoinverse", "compact", "iso"
            
        Returns
        -------
        Various types depending on format
        """
        format_map = {
            "pandas": self.pandas_timestamp,
            "matplotlib": self.matplotlib_date,
            "datetime": self.datetime_obj,
            "timestamp": self.timestamp_seconds,
            "numpy": self.numpy_datetime64,
            "nonlinloc": self.nonlinloc_str,
            "hypoinverse": self.hypoinverse_str,
            "compact": self.compact_str,
            "iso": self.iso_str,
        }
        
        if format_type.lower() not in format_map:
            raise ValueError(f"Unknown format: {format_type}. Available: {list(format_map.keys())}")
        
        return format_map[format_type.lower()]
    
    @classmethod
    def convert_list(cls, time_list, format_type="UTCDateTime"):
        """
        Convert a list of times to specified format.
        
        Parameters
        ----------
        time_list : list
            List of time objects (strings, UTCDateTime, etc.)
        format_type : str
            Output format type
            
        Returns
        -------
        list
            List of converted times
        """
        # Convert all to VUTCDateTime first
        vutc_times = [cls(t) if not isinstance(t, cls) else t for t in time_list]
        
        if format_type.lower() == "utcdatetime":
            return vutc_times
        else:
            return [t.to_format(format_type) for t in vutc_times]


# Convenience functions
def vutcnow():
    """Get current time as VUTCDateTime."""
    return VUTCDateTime.utcnow()


def vutcrange(start, end, freq="1D"):
    """Create a range of VUTCDateTime objects."""
    from vdapseisutils.utils.timeutils import date_range
    times = date_range(start, end, freq=freq, dtype="UTCDateTime")
    return [VUTCDateTime(t) for t in times] 