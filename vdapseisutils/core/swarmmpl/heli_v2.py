import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.waveform import WaveformPlotting
from obspy import UTCDateTime


# Colors used by Swarm and other software
# - swarm (https://volcanoes.usgs.gov/software/swarm/download.shtml)
# - obspy (https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.plot.html)
swarm_colors_hex = [
    "#0000ff",
    "#0000cd",
    "#00009b",
    "#000069",
]
swarm_colors_rgba = [
    (0, 0, 255, 255),
    (0, 0, 205, 255),
    (0, 0, 155, 255),
    (0, 0, 105, 255),
]
greyscale_hex = [
    # "#D0D3D4", #"#D7DBDD",
    # "#B3B6B7", #"#E5E7E9",
    # "#979A9A", #"#F2F3F4",
    # "#7B7D7D", #"#F8F9F9",
    "#757575",
    "#616161",
    "#424242",
    "#212121",
]
# earthworm_colors_hex = ['#B2000F', '#004C12', 'black', '#0E01FF']  # Colors used by Earthworm helicorder
earthworm_colors_hex = ['#B2000F', 'black', '#0E01FF']  # Colors used by Earthworm helicorder (no green)
obspy_colors_hex = ('#B2000F', '#004C12', '#847200', '#0E01FF')


def parse_timezone(input):
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


class Heli(WaveformPlotting):
    """
    Creates a Helicorder by modifying ObsPy's Stream.plot(type="dayplot") command.

    Differences from dayplot include:
    - includes additional methods such as (see method documentation for more details)
        - plot_tags : plots a custom marker at any point in time on the helicorder
        - plot_catalog : plots Events from an ObsPy Catalog with custom markers
        - highlight : highlights the lines between two dates
    - forces Stream starttime, endtime to be divisible by 'interval' minutes (see __force_heli_startstop_...() for details)
    - automatically clips values that are 7x greater than 'vertical_scaling_range' (mimics plotting in SWARM)
        - both vertical_scaling_range and clip_threshold can be set manually by passing them as keyword arguments
    - understands 'greyscale' (default), 'swarm', 'earthworm', and 'obspy' (dayplot default) as inputs for 'color'
    - changes default 'interval' from 15 to 30 (minutes)
    - changes default logic for yticks and format for yticklabels (see __default_y_ticks() for details)
    - understands various timezone definitions for left and right ytick and yticklabels

    """

    def __init__(self, stream, **kwargs):
        super().__init__(stream=stream, type="day_plot")
        # Initialize attributes specific to Heli
        self.stream_original = stream.copy()
        self.stream = stream
        self.fig = plt.figure(figsize=(8, 6))

        # Process keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Changes to default behavior
        # Vertical scaling and clipping
        self.vertical_scaling_range = kwargs.get('vertical_scaling_range', self.__auto_vertical_scaling_range(0.995))
        self.clip_threshold = kwargs.get('clip_threshold', 7.5 * self.vertical_scaling_range)  # Not currently used
        print("Vertical Scaling Range : {}".format(self.vertical_scaling_range))
        print("Clip Threshold         : {}".format(self.clip_threshold))
        print("")
        self.stream[0].data[self.stream[0].data > self.clip_threshold] = self.clip_threshold
        self.stream[0].data[self.stream[0].data < -1 * self.clip_threshold] = -1 * self.clip_threshold

        self.right_vertical_labels = False  # RIGHT_VERTICAL_LABELS IS NOT ALLOWED. SET TO FALSE.

        color = kwargs.get('color', "greyscale")
        if color == "swarm":
            # print("swarm colors used")
            self.color = swarm_colors_hex
        elif color == "earthworm":
            # print("earthworm colors used")
            self.color = earthworm_colors_hex
        elif color == "obspy":
            # print("obspy colors used")
            self.color = obspy_colors_hex
        elif color == "greyscale":
            # print("greyscale colors used")
            self.color = greyscale_hex
        else:
            # print("other colors used")
            self.color = color

        self.interval = 60 * kwargs.get('interval', 30)  # original value of interval is 15 (user specifies in minutes, object stores it as seconds)
        self.tick_format = kwargs.get('tick_format', "%H:%M")  # original value of tick_format is %H:%M:%S

        # New variables
        self.tz_left = kwargs.get('tz_left', "UTC")
        self.tz_right = kwargs.get('tz_right', None)

        # New processing
        self.__force_heli_startstop_to_be_divisible_by_interval()

        self.plot_day()
        self.__dayplot_y_ticks(tz_left=self.tz_left, tz_right=self.tz_right)

    def __force_heli_startstop_to_be_divisible_by_interval(self):
        """Pads the Stream object such that starttime, endtime are both divisble by interval"""
        original_start_time = self.stream[0].stats.starttime
        original_end_time = self.stream[0].stats.endtime
        n = self.interval / 60  # convert seconds to minutes
        new_start_time = original_start_time - (
                original_start_time.minute % n) * 60 - original_start_time.second - original_start_time.microsecond / 1e6
        new_end_time = original_end_time + (n - (
                    original_end_time.minute % n)) * 60 - original_end_time.second - original_end_time.microsecond / 1e6
        self.stream = self.stream.trim(new_start_time, new_end_time, pad=True)
        self.starttime = new_start_time
        self.endtime = new_end_time

    def __auto_vertical_scaling_range(self, threshold):
        """Sets the vertical_scaling_range to the threshold percentile of the data points"""
        # That is, threshold percent of data points will fit within the bounds of the helicorder bar

        # SWARM one_bar_range for goma_waveform.mseed = 274 (~31% of samples, or 5% of abs max value)
        # I thing that one_bar_range of 1000 looks pretty good, which means fitting 86% of the samples
        # Fitting 70% of the samples yields a one_bar_range of 832

        data = np.sort(np.abs(self.stream_original[0].data))
        return data[int(len(data) * threshold)]

    # vastly modified from original code
    def __dayplot_y_ticks(self, tz_left="UTC", tz_right=None):  # @UnusedVariable
        """

        Produces y_ticklabels similar to the default ObsPy day_plot except:
        - ticks and ticklabels can be added to the left and right by defining tz_left (default="UTC") and tz_right (default=None)
        - tz_left & tz_right can be specified as an...
            - integer (hours different from UTC)
            - datetime.timedelta object
            - datetime.timezone object
            - str formatted as an IANA code understood by ZoneInfo (e.g., "Asia/Singapore")
        - the top tick label is always the date ('%Y/%m/%d')
        - the bottom tick label is the UTC offset (e.g., 'UTC+08:00')
        - the ylabel is removed

        :param tz_left: a IANA Time Zone code or datetime.timedelta or datetime.timezone object (default 'UTC')
        :param tz_right: Same as tz_left. Default: None
        :return:
        """

        # Define tz_left and tz_right
        # datetime.timezone object or an empty list, which can be looped over later
        tz_left = parse_timezone(tz_left) if tz_left is not None else []
        tz_right = parse_timezone(tz_right) if tz_right is not None else []

        # Sets the yticks for the dayplot.
        intervals = self.extreme_values.shape[0]  # Number of lines on the helicorder
        heli_start = self.stream[0].stats.starttime
        heli_end = self.stream[0].stats.endtime
        duration = round((heli_end - heli_start) / 3600)  # in hours

        # Define tick_interval in hours
        if duration <= 12:
            tick_interval = 3  # duration <= 12 hours --> every 3 hours
        elif duration <= 36:
            tick_interval = 6  # duration <= 36 hours --> every 6 hours
        elif duration <= 72:
            tick_interval = 12  # duration <= 72 hours --> every 12 hours
        elif duration <= 5 * 24:
            tick_interval = 24  # duration <= 5 days --> every day
        else:
            tick_interval = (5 * 24) // (duration // (5 * 24))  # duration > 5 days --> every ndays//5

        # Set up a twin axis on the right side of the plot (it may not get used)
        self.y_axis = [self.axis[0], self.axis[0].twinx()]
        yrange = self.axis[0].get_ylim()  # (bottom, top)
        self.y_axis[0].set_ylim(yrange)
        self.y_axis[0].set_ylabel('')

        # Create ticks and ticklabels for each timezone y axis
        for j, tz in enumerate([tz_left, tz_right]):

            if tz:

                # create a tick for every line
                tick_steps = list(range(1, intervals))
                ticks = np.arange(intervals, 1, -1, dtype=float)
                ticks -= 0.5

                # Define ticktimes for everyline based on specified timezone
                ticktimes = [(self.starttime + (i + j) * self.interval) for i in tick_steps]
                ticktimes = [t.datetime.replace(tzinfo=ZoneInfo("UTC")) for t in ticktimes]  # make timezone aware (UTC) (UTCDateTime objects)
                ticktimes = [t.astimezone(tz) for t in ticktimes]  # convert to specific timezone (datetime.datetime objects)

                # limit ticks to those that occur at proper interval
                idx = [i for i, t in enumerate(ticktimes) if t.hour % tick_interval == 0 and t.minute == 0 and t.second == 0]
                # idx = sorted(list(set([0] + idx)))  # always include a tick at top line; this adds those indices but ensures there are no duplicates
                ticks = ticks[idx]
                ticktimes = [ticktimes[i] for i in idx]

                # Create y tick labels from proper timezone formatting
                y_ticklabels = [
                    t.strftime("%Y/%m/%d") if t.hour == 0 and t.minute == 0 else t.strftime(self.tick_format)
                    for t in ticktimes
                ]  # converts datetime.datetime objects to str
                utc_delta = ticktimes[0].utcoffset()  # Get the UTC offset as a datetime.timedelta object

                # Add tick and ticklabel at very top and bottom
                ticks = [yrange[1]] + list(ticks) + [yrange[0]]
                heli_start_local = heli_start.datetime.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
                top_ticklabel = heli_start_local.strftime("%Y/%m/%d")
                bottom_ticklabel = "\nUTC{hours:+03d}:{minutes:02d}".format(hours=utc_delta.seconds // 3600, minutes=(utc_delta.seconds // 60) % 60)  # append utc offset as (e.g.) +00:00 to new line on bottom label
                y_ticklabels = [top_ticklabel] + y_ticklabels + [bottom_ticklabel]

                self.y_axis[j].set_yticks(ticks)
                self.y_axis[j].set_yticklabels(y_ticklabels, size=self.y_labels_size)

            else:

                self.y_axis[j].set_yticks([])

        # print("::: INFO :::")
        # print("Duration      : {} hours".format(duration))
        # print("Tick Interval : {} hours".format(tick_interval))
        # print("n Ticks       : {}".format(len(ticks)))
        # print("::::::::::::")
        # print()

    # This code is burried in obspy.imaging.waveform.WaveformPlotting.plot_event(),
    # so I had to steal it and put it here.
    # def _time2xy(self, time):
    #     frac = (time - self.starttime) / (self.endtime - self.starttime)
    #     int_frac = (self.interval) / (self.endtime - self.starttime)
    #     event_frac = frac / int_frac
    #     y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
    #     x_pos = (event_frac - int(event_frac)) * self.width
    #     return x_pos, y_pos

    # This code is burried in obspy.imaging.waveform.WaveformPlotting.plot_event(),
    # so I had to steal it and put it here.
    # I also modified it to take a list of times and return a list of x and y positions
    def _time2xy(self, times):

        x_pos = []
        y_pos = []

        if type(times) is not list:
            times = [times]

        for time in times:
            time = UTCDateTime(time)  # ensure that time is a UTCDateTime
            # Nothing to do if the event is not on the plot. # Excludes points not within time range
            if time < self.starttime or time > self.endtime:
                pass
            else:
                frac = (time - self.starttime) / (self.endtime - self.starttime)
                int_frac = (self.interval) / (self.endtime - self.starttime)
                event_frac = frac / int_frac
                y_pos.append(self.extreme_values.shape[0] - int(event_frac) - 0.5)
                x_pos.append((event_frac - int(event_frac)) * self.width)
        return x_pos, y_pos

    # Plot times with custom markers on the helicorder
    def plot_tags(self, times: object, marker: object = "o", color: object = "red",
                  markersize: object = 8, markeredgecolor: object = "black",
                  verbose: object = False, **kwargs: object) -> object:

        if type(times) is not list:
            times = [times]

        x_pos = []
        y_pos = []
        for time in times:

            if verbose:
                print("Plotting time: {}".format(time))  # For troubleshooting
            time = UTCDateTime(time)

            # Nothing to do if the event is not on the plot.
            if time < self.starttime or time > self.endtime:
                return

            # Now find the position of the event in plot coordinates.
            x, y = self._time2xy(time)
            x_pos.append(x)
            y_pos.append(y)

        self.axis[0].plot(x_pos, y_pos, marker, color=color,
                markersize=markersize, linewidth=self.linewidth, markeredgecolor=markeredgecolor, **kwargs)

    # Plot an ObsPy catalog object
    def plot_catalog(self, catalog, marker="o", color="red", markersize=8, markeredgecolor="black", alpha=0.5, **kwargs):

        from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm

        ax = self.fig.axes[0]

        # catdf = prep_catalog_data_mpl(catalog, time_format="UTCDateTime")
        catdf = catalog2txyzm(catalog)

        # if s == "magnitude":
        #     s = catdf["size"]
        # else:
        #     s = s
        #
        # if c == "time":
        #     c = catdf["time"]
        # else:
        #     c = c

        x_pos, y_pos = self._time2xy_multi(catdf["time"])

        # ax.plot(x_pos, y_pos, marker, color=color, markersize=markersize, linewidth=self.linewidth, markeredgecolor=markeredgecolor)
        ax.plot(x_pos, y_pos, marker, color=color, alpha=alpha, markersize=markersize, linewidth=self.linewidth, markeredgecolor=markeredgecolor)

    # Use ObsPy's original internal _plot_event to plot an ObsPy Event object
    def plot_events(self, events):
        ### PLOT_EVENTS Plot events in Catalog (not quite the same as ObsPy functionality)

        for event in events:
            self._plot_event(event)


def main():

    # Example usage
    from obspy import read

    st = read("/Users/jwellik/PROJECTS/dev_sandbox/obspy_dayplot/goma_waveform.mseed")
    heli = Heli(stream=st)
    heli.plot_tags(UTCDateTime("2021/05/20 01:36:30"), color="yellow", markersize=8)  # plot a single time
    heli.plot_tags("2021/05/20 17:19:35", marker="|", markeredgecolor="blue", markersize=15)  # plot a single time as a P arrival
    heli.plot_tags(UTCDateTime("2021/05/20 17:25:00"), marker="|", markeredgecolor="red", markersize=15)   # plot a single time as a S arrival
    heli.plot_tags(UTCDateTime("2021/05/20 17:44:45"), marker="|", markeredgecolor="black", markersize=15)  # plot a single time as a coda end
    heli.plot_tags([UTCDateTime("2021/05/20 08:31:00"), "2021/05/20 02:08:00"], color="yellow")  # plot a list of times given in any format
    plt.show()

    st = read("/Users/jwellik/PROJECTS/dev_sandbox/obspy_dayplot/goma_waveform.mseed")
    h1 = Heli(stream=st, color="swarm", vertical_sclaing_range=1500, tz_left="UTC", tz_right=2)
    plt.show()

    st = read("/Users/jwellik/PROJECTS/dev_sandbox/obspy_dayplot/goma_waveform.mseed")
    h2 = Heli(stream=st, color="earthworm", vertical_scaling_range=1500, tz_left=2, tz_right="Africa/Cairo")
    plt.show()

    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    print("Downloading data for Hood...")
    st = client.get_waveforms("CC", "YOCR", "*", "BHZ", UTCDateTime("2021/06/28 00:04:31"), UTCDateTime("2021/06/29 23:59:01"))
    st = st.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print("Download complete.")
    h3 = Heli(stream=st, interval=60, color="greyscale", tz_left="UTC", tz_right="America/Los_Angeles")
    plt.show()

    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    print("Downloading data for Shishaldin MS-SA...")
    st = client.get_waveforms("AV", "SSBA", "*", "BHZ", UTCDateTime("2023/07/16 01:00"), UTCDateTime("2023/07/17 19:00"))
    print("Download complete.")
    h4 = Heli(stream=st, interval=60, color="swarm", tz_left="UTC", tz_right="America/Anchorage")
    plt.show()

    from obspy.clients.fdsn import Client
    client = Client("IRIS")
    print("Downloading data for Shishaldin tremor...")
    st = client.get_waveforms("AV", "SSBA", "*", "BHZ", UTCDateTime("2023/09/03 12:00"), UTCDateTime("2023/09/06"))
    print("Download complete.")
    h5 = Heli(stream=st, interval=60, color="greyscale", tz_left="UTC", tz_right="America/Anchorage")
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()

# To Do List
# TODO UTC offset is not handling negative values correctly

# IDEA LIST
# TODO set_tz_label()
# TODO self.__set_yticks(...)
#   self.__set_yticks('left', tick_locations, 'time')  # --> start at y_ticklabels = ...
#   self.__set_yticks('left', tick_locations, 'line')  # --> start from beginning ...
# TODO __set_yticklabels(ax, 'left'|'right', yticktimes)
# TODO tz_left/tz_right="local" tries to use timezone of Stream
# TODO interval should be 15', 30', or divisible by 60'
