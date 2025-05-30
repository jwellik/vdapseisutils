"""
Python scripts for swarmmpl earthquake catalogs at volcanoes.

Author: Jay Wellik, jwellik@vdap.org
Created: 2025 January 31

TODO plot_catalog - verify
[x] plot_tags - verrify
TODO highlight(start, stop, color, alpha, *kwargs)
[x] yticklabels - general format
[x] yticklabels - date format
[x] yticklabels - UTC Offset
[x] yticklabels - nlines, spacing, etc.
[x] Set yticklabel spacing based on hours, not lines
TODO yticklabels - extend to right axis
TODO Default behavior & Swarm behavior
TODO Suptitle needs slight alpha background
[x] Force interval to be 15, 30, 60, or divisible by 60
TODO Clip threshold
TODO Why isn't title getting added by default?
"""

import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from obspy import UTCDateTime

from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm
from vdapseisutils.style.colors import greyscale, swarm_colors, earthworm_colors, obspy_dayplot
from vdapseisutils.style import load_custom_rc
load_custom_rc("swarmmplrc")

cmap = "viridis_r"
norm = None

def round_to_minute(utc_datetime):
    if utc_datetime.second >= 30:
        return utc_datetime.replace(second=0, microsecond=0) + 60  # Round up
    else:
        return utc_datetime.replace(second=0, microsecond=0)  # Round down


class Helicorder(plt.Figure):

    """
    Differences between Helicorder and ObsPy.stream(dayplot)
    Additional functionality
    - returns a matplotlib Figure object
    - plot_catalog, plot_tags
    - color: "greyscale" (default), "swarm", "earthworm", or "obspy"

    Different default behavior
    - plot_events(*args, **kwargs) (different default behavior)
    - different yticklabels
    - set yticklabels on left/right based on timezone
    - no ylabel
    """

    name = "helicorder"

    def __init__(self, st, interval=60, color="greyscale",
                 one_bar_range=None, clip_threshold=None,
                 title=None,
                 utc_offset_left="UTC", utc_offset_right="UTC",
                 figsize=(8, 6),
                 dpi=100, **kwargs):

        # Initialize the Figure object
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        super().__init__(**kwargs)
        self.__dict__ = fig.__dict__
        self._dpi = dpi  # plt.show() produces an error without this. Problematic???

        # Save Helicorder settings
        # self.stream = st.deep_copy()
        self.stream = st.copy()

        # parse color
        if color.lower() == "greyscale":
            color = greyscale
        elif color.lower() == "swarm":
            color = swarm_colors
        elif color.lower() == "earthworm":
            color = earthworm_colors
        elif color.lower() == "obspy":
            color = obspy_dayplot
        else:
            color = color

        # Allow starttime/endtime in case you want it to be greater than extent of Stream
        self.starttime = kwargs.get('starttime', None)
        self.endtime = kwargs.get('endtime', None)
        if not self.starttime:
            self.starttime = min([trace.stats.starttime for trace in self.stream])
        if not self.endtime:
            self.endtime = max([trace.stats.endtime for trace in self.stream])
        # self.stream.trim(self.starttime, self.endtime)
        self.stream.trim(self.starttime, self.endtime, pad=True)  # JJW: Pad the Stream
        
        # force interval to be 15, 30, 60, or a multiple of 60
        if interval <= 15:
            self.line_len_min =  15
        elif interval <= 30:
            self.line_len_min = 30
        elif interval <= 60:
            self.line_len_min = 60
        else:
            self.line_len_min = math.ceil(interval / 60) * 60  # Round up to the next multiple of 60
        if self.line_len_min != interval:
            print("INTERVAL rounded from {} to {}. Value must be 15, 30, or a multiple of 60.".format(interval, self.line_len_min))
        self.interval = 60 * self.line_len_min  # save as seconds, thus: 60 seconds * minutes

        self.size = figsize
        self.width, self.height = self.size  # Does not consider dpi
        self.one_bar_range = one_bar_range  # JJW: Not currently used
        self.clip_threshold = clip_threshold  # JJW: Not currently used?
        self.title = kwargs.get('title', st[0].id)

        # self.extreme_values.shape[0] --> intervals?
        # self.interval (line_len_minutes)--> length of line (user provides minutes; stored as seconds)
        # self.intervals (nlines) --> # of lines on the helicorder
        # self.repeat (label_spacing) --> # of lines of spacing between yticks

        # Create plot with ObsPy dayplot
        st.plot(type="dayplot", fig=fig,
                interval=self.line_len_min, color=color,
                # show_y_UTC_label=True,
                title=title)

        # important values for yticklabel spacing
        self.nlines = len(self.axes[0].get_lines())  # Number of lines on helicorder
        self.label_spacing = kwargs.get('label_spacing', 4)  # hours in between yticklabels

        # set axes labes and tick labels
        fig.axes[0].set_ylabel("")
        fig.axes[0].set_xlabel("")
        self.set_tticks()  # By default, this adds timezone ticklabels at the bottom too

    def info(self):
        print("::: HELICORDER :::")
        print(self.properties)
        print()

    def plot_catalog(self, catalog, marker="o", color="red", markersize=8, markeredgecolor="black", alpha=0.5, **kwargs):

        # ax = self.fig.axes[0]
        ax = self.axes[0]

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
        ax.plot(x_pos, y_pos, marker, color=color, alpha=alpha, markersize=markersize, markeredgecolor=markeredgecolor, **kwargs)

    def plot_events(self, events, *args, **kwargs):
        ### PLOT_EVENTS Plot events in Catalog (not quite the same as ObsPy functionality)

        for event in events:
            self._plot_event(event, *args, **kwargs)

    def plot_tags(self, times: object, marker: object = "o", color: object = "red",
                  markersize: object = 15, markeredgecolor: object = "black",
                  linewidth = 0,
                  verbose: object = False,
                  **kwargs: object) -> object:

        # ax = self.fig.axes[0]
        ax = self.axes[0]

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

        ax.plot(x_pos, y_pos, marker=marker, color=color,
                markersize=markersize, markeredgecolor=markeredgecolor,
                linewidth=linewidth, **kwargs)

    def highlight(self, start_end_times, color="yellow", alpha=0.7, **kwargs):
        """

        :param start_end_times: list of start and time tuples
        :param color:
        :param alpha:
        :param kwargs:
        :return:
        """

        import matplotlib.patches as patches

        for times in start_end_times:

            x0, y0 = self._time2xy(times[0])
            x1, y1 = self._time2xy(times[1])

            yvals = np.arange(y0, y1 - 1, -1)  # create array of y values
            xvals0 = np.zeros(np.shape(yvals))  # create array of zeros
            xvals1 = np.zeros(np.shape(yvals)) + self.width * self._dpi
            xvals0[0] = x0
            xvals1[-1] = x1
            widths = xvals1 - xvals0

            for x, y, w in zip(xvals0, yvals, widths):
                # Add -0.5 + 0.05 to y so that y value is just a little above bottom of line
                # make height=0.9 so that it doesn't take up quite the whole line
                rect = patches.Rectangle((x, y - 0.5 + 0.05), w, 0.9, edgecolor=color, facecolor=color, alpha=alpha, **kwargs)
                self.axes[0].add_patch(rect)


    def _time2xy(self, time):
        """JJW: Stolen from ObsPy source code (https://docs.obspy.org/_modules/obspy/imaging/waveform.html#WaveformPlotting.plot_day)"""
        """
            frac = (time - self.starttime) / (self.endtime - self.starttime)
            int_frac = (self.interval) / (self.endtime - self.starttime)
            event_frac = frac / int_frac
            y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
            x_pos = (event_frac - int(event_frac)) * self.width
        """

        frac = (time - self.starttime) / (self.endtime - self.starttime)  # time as frac of helicorder
        int_frac = (self.interval) / (self.endtime - self.starttime)  # interval as frac of helicorder
        event_frac = frac / int_frac  # frac of y value of time of helicorder
        # y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
        y_pos = self.nlines - int(event_frac) - 0.5
        x_pos = (event_frac - int(event_frac)) * self.width * self._dpi

        return x_pos, y_pos

    def _time2xy_multi(self, times):
        """JJW: Stolen from ObsPy source code (https://docs.obspy.org/_modules/obspy/imaging/waveform.html#WaveformPlotting.plot_day)"""

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
                # y_pos.append(self.extreme_values.shape[0] - int(event_frac) - 0.5)
                y_pos.append(self.nlines - int(event_frac) - 0.5)
                x_pos.append((event_frac - int(event_frac)) * self.width)
        return x_pos, y_pos

    def _plot_event(self, event, marker="*", color="yellow", markersize=12, **kwargs):
        """
        Helper function to plot an event into the dayplot.
        JJW: Stolen from ObsPy source code (https://docs.obspy.org/_modules/obspy/imaging/waveform.html#WaveformPlotting.plot_day)
        """

        from obspy.geodetics import FlinnEngdahl

        # ax = self.fig.axes[0]
        ax = self.axes[0]
        seed_id = self.stream[0].id
        if hasattr(event, "preferred_origin"):
            # Get the time from the preferred origin, alternatively the first
            # origin.
            origin = event.preferred_origin()
            if origin is None:
                if event.origins:
                    origin = event.origins[0]
                else:
                    return
            time = origin.time

            # Do the same for the magnitude.
            mag = event.preferred_magnitude()
            if mag is None:
                if event.magnitudes:
                    mag = event.magnitudes[0]
            if mag is None:
                mag = ""
            else:
                mag = "%.1f %s" % (mag.mag, mag.magnitude_type)

            region = FlinnEngdahl().get_region(origin.longitude,
                                               origin.latitude)
            text = region
            if mag:
                text += ", %s" % mag
        else:
            time = event["time"]
            text = event["text"] if "text" in event else None

        # Nothing to do if the event is not on the plot.
        if time < self.starttime or time > self.endtime:
            return

        # Now find the position of the event in plot coordinates.
        x_pos, y_pos = self._time2xy(time)

        if text:
            # Some logic to get a somewhat sane positioning of the annotation
            # box and the arrow..
            text_offset_x = 0.10 * self.width
            text_offset_y = 1.00
            # Relpos determines the connection of the arrow on the box in
            # relative coordinates.
            relpos = [0.0, 0.5]
            # Arc strength is the amount of bending of the arrow.
            arc_strength = 0.25
            if x_pos < (self.width / 2.0):
                text_offset_x_sign = 1.0
                ha = "left"
                # Arc sign determines the direction of bending.
                arc_sign = "+"
            else:
                text_offset_x_sign = -1.0
                ha = "right"
                relpos[0] = 1.0
                arc_sign = "-"
            if y_pos < (self.extreme_values.shape[0] / 2.0):
                text_offset_y_sign = 1.0
                va = "bottom"
            else:
                text_offset_y_sign = -1.0
                va = "top"
                if arc_sign == "-":
                    arc_sign = "+"
                else:
                    arc_sign = "-"

            # Draw the annotation including box.
            ax.annotate(text,
                        # The position of the event.
                        xy=(x_pos, y_pos),
                        # The position of the text, offset depending on the
                        # previously calculated variables.
                        xytext=(x_pos + text_offset_x_sign * text_offset_x,
                                y_pos + text_offset_y_sign * text_offset_y),
                        # Everything in data coordinates.
                        xycoords="data", textcoords="data",
                        # Set the text alignment.
                        ha=ha, va=va,
                        # Text box style.
                        bbox=dict(boxstyle="round", fc="w", alpha=0.6),
                        # Arrow style
                        arrowprops=dict(
                            arrowstyle="-",
                            connectionstyle="arc3, rad=%s%.1f" % (
                                arc_sign, arc_strength),
                            relpos=relpos, shrinkB=7),
                        zorder=10)
        # Draw the actual point. Use a marker with a star shape.
        ax.plot(x_pos, y_pos, marker, color=color,
                markersize=markersize, **kwargs)

        for pick in getattr(event, 'picks', []):
            # check that observatory/station/location matches
            if pick.waveform_id.get_seed_string().split(".")[:-1] != \
               seed_id.split(".")[:-1]:
                continue
            x_pos, y_pos = self._time2xy(pick.time)
            ax.plot(x_pos, y_pos, "|", color="red",
                    # markersize=50, markeredgewidth=self.linewidth * 4)
                    markersize=50, markeredgewidth=4)

    def set_tticks(self, label_spacing=None, utc_offset=0, axes="left", update_tzlabels=True):
        """
        Set yticks and yticklabels for the helicorder

        label_spacing : puts a yticklabel every nth hour
        """

        # Update helicorder label_spacing, if provided
        if label_spacing:
            self.label_spacing = label_spacing

        # Create yticks and get timestamps for every line on helicorder
        yticks = np.arange(self.nlines, 0, -1, dtype=float)  # get y value for every line
        yticks -= 0.5  # Subtract -0.5 so that the label is centered on the line
        tticks_left = [(self.endtime - (i + 0.5) * self.interval) for i in yticks]  # timestamp for every line (left)
        tticks_right = [(self.endtime - (i + 1.5) * self.interval) for i in yticks]  # timestamp for every line (right)
        tticks_left = [round_to_minute(time) for time in tticks_left]  # Round UTCDateTime objects to the nearest minute
        tticks_right = [round_to_minute(time) for time in tticks_right]

        # Find the times that lie on every nth hour starting at midnight
        subset = []
        indices = []

        for i, time in enumerate(tticks_left):
            # Check if the hour is divisible by n (e.g., 0, 4, 8, etc.)
            if time.hour % self.label_spacing == 0 and time.minute == 0:
                subset.append(time)
                indices.append(i)
        yticks = yticks[indices]
        tticks_left = np.array(tticks_left)[indices]  # convert to array so that we can use indices
        tticks_right = np.array(tticks_right)[indices]

        tticklabels_left = self._format_tticklabels(tticks_left)
        tticklabels_right = self._format_tticklabels(tticks_right)

        self.axes[0].set_yticks(yticks)
        self.axes[0].set_yticklabels(tticklabels_left)

        if update_tzlabels:
            self.set_tzticklabel(utc_offset=utc_offset, axes=axes)

    def set_tzticklabel(self, custom=None, utc_offset=0, axes="left"):
        """
        Set timezone label at the bottom of the axes. Default: "UTC+00:00", "UTC-08:00", etc.
        Use 'custom' to override default behavior

        Timezone label should be parallel with the bottom of the axes; y-value -0.5.
        It will hang a little lower than the last line of seismic data.
        """

        # get existing ticks and ticklabels
        yticks = self.axes[0].get_yticks()
        yticklabels = self.axes[0].get_yticklabels()

        if -0.5 in yticks:  # Check if a tick exists at the bottom of the helicorder
            idx = yticks.index(-0.5)  # Find the correct index of the label (should also be 0)
        else:  # else, add a zero tick at the end and specify the index
            yticks = np.append(yticks, -0.5)
            yticklabels = np.append(yticklabels, "")
            idx = -1
        if custom:
            yticklabels[idx] = custom  # set custom label
        else:
            yticklabels[idx] = f"UTC{utc_offset:+03}:00"  # default: zero-padded signed integer, 3 spaces for all of it

        # Reset all yticks and yticklabels
        self.axes[0].set_yticks(yticks)
        self.axes[0].set_yticklabels(yticklabels)

    def _format_tticklabels(self, ticktimes):
        """Formats y-axis tick labels according to tick times.

        Top tick always formatted as YYYY/MM/DD
        Subsequent new dates on the date are also YYYY/MM/DD
        New dates not on the date are YYYY/MM/DD HH:MM
        All other times are HH:MM

        """

        ticktimes = [t.datetime for t in ticktimes]  # convert UTCDateTime to datetime objects

        formatted_labels = []
        previous_date = None

        for i, dt in enumerate(ticktimes):
            current_date = dt.date()
            is_new_date = (i == 0) or (current_date != previous_date)  # first line or new date

            if is_new_date and dt.hour == 0 and dt.minute == 0:
                label = dt.strftime("%Y/%m/%d")
            elif is_new_date:
                label = dt.strftime("%Y/%m/%d\n%H:%M")
            else:
                label = dt.strftime("%H:%M")

            formatted_labels.append(label)
            previous_date = current_date

        return formatted_labels
