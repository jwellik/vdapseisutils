# TODO Make plot_catalog (etc) consistent w MapClass options
# TODO Make DateTick formatting consistent w other waveform swarmmpl options
# TODO Make suptitle and title option

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util import create_empty_data_chunk
from obspy.geodetics import FlinnEngdahl
from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm
from vdapseisutils.style.colors import swarm_colors_hex

MINMAX_ZOOMLEVEL_WARNING_TEXT = "Warning: Zooming into MinMax Plot!"
SECONDS_PER_DAY = 3600.0 * 24.0
DATELOCATOR_WARNING_MSG = (
    "AutoDateLocator was unable to pick an appropriate interval for this date "
    "range. It may be necessary to add an interval value to the "
    "AutoDateLocator's intervald dictionary.")


class Helicorder(object):
    """
    Class that provides several solutions for swarmmpl helicorders.

    It uses matplotlib to plot the waveforms.
    """

    def __init__(self, st, *args, **kwargs):
        """
        Extend the seismogram.
        """

        ##############################################################
        # vvvvv

        self.kwargs = kwargs
        # self.st = kwargs.get('st')
        self.stream = st
        # Check if it is a Stream or a Trace object.
        if isinstance(self.stream, Trace):
            self.stream = Stream([self.stream])
        elif not isinstance(self.stream, Stream):
            msg = 'Plotting is only supported for Stream or Trace objects.'
            raise TypeError(msg)
        # Stream object should contain at least one Trace
        if len(self.stream) < 1:
            msg = "Empty st object"
            raise IndexError(msg)
        self.stream = self.stream.copy()

        self.starttime = kwargs.get('starttime', None)
        self.endtime = kwargs.get('endtime', None)
        self.fig_obj = kwargs.get('fig', None)
        # If no times are given take the min/max values from the st object.
        if not self.starttime:
            self.starttime = min([trace.stats.starttime for trace in
                                  self.stream])
        if not self.endtime:
            self.endtime = max([trace.stats.endtime for trace in self.stream])
        self.stream.trim(self.starttime, self.endtime)  # TODO Add pad=True, if you want to make start/end longer than available data
        # Assigning values for type 'section'
        self.plotting_method = kwargs.get('method', None)
        # Below that value the data points will be plotted normally. Above it
        # the data will be plotted using a different approach (details see
        # below). Can be overwritten by the above self.plotting_method kwarg.
        # if self.type == 'section':
        #     # section may consists of hundreds of seismograms
        #     self.max_npts = 10000
        # else:
        #     self.max_npts = 400000
        # self.max_npts = 400000
        # If automerge is enabled, merge traces with the same id for the plot.
        # self.automerge = kwargs.get('automerge', True)
        # If equal_scale is enabled, all plots are equally scaled.
        # self.equal_scale = kwargs.get('equal_scale', True)
        # Set default values.
        # The default value for the size is determined dynamically because
        # there might be more than one channel to plot.
        self.size = kwargs.get('size', None)
        # Values that will be used to calculate the size of the plot.
        # self.default_width = 800
        # self.default_height_per_channel = 250
        # if not self.size:
        #     self.width = 800
        #     # Check the kind of plot.
        #     self.height = 600
        #     if self.type == 'dayplot':
        #         self.height = 600
        #     elif self.type == 'section':
        #         self.width = 1000
        #         self.height = 600
        #     else:
        #         # One plot for each trace.
        #         if self.automerge:
        #             count = self.__get_mergable_ids()
        #             count = len(count)
        #         else:
        #             count = len(self.st)
        #         self.height = count * 250
        # else:
        #     self.width, self.height = self.size
        self.size = kwargs.get('size', (8, 6))
        self.width, self.height = self.size
        # Interval length in minutes for dayplot.
        self.interval = 60 * kwargs.get('interval', 15)  # in seconds, thus: 60 seconds * minutes
        # Scaling.
        self.vertical_scaling_range = kwargs.get('vertical_scaling_range', None)
        # Dots per inch of the plot. Might be useful for printing plots.
        self.dpi = kwargs.get('dpi', 300)  # JJW modified from 100 to 300
        self.color = kwargs.get('color', swarm_colors_hex)  # Color of the graph.
        self.background_color = kwargs.get('bgcolor', 'w')
        self.face_color = kwargs.get('face_color', 'w')
        self.grid_color = kwargs.get('grid_color', 'black')
        self.grid_linewidth = kwargs.get('grid_linewidth', 0.5)
        self.grid_linestyle = kwargs.get('grid_linestyle', ':')
        # Transparency. Overwrites background and facecolor settings.
        self.transparent = kwargs.get('transparent', False)
        if self.transparent:
            self.background_color = None
        # Ticks.
        self.tick_format = kwargs.get('tick_format', '%H:%M:%S')
        self.tick_rotation = kwargs.get('tick_rotation', 0)

        # Whether or not to save a file.
        self.outfile = kwargs.get('outfile')
        self.handle = kwargs.get('handle')

        # File format of the resulting file. Usually defaults to PNG but might
        # be dependent on your matplotlib backend.
        self.format = kwargs.get('format')
        self.show = kwargs.get('show', True)
        self.draw = kwargs.get('draw', True)
        self.block = kwargs.get('block', True)

        # plot parameters options
        self.x_labels_size = kwargs.get('x_labels_size', 8)
        self.y_labels_size = kwargs.get('y_labels_size', 8)
        self.title_size = kwargs.get('title_size', 10)
        self.linewidth = kwargs.get('linewidth', 1)
        self.linestyle = kwargs.get('linestyle', '-')
        self.subplots_adjust_left = kwargs.get('subplots_adjust_left', 0.12)
        self.subplots_adjust_right = kwargs.get('subplots_adjust_right', 0.88)
        self.subplots_adjust_top = kwargs.get('subplots_adjust_top', 0.95)
        self.subplots_adjust_bottom = kwargs.get('subplots_adjust_bottom', 0.1)
        self.right_vertical_labels = kwargs.get('right_vertical_labels', False)
        self.one_tick_per_line = kwargs.get('one_tick_per_line', False)
        self.show_y_UTC_label = kwargs.get('show_y_UTC_label', True)
        self.title = kwargs.get('title', self.stream[0].id)
        self.fillcolor_pos, self.fillcolor_neg = \
            kwargs.get('fillcolors', (None, None))

        self.one_bar_range = kwargs.get('one_bar_range', None)  # Not currently used
        self.clip_threshold = kwargs.get('clip_threshold', None)

        # ^^^
        ##############################################################

        self.number_of_ticks = 5  # x ticks

        ##############################################################

        # Experiment with setting clip_threshold
        if self.clip_threshold:
            for tr in self.stream:
                tr.data[np.where(tr.data > self.clip_threshold)] = self.clip_threshold
                tr.data[np.where(tr.data < self.clip_threshold*-1)] = self.clip_threshold*-1

        # Merge and trim to pad
        self.stream.merge()
        if len(self.stream) != 1:
            msg = "All traces need to be of the same id for a dayplot"
            raise ValueError(msg)
        self.stream.trim(self.stream[0].stats.starttime, self.stream[0].stats.endtime, pad=True)
        # Get minmax array.
        self.__dayplot_get_min_max_values(self, *args, **kwargs)
        # Normalize array
        self.__dayplot_normalize_values(self, *args, **kwargs)
        # Get timezone information. If none is given, use local time.
        self.time_offset = kwargs.get(
            'time_offset',
            round((UTCDateTime(datetime.now()) - UTCDateTime()) / 3600.0, 2))
        self.timezone = kwargs.get('timezone', 'local time')
        # Try to guess how many steps are needed to advance one full time unit.
        self.repeat = None
        intervals = self.extreme_values.shape[0]
        if self.interval < 60 and 60 % self.interval == 0:
            self.repeat = 60 // self.interval
        elif self.interval < 1800 and 3600 % self.interval == 0:
            self.repeat = 3600 // self.interval
        # Otherwise use a maximum value of 10.
        else:
            # if intervals >= 10:
            #     self.repeat = 10
            if intervals >= 4:  # JJW changed from 10 to 4 for helicorders
                self.repeat = 4
            else:
                self.repeat = intervals

        self.fig = plt.figure(figsize=self.size)

    def plot(self, *args, **kwargs):

        # Create axis to plot on.
        if self.background_color:
            axis_facecolor_kwargs = dict(facecolor=self.background_color)
            ax = self.fig.add_subplot(1, 1, 1, **axis_facecolor_kwargs)
        else:
            ax = self.fig.add_subplot(1, 1, 1)
        # Adjust the subplots
        self.fig.subplots_adjust(left=self.subplots_adjust_left,
                                 right=self.subplots_adjust_right,
                                 top=self.subplots_adjust_top,
                                 bottom=self.subplots_adjust_bottom)
        # Create x_value_array.
        x_values = np.repeat(np.arange(self.width), 2)
        intervals = self.extreme_values.shape[0]

        for _i in range(intervals):
            # Create offset array.
            y_values = np.ma.empty(self.width * 2)
            y_values.fill(intervals - (_i + 1))
            # Add min and max values.
            y_values[0::2] += self.extreme_values[_i, :, 0]
            y_values[1::2] += self.extreme_values[_i, :, 1]
            # Plot the values.
            ax.plot(x_values, y_values,
                    color=self.color[_i % len(self.color)],
                    linewidth=self.linewidth, linestyle=self.linestyle)
        # Plot the scale, if required.
        scale_unit = kwargs.get("data_unit", None)
        if scale_unit is not None:
            self._plot_dayplot_scale(unit=scale_unit)
        # Set ranges.
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(-0.3, intervals + 0.3)
        self.axis = [ax]
        # Set ticks.
        self.__dayplot_set_y_ticks(*args, **kwargs)
        self.__dayplot_set_x_ticks(*args, **kwargs)
        # Choose to show grid but only on the x axis.
        self.fig.axes[0].grid(color=self.grid_color,
                              linestyle=self.grid_linestyle,
                              linewidth=self.grid_linewidth)
        self.fig.axes[0].yaxis.grid(False)
        # Set the title of the plot.
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_size)

    # def plot_catalog(self, catalog, s="magnitude", c="time", cmap="viridis_r", alpha=0.5, **kwargs):
    def plot_catalog(self, catalog, marker="o", color="red", markersize=8, markeredgecolor="black", alpha=0.5, **kwargs):

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

    def plot_events(self, events, min_magnitude=None, *args, **kwargs):
        ### PLOT_EVENTS Plot events in Catalog (not quite the same as ObsPy functionality)

        for event in events:
            self._plot_event(event)

    def plot_tags(self, times: object, marker: object = "o", color: object = "red", markersize: object = 8, markeredgecolor: object = "black", verbose: object = False,
                  **kwargs: object) -> object:

        ax = self.fig.axes[0]

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

        ax.plot(x_pos, y_pos, marker, color=color,
                markersize=markersize, linewidth=self.linewidth, markeredgecolor=markeredgecolor)

    def _time2xy(self, time):
        frac = (time - self.starttime) / (self.endtime - self.starttime)
        int_frac = (self.interval) / (self.endtime - self.starttime)
        event_frac = frac / int_frac
        y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
        x_pos = (event_frac - int(event_frac)) * self.width
        return x_pos, y_pos

    def _time2xy_multi(self, times):

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

    def _plot_event(self, event, marker="*", color="yellow", markersize=12, *args, **kwargs):
        """
        Helper function to plot an event into the dayplot.
        """
        ax = self.fig.axes[0]
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

        # def time2xy(time):
        #     frac = (time - self.starttime) / (self.endtime - self.starttime)
        #     int_frac = (self.interval) / (self.endtime - self.starttime)
        #     event_frac = frac / int_frac
        #     y_pos = self.extreme_values.shape[0] - int(event_frac) - 0.5
        #     x_pos = (event_frac - int(event_frac)) * self.width
        #     return x_pos, y_pos
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
                markersize=markersize, linewidth=self.linewidth)

        for pick in getattr(event, 'picks', []):
            # check that observatory/station/location matches
            if pick.waveform_id.get_seed_string().split(".")[:-1] != \
               seed_id.split(".")[:-1]:
                continue
            x_pos, y_pos = self._time2xy(pick.time)
            ax.plot(x_pos, y_pos, "|", color="red",
                    markersize=50, markeredgewidth=self.linewidth * 4)

    def _plot_dayplot_scale(self, unit):
        """
        Plots the dayplot scale if requested.
        """
        left = self.width
        right = left + 5
        top = 2
        bottom = top - 1

        very_right = right + (right - left)
        middle = bottom + (top - bottom) / 2.0

        verts = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
            (right, middle),
            (very_right, middle)
        ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.MOVETO,
                 Path.LINETO
                 ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, lw=1, facecolor="none")
        patch.set_clip_on(False)
        self.fig.axes[0].add_patch(patch)
        factor = self._normalization_factor
        # Manually determine the number of digits after decimal
        if factor >= 1000:
            fmt_string = "%.0f %s"
        elif factor >= 100:
            fmt_string = "%.1f %s"
        else:
            fmt_string = "%.2f %s"
        self.fig.axes[0].text(
            very_right + 3, middle,
            fmt_string % (self._normalization_factor, unit), ha="left",
            va="center", fontsize="small")

    def __dayplot_get_min_max_values(self, *args, **kwargs):  # @UnusedVariable
        """
        Takes a Stream object and calculates the min and max values for each
        pixel in the dayplot.

        Writes a three dimensional array. The first axis is the step, i.e
        number of trace, the second is the pixel in that step and the third
        contains the minimum and maximum value of the pixel.
        """
        # Helper variables for easier access.
        trace = self.stream[0]
        trace_length = len(trace.data)

        # Samples per interval.
        spi = int(self.interval * trace.stats.sampling_rate)
        # Check the approximate number of samples per pixel and raise
        # error as fit.
        spp = float(spi) / self.width
        if spp < 1.0:
            msg = """
            Too few samples to use dayplot with the given arguments.
            Adjust your arguments or use a different swarmmpl method.
            """
            msg = " ".join(msg.strip().split())
            raise ValueError(msg)
        # Number of intervals plotted.
        noi = float(trace_length) / spi
        inoi = int(round(noi))
        # Plot an extra interval if at least 2 percent of the last interval
        # will actually contain data. Do it this way to lessen floating point
        # inaccuracies.
        if abs(noi - inoi) > 2E-2:
            noi = inoi + 1
        else:
            noi = inoi

        # Adjust data. Fill with masked values in case it is necessary.
        number_of_samples = noi * spi
        delta = number_of_samples - trace_length
        if delta < 0:
            trace.data = trace.data[:number_of_samples]
        elif delta > 0:
            trace.data = np.ma.concatenate(
                [trace.data, create_empty_data_chunk(delta, trace.data.dtype)])

        # Create array for min/max values. Use masked arrays to handle gaps.
        extreme_values = np.ma.empty((noi, self.width, 2))
        trace.data.shape = (noi, spi)

        ispp = int(spp)
        fspp = spp % 1.0
        if fspp == 0.0:
            delta = None
        else:
            delta = spi - ispp * self.width

        # Loop over each interval to avoid larger errors towards the end.
        for _i in range(noi):
            if delta:
                cur_interval = trace.data[_i][:-delta]
                rest = trace.data[_i][-delta:]
            else:
                cur_interval = trace.data[_i]
            cur_interval.shape = (self.width, ispp)
            extreme_values[_i, :, 0] = cur_interval.min(axis=1)
            extreme_values[_i, :, 1] = cur_interval.max(axis=1)
            # Add the rest.
            if delta:
                extreme_values[_i, -1, 0] = min(extreme_values[_i, -1, 0],
                                                rest.min())
                extreme_values[_i, -1, 1] = max(extreme_values[_i, -1, 0],
                                                rest.max())
        # Set class variable.
        self.extreme_values = extreme_values

    def __dayplot_normalize_values(self, *args, **kwargs):  # @UnusedVariable
        """
        Normalizes all values in the 3 dimensional array, so that the minimum
        value will be 0 and the maximum value will be 1.

        It will also convert all values to floats.
        """
        # Convert to native floats.
        self.extreme_values = self.extreme_values.astype(float) * \
            self.stream[0].stats.calib
        # Make sure that the mean value is at 0
        # raises underflow warning / error for numpy 1.9
        # even though mean is 0.09
        # self.extreme_values -= self.extreme_values.mean()
        self.extreme_values -= self.extreme_values.sum() / \
            self.extreme_values.size

        # Scale so that 99.5 % of the data will fit the given range.
        if self.vertical_scaling_range is None:
            percentile_delta = 0.005
            max_values = self.extreme_values[:, :, 1].compressed()
            min_values = self.extreme_values[:, :, 0].compressed()
            # Remove masked values.
            max_values.sort()
            min_values.sort()
            length = len(max_values)
            index = int((1.0 - percentile_delta) * length)
            max_val = max_values[index]
            index = int(percentile_delta * length)
            min_val = min_values[index]
        # Exact fit.
        elif float(self.vertical_scaling_range) == 0.0:
            max_val = self.extreme_values[:, :, 1].max()
            min_val = self.extreme_values[:, :, 0].min()
        # Fit with custom range.
        else:
            max_val = min_val = abs(self.vertical_scaling_range) / 2.0

        # Normalization factor.
        self._normalization_factor = max(abs(max_val), abs(min_val)) * 2

        # Scale from 0 to 1.
        # raises underflow warning / error for numpy 1.9
        # even though normalization_factor is 2.5
        # self.extreme_values = self.extreme_values / \
        #     self._normalization_factor
        self.extreme_values = self.extreme_values * \
            (1. / self._normalization_factor)
        self.extreme_values += 0.5

    def __dayplot_set_x_ticks(self, *args, **kwargs):  # @UnusedVariable
        """
        Sets the xticks for the dayplot.
        """
        localization_dict = kwargs.get('localization_dict', {})
        localization_dict.setdefault('seconds', 'seconds')
        localization_dict.setdefault('minutes', 'minutes')
        localization_dict.setdefault('hours', 'hours')
        localization_dict.setdefault('time in', 'time in')
        max_value = self.width - 1
        # Check whether it is sec/mins/hours and convert to a universal unit.
        if self.interval < 240:
            time_type = localization_dict['seconds']
            time_value = self.interval
        elif self.interval < 24000:
            time_type = localization_dict['minutes']
            time_value = self.interval / 60
        else:
            time_type = localization_dict['hours']
            time_value = self.interval / 3600
        count = None
        # Hardcode some common values. The plus one is intentional. It had
        # hardly any performance impact and enhances readability.
        if self.interval == 15 * 60:
            count = 15 + 1
        elif self.interval == 20 * 60:
            count = 4 + 1
        elif self.interval == 30 * 60:
            count = 6 + 1
        elif self.interval == 60 * 60:
            count = 4 + 1
        elif self.interval == 90 * 60:
            count = 6 + 1
        elif self.interval == 120 * 60:
            count = 4 + 1
        elif self.interval == 180 * 60:
            count = 6 + 1
        elif self.interval == 240 * 60:
            count = 6 + 1
        elif self.interval == 300 * 60:
            count = 6 + 1
        elif self.interval == 360 * 60:
            count = 12 + 1
        elif self.interval == 720 * 60:
            count = 12 + 1
        # Otherwise run some kind of autodetection routine.
        if not count:
            # Up to 15 time units and if it's a full number, show every unit.
            if time_value <= 15 and time_value % 1 == 0:
                count = int(time_value)
            # Otherwise determine whether they are divisible for numbers up to
            # 15. If a number is not divisible just show 10 units.
            else:
                count = 10
                for _i in range(15, 1, -1):
                    if time_value % _i == 0:
                        count = _i
                        break
            # Show at least 5 ticks.
            if count < 5:
                count = 5
        # Everything can be overwritten by user-specified number of ticks.
        if self.number_of_ticks:
            count = self.number_of_ticks
        # Calculate and set ticks.
        ticks = np.linspace(0.0, max_value, count)
        ticklabels = ['%i' % _i for _i in np.linspace(0.0, time_value, count)]
        self.axis[0].set_xticks(ticks)
        self.axis[0].set_xticklabels(ticklabels, rotation=self.tick_rotation,
                                     size=self.x_labels_size)
        self.axis[0].set_xlabel('%s %s' % (localization_dict['time in'],
                                           time_type), size=self.x_labels_size)

    # Modified from obspy.waveform imaging
    def __dayplot_set_y_ticks(self, *args, **kwargs):
        """
        Sets the yticks for the dayplot.
        """
        intervals = self.extreme_values.shape[0]
        # Only display all ticks if there are five or less steps or if option
        # is set.
        if intervals <= 5 or self.one_tick_per_line:
            # tick_steps = list(range(0, intervals))
            tick_steps = list(range(0, intervals+1))  # JJW: Includes 1 more tick step for UTC offset
            ticks = np.arange(intervals, 0, -1, dtype=float)
            ticks -= 0.5
        else:
            # tick_steps = list(range(0, intervals, self.repeat))
            tick_steps = list(range(0, intervals+1, self.repeat))
            ticks = np.arange(intervals, 0, -1 * self.repeat, dtype=float)
            # ticks = np.arange(intervals, -1, -1 * self.repeat, dtype=float)  # JJW: Adjusted to include 0 at end; will hold UTC offset
            ticks -= 0.5
        ticks = np.append(ticks, 0.0)  # JJW: Add 0 at end; will hold timezone marking

        # Complicated way to calculate the label of
        # the y-axis showing the second time zone
        sign = '%+i' % self.time_offset
        sign = sign[0]
        # label = "UTC (%s = UTC %s %02i:%02i)" % (
        #     self.timezone.strip(), sign, abs(self.time_offset),
        #     (self.time_offset % 1 * 60))
        # ticklabels = [(self.starttime + _i *
        #                self.interval).strftime(self.tick_format)
        #               for _i in tick_steps]
        ticklabels = [(self.starttime + (_i + 0.5) *
                       self.interval).strftime(self.tick_format)
                      for _i in np.flip(ticks)]
        start_tick_format = "%Y/%m/%d"  # JJW: Format for first tick
        end_tick_format = "UTC%s%02i:%02i" % (sign, abs(self.time_offset), self.time_offset % 1 * 60)
        ticklabels[0] = self.starttime.strftime(start_tick_format)  # JJW: Change first tick to proper format
        ticklabels[-1] = self.starttime.strftime(end_tick_format)  # JJW: Change last tick to time zone format
        self.axis[0].set_yticks(ticks)
        self.axis[0].set_yticklabels(ticklabels, size=self.y_labels_size)
        # Show time zone label if requested
        if self.show_y_UTC_label:
            self.axis[0].set_ylabel(label)
        if self.right_vertical_labels:
            yrange = self.axis[0].get_ylim()
            self.twin_x = self.axis[0].twinx()
            self.twin_x.set_ylim(yrange)
            self.twin_x.set_yticks(ticks)
            # y_ticklabels_twin = [(self.starttime + (_i + 1) *
            #                       self.interval).strftime(self.tick_format)
            #                      for _i in tick_steps]
            y_ticklabels_twin = [(self.starttime + (_i + 1.5) *
                                  self.interval).strftime(self.tick_format)
                                 for _i in np.flip(ticks)]
            y_ticklabels_twin[0] = self.starttime.strftime(start_tick_format)  # ? JJW: Change first tick to proper format
            y_ticklabels_twin[-1] = self.starttime.strftime(end_tick_format)  # ? JJW: Change last tick to time zone format
            self.twin_x.set_yticklabels(y_ticklabels_twin,
                                        size=self.y_labels_size)
