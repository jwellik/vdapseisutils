import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, UTCDateTime

from vdapseisutils.core.datasource.waveID import waveID
from vdapseisutils.style import colors as vdap_colors
from vdapseisutils.utils.timeutils import convert_timeformat
from vdapseisutils.core.maps.maps import prep_catalog_data_mpl

# Plot defaults
spectrogram_defaults = {"samp_rate": 50, "wlen": 6, "overlap": 0.5, "dbscale": True, "log_power": False,
                        "cmap": vdap_colors.inferno_u}

# spectrogram_defaults_bw = {**spectrogram_defaults, **{"cmap": "binary", "dbscale": False, "clip": [0.0, 1.0]}}
wave_defaults = {"color": "k"}

# Figure and Axis defaults
figsize_default = (10.0, 6.0)
fontsize_default = 10
w_ax_defaults = {"color": "k"}
g_ax_defaults = {"ylim": [0.1, 10.0]}
s_ax_defaults = {"color": "k", "mode": "loglog", "ylim": [1.0, 10.0]}


def _set_xaxis_relative_labels(ax, reference_time=None):
    """Set relative xtick labels for a xaxis with datetime labels"""

    from obspy.imaging.util import _set_xaxis_obspy_dates
    _set_xaxis_obspy_dates(ax)  # set xticks as datetimes (always); doesn't hurt to do this again

    if not reference_time:
        reference_time = ax.get_xlim()[0]  # earliest time stamp on the x-axis

    xticks = ax.get_xticks()
    nticks = len(xticks) - 1
    ax.set_xticks(xticks[1:])  # reset the xticks so that set_xticklabels is happy
    xticklabels = [str(round(x)) for x in ((xticks[1:] - xticks[0]) * 86400 / 60)]  # minutes
    ax.set_xticklabels(xticklabels)


def t2axiscoords(times, textent, axextent, unit="datetime"):
    """T2AXISCOORDS Converts time to axis coordinates. Requires the axis object and the actual time extent for the axis
    textent should be a list-like pair of datetime objects
    axextent should be the xlim of the axis (in sample units, in other words)
    unit : Are times given as datetimes or "relative" seconds from the beginning of the axis?
    """

    # input
    times = [UTCDateTime(t) for t in times]  # convert to UTCDateTime
    t1 = UTCDateTime(textent[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(textent[1])  # convet t2 to UTCDateTime
    t0 = t1  # preserve the original t1

    if unit == "datetime":
        xlim = axextent  # original extent of x axis
        axis_length = xlim[1] - xlim[0]  # length of original axis
        plot_duration_min = (t2 - t1) / 60  # duration in minutes that axis is meant to represent
        axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes
        x = [xlim[0] + ((t - t0) * axis_samp_rate) for t in times]

    else:  # "relative"
        x = times

    return x


def plot_trace(tr, mode="wg", tick_type="datetime", relative_offset=0, fig=None,
               wave_settings=wave_defaults, spectrogram_settings=spectrogram_defaults):
    """PLOT_TRACE Plots a single Trace and returns a MatPlotLib Figure object"""

    # TODO Add check to verify tr is a single Trace

    if not fig:
        fig = plt.figure(figsize=figsize_default)

    if mode == "wg":
        nrows = 2
        height_ratios = [1, 3]
        waxn = 0
        gaxn = 1
    elif mode == "w":
        nrows = 1
        height_ratios = [1]
        waxn = 0
        gaxn = None
    elif mode == "g":
        nrows = 1
        height_ratios = [1]
        waxn = None
        gaxn = 0

    gs = fig.add_gridspec(nrows, 1, height_ratios=height_ratios, hspace=0.0)
    for i in range(nrows):
        fig.add_subplot(gs[i])

    if waxn is not None:
        plot_wave(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[waxn], **wave_settings)
    if gaxn is not None:
        plot_spectrogram(tr, tick_type=tick_type, relative_offset=relative_offset, ax=fig.axes[gaxn],
                         **spectrogram_settings)

    # NSLC Label - Horizontal label with two rows
    s = tr.id  # returns net.sta.loc.cha
    idx = s.index(".", 3)  # ? gets second instance of "." assuming NN.SS....
    s1, s2 = s[:idx], s[idx + 1:]
    fig.axes[-1].text(
        # -0.01, 0.67, s1 + "\n" + s2, transform=fig.axes[-1].transAxes, rotation='horizontal',
        -0.01, 0.67, idx, transform=fig.axes[-1].transAxes, rotation='horizontal',
        horizontalalignment="right", verticalalignment='center', fontsize=fontsize_default,
    )

    # fig.subplots_adjust(hspace=None)  # Remove horizontal space between subplots
    return fig


def plot_wave(tr, tick_type="datetime", relative_offset=0, color="k", ax=None, **kwargs):
    """PLOT_WAVE Plots waveform of single Trace. Returns a MatPlotLib Axes object"""

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize_default)

    # Convert time values to datetime objects
    if tick_type == "datetime":
        times_w = [(tr.stats.starttime + timedelta(seconds=t)).datetime for t in
                   tr.times()]  # time vector for the waveform (w)
    else:  # "relative"
        times_w = [relative_offset + t for t in tr.times()]
    ax.plot(times_w, tr.data, color=color, **kwargs)
    ax.yaxis.set_ticks_position("right")
    ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{:2.1f}"))

    return ax


def plot_spectrogram(tr, samp_rate=None, wlen=2.0, overlap=0.86, dbscale=True, log_power=False,
                     cmap=vdap_colors.inferno_u, tick_type="datetime", relative_offset=0, ax=None):
    """PLOT_WAVE Plots spectrogram of single Trace. Returns a MatPlotLib Axes object"""

    # TODO Don't make spectrogram if data are empty or if all NaN

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    from datetime import timedelta
    from obspy.imaging.spectrogram import _nearest_pow_2

    if samp_rate:
        tr.resample(float(samp_rate))
    else:
        samp_rate = tr.stats.sampling_rate

    # data and sample rates
    fs = tr.stats.sampling_rate
    signal = tr.data

    # Determine variables
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(signal)

    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    # if mult is not None:
    #     mult = int(_nearest_pow_2(mult))
    #     mult = mult * nfft
    nlap = int(nfft * float(overlap))

    signal = signal - signal.mean()

    frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=nfft, noverlap=nlap, scaling='spectrum')

    # db scale and remove zero/offset for amplitude
    if dbscale:
        Sxx = 10 * np.log10(Sxx[1:, :])
    else:
        Sxx = np.sqrt(Sxx[1:, :])
    frequencies = frequencies[1:]

    # vmin, vmax = clip
    # if vmin < 0 or vmax > 1 or vmin >= vmax:
    #     msg = "Invalid parameters for clip option."
    #     raise ValueError(msg)
    # _range = float(Sxx.max() - Sxx.min())
    # vmin = Sxx.min() + vmin * _range
    # vmax = Sxx.min() + vmax * _range
    # norm = Normalize(vmin, vmax, clip=True)

    # Convert time values to datetime objects
    start_date = tr.stats.starttime.datetime
    if tick_type == "datetime":
        times_g = [start_date + timedelta(seconds=t) for t in times]  # time vector for g_kwargs (g)
    else:  # "relative"
        times_g = [relative_offset + t for t in times]

    # Plot the g_kwargs with dates on the x-axis
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize_default)

    ax.pcolormesh(times_g, frequencies, Sxx, shading='auto', cmap=cmap)  # plot g_kwargs
    if log_power:
        ax.set_yscale('log')  # Use a logarithmic scale for the y-axis
    ax.set_ylim(0.1, samp_rate / 2.0)  # Set the frequency range to 0.5 - 25 Hz
    ax.yaxis.set_ticks_position("right")
    ax.set_xlabel("")
    ax.set_ylabel("")

    data = {"freq": frequencies, "times": times_g, "power": Sxx}

    return ax, data

def plot_swarmwg(tr, tick_type="time", figsize=(10, 3), height_ratios=[0.2, 0.8],
                 alim=None, flim=None, overlap=0.86, wlen=2.0, dbscale=True,
                 log_power=False, cmap="viridis"):
    """Plot a single trace with waveform and spectrogram."""

    from vdapseisutils.core.swarmmpl.clipboard import plot_wave, plot_spectrogram
    from obspy.imaging.util import _set_xaxis_obspy_dates
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import datetime
    from matplotlib.dates import num2date, date2num

    # Create a figure with a specific size
    fig = plt.figure(figsize=figsize)

    # Create a gridspec with height ratios
    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)  # Top: 20%, Bottom: 80%

    # Create the first subplot (top) - waveform
    ax0 = fig.add_subplot(gs[0])
    tr_copy = tr.copy()
    ax0 = plot_wave(tr_copy, ax=ax0)
    ax0.set_xlim(tr_copy.times("utcdatetime")[0].datetime, tr_copy.times("utcdatetime")[-1].datetime)
    if alim is not None:
        ax0.set_ylim(alim)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Create the second subplot (bottom) - spectrogram
    ax1 = fig.add_subplot(gs[1])
    ax1, data = plot_spectrogram(tr, overlap=overlap, wlen=wlen, dbscale=dbscale,
                                log_power=log_power, cmap=cmap, ax=ax1)
    ax1.set_xlim(tr.times("utcdatetime")[0].datetime, tr.times("utcdatetime")[-1].datetime)
    ax1.set_ylabel(tr.id)
    if flim is not None:
        ax1.set_ylim(flim)

    ax1.set_yticks(ax1.get_yticks())

    # Handle different tick types
    _set_xaxis_obspy_dates(ax1)  # set xticks as datetimes (always)

    if tick_type in ["relative", "seconds"]:
        # Use seconds for relative time
        xticks = ax1.get_xticks()  # matplotlib datenums
        nticks = len(xticks)
        xtick_dt = [num2date(x) for x in xticks]
        start_time = xtick_dt[0]

        # Calculate relative seconds from start
        relative_seconds = [(dt - start_time).total_seconds() for dt in xtick_dt]

        # Create nice tick positions (every ~30 seconds or so)
        duration = relative_seconds[-1]
        if duration <= 120:  # <= 2 minutes
            tick_interval = 30  # 30 second intervals
        elif duration <= 600:  # <= 10 minutes
            tick_interval = 60  # 1 minute intervals
        else:
            tick_interval = 120  # 2 minute intervals

        nice_ticks = list(range(0, int(duration) + tick_interval, tick_interval))
        nice_tick_dt = [start_time + datetime.timedelta(seconds=s) for s in nice_ticks]
        nice_tick_labels = [f"{s}" for s in nice_ticks]
        nice_tick_labels[-1] += " s"

        xticks_new = [date2num(x) for x in nice_tick_dt]
        ax1.set_xticks(xticks_new)
        ax1.set_xticklabels(nice_tick_labels)

    elif tick_type == "minutes":
        # Use minutes for relative time
        xticks = ax1.get_xticks()
        nticks = len(xticks)
        xtick_dt = [num2date(x) for x in xticks]
        start_time = xtick_dt[0]

        # Calculate relative minutes from start
        relative_minutes = [(dt - start_time).total_seconds() / 60 for dt in xtick_dt]

        # Create nice tick positions
        duration_min = relative_minutes[-1]
        if duration_min <= 10:  # <= 10 minutes
            tick_interval = 2  # 2 minute intervals
        elif duration_min <= 60:  # <= 1 hour
            tick_interval = 10  # 10 minute intervals
        else:
            tick_interval = 30  # 30 minute intervals

        nice_ticks = [i * tick_interval for i in range(int(duration_min / tick_interval) + 1)]
        nice_tick_dt = [start_time + datetime.timedelta(minutes=m) for m in nice_ticks]
        nice_tick_labels = [f"{m}" for m in nice_ticks]
        nice_tick_labels[-1] += " min"

        xticks_new = [date2num(x) for x in nice_tick_dt]
        ax1.set_xticks(xticks_new)
        ax1.set_xticklabels(nice_tick_labels)

    elif tick_type == "hours":
        # Use hours for relative time
        xticks = ax1.get_xticks()
        nticks = len(xticks)
        xtick_dt = [num2date(x) for x in xticks]
        start_time = xtick_dt[0]

        # Calculate relative hours from start
        relative_hours = [(dt - start_time).total_seconds() / 3600 for dt in xtick_dt]

        # Create nice tick positions
        duration_hr = relative_hours[-1]
        if duration_hr <= 6:  # <= 6 hours
            tick_interval = 1  # 1 hour intervals
        elif duration_hr <= 24:  # <= 1 day
            tick_interval = 4  # 4 hour intervals
        else:
            tick_interval = 12  # 12 hour intervals

        nice_ticks = [i * tick_interval for i in range(int(duration_hr / tick_interval) + 1)]
        nice_tick_dt = [start_time + datetime.timedelta(hours=h) for h in nice_ticks]
        nice_tick_labels = [f"{h}" for h in nice_ticks]
        nice_tick_labels[-1] += " hr"

        xticks_new = [date2num(x) for x in nice_tick_dt]
        ax1.set_xticks(xticks_new)
        ax1.set_xticklabels(nice_tick_labels)

    # If tick_type is "time" or anything else, keep the default datetime formatting

    # Adjust the layout
    plt.subplots_adjust(hspace=0.0)

    return fig


def plot_clipboard(st, tick_type="time", figsize=(10, 8), trace_height=3,
                   alim=None, flim=None, overlap=0.86, wlen=2.0,
                   dbscale=True, log_power=False, cmap="viridis"):
    """Plot multiple traces in a clipboard-style layout with each trace in its own row."""

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from vdapseisutils.core.swarmmpl.clipboard import plot_wave, plot_spectrogram
    from obspy.imaging.util import _set_xaxis_obspy_dates
    import datetime
    from matplotlib.dates import num2date, date2num

    # Calculate the number of traces
    n_traces = len(st)

    # Adjust figure height based on number of traces
    total_height = trace_height * n_traces
    fig = plt.figure(figsize=(figsize[0], total_height))

    # Create gridspec for multiple rows (each row will have 2 subplots: wave + spectrogram)
    gs = gridspec.GridSpec(n_traces * 2, 1, hspace=0.1,
                          height_ratios=[0.2, 0.8] * n_traces)

    axes = []

    for i, tr in enumerate(st):
        # Calculate indices for waveform and spectrogram
        wave_idx = i * 2
        spec_idx = i * 2 + 1

        # Waveform subplot
        ax_wave = fig.add_subplot(gs[wave_idx])
        tr_copy = tr.copy()
        ax_wave = plot_wave(tr_copy, ax=ax_wave)
        ax_wave.set_xlim(tr_copy.times("utcdatetime")[0].datetime,
                        tr_copy.times("utcdatetime")[-1].datetime)
        if alim is not None:
            ax_wave.set_ylim(alim)
        ax_wave.set_xticks([])
        ax_wave.set_yticks([])

        # Spectrogram subplot
        ax_spec = fig.add_subplot(gs[spec_idx])
        ax_spec, data = plot_spectrogram(tr, overlap=overlap, wlen=wlen,
                                        dbscale=dbscale, log_power=log_power,
                                        cmap=cmap, ax=ax_spec)
        ax_spec.set_xlim(tr.times("utcdatetime")[0].datetime,
                        tr.times("utcdatetime")[-1].datetime)
        ax_spec.set_ylabel(tr.id)
        if flim is not None:
            ax_spec.set_ylim(flim)

        ax_spec.set_yticks(ax_spec.get_yticks())

        # Handle tick formatting (only for the bottom trace)
        _set_xaxis_obspy_dates(ax_spec)

        if i == n_traces - 1:  # Only add x-axis labels to the bottom trace
            if tick_type in ["relative", "seconds"]:
                # Use seconds for relative time
                xticks = ax_spec.get_xticks()
                xtick_dt = [num2date(x) for x in xticks]
                start_time = xtick_dt[0]

                # Calculate relative seconds from start
                relative_seconds = [(dt - start_time).total_seconds() for dt in xtick_dt]

                # Create nice tick positions
                duration = relative_seconds[-1]
                if duration <= 120:  # <= 2 minutes
                    tick_interval = 30  # 30 second intervals
                elif duration <= 600:  # <= 10 minutes
                    tick_interval = 60  # 1 minute intervals
                else:
                    tick_interval = 120  # 2 minute intervals

                nice_ticks = list(range(0, int(duration) + tick_interval, tick_interval))
                nice_tick_dt = [start_time + datetime.timedelta(seconds=s) for s in nice_ticks]
                nice_tick_labels = [f"{s}" for s in nice_ticks]
                nice_tick_labels[-1] += " s"

                xticks_new = [date2num(x) for x in nice_tick_dt]
                ax_spec.set_xticks(xticks_new)
                ax_spec.set_xticklabels(nice_tick_labels)

            elif tick_type == "minutes":
                # Use minutes for relative time
                xticks = ax_spec.get_xticks()
                xtick_dt = [num2date(x) for x in xticks]
                start_time = xtick_dt[0]

                # Calculate relative minutes from start
                relative_minutes = [(dt - start_time).total_seconds() / 60 for dt in xtick_dt]

                # Create nice tick positions
                duration_min = relative_minutes[-1]
                if duration_min <= 10:  # <= 10 minutes
                    tick_interval = 2  # 2 minute intervals
                elif duration_min <= 60:  # <= 1 hour
                    tick_interval = 10  # 10 minute intervals
                else:
                    tick_interval = 30  # 30 minute intervals

                nice_ticks = [i * tick_interval for i in range(int(duration_min / tick_interval) + 1)]
                nice_tick_dt = [start_time + datetime.timedelta(minutes=m) for m in nice_ticks]
                nice_tick_labels = [f"{m}" for m in nice_ticks]
                nice_tick_labels[-1] += " min"

                xticks_new = [date2num(x) for x in nice_tick_dt]
                ax_spec.set_xticks(xticks_new)
                ax_spec.set_xticklabels(nice_tick_labels)

            elif tick_type == "hours":
                # Use hours for relative time
                xticks = ax_spec.get_xticks()
                xtick_dt = [num2date(x) for x in xticks]
                start_time = xtick_dt[0]

                # Calculate relative hours from start
                relative_hours = [(dt - start_time).total_seconds() / 3600 for dt in xtick_dt]

                # Create nice tick positions
                duration_hr = relative_hours[-1]
                if duration_hr <= 6:  # <= 6 hours
                    tick_interval = 1  # 1 hour intervals
                elif duration_hr <= 24:  # <= 1 day
                    tick_interval = 4  # 4 hour intervals
                else:
                    tick_interval = 12  # 12 hour intervals

                nice_ticks = [i * tick_interval for i in range(int(duration_hr / tick_interval) + 1)]
                nice_tick_dt = [start_time + datetime.timedelta(hours=h) for h in nice_ticks]
                nice_tick_labels = [f"{h}" for h in nice_ticks]
                nice_tick_labels[-1] += " hr"

                xticks_new = [date2num(x) for x in nice_tick_dt]
                ax_spec.set_xticks(xticks_new)
                ax_spec.set_xticklabels(nice_tick_labels)

            # If tick_type is "time" or anything else, keep the default datetime formatting
        else:
            # Remove x-axis labels for all but the bottom trace
            ax_spec.set_xticks([])

        axes.append((ax_wave, ax_spec))

    return fig, axes