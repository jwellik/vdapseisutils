from obspy import UTCDateTime, Stream, read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vdapseisutils.sandbox.swarmmpl.clipboardclass import ClipboardClass
from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors

"""
Test 1a: Most simple
- waveform extents are same
- taxis: datetime
- sharex: True
- mode: wg
- add axvline

Test 1b: Pretty simple
- waveform extents are same
- sharex = True
- taxis: relative
- mode: wg

Test 2a:
- waveform extents are different
- taxis: datetime
- sharex: True
- mode: wg

Test 2b:
- waveform extents are different
- taxis: datetime
- sharex: False
- mode: wg

Test 2c:
- waveform extents are different
- taxis: relative
- sharex: True
- mode: wg

Test 2d:
- waveform extents are different
- taxis: relative
- sharex: False
- mode: wg



"""

def make_ugly_gareloi_stream(st):

    st = st.copy()
    st2 = Stream()
    st2 += st[0].slice(UTCDateTime("2022/07/10 01:05"), UTCDateTime("2022/07/10 01:06:59.999"))
    st2 += st[1].slice(UTCDateTime("2022/07/10 01:00"), UTCDateTime("2022/07/10 01:07:59.999"))
    st2 += st[2].slice(UTCDateTime("2022/07/10 01:01"), UTCDateTime("2022/07/10 01:10:"))
    st2 += st[3].slice(UTCDateTime("2022/07/10 01:00"), UTCDateTime("2022/07/10 01:09:59.9999"))
    st2 += st[4].slice(UTCDateTime("2022/07/10 01:00"), UTCDateTime("2022/07/10 01:09:59.9999"))

    return st2


def t2axiscoords(times, textent, axextent, tick_type="datetime"):
    """T2AXISCOORDS Converts time to axis coordinates. Requires the axis object and the actual time extent for the axis
    textent should be a list-like pair of datetime objects
    axextent should be the xlim of the axis (in sample units, in other words)

    """

    # input
    times = [UTCDateTime(t) for t in times]  # convert to UTCDateTime
    t1 = UTCDateTime(textent[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(textent[1])  # convet t2 to UTCDateTime
    t0 = t1  # preserve the original t1

    if tick_type == "datetime":
        xlim = axextent  # original extent of x axis
        axis_length = xlim[1] - xlim[0]  # length of original axis
        plot_duration_min = (t2 - t1) / 60  # duration in minutes that axis is meant to represent
        axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes

    if tick_type == "relative":
        # This is if the time axis should be in seconds instead of datetime
        raise Exception("Relative time axis not yet supported :-(")

    x = [(t - t0) * axis_samp_rate for t in times]

    return x


def __define_t_ticks__(trange, xrange, tick_type="datetime"):
    """
    Returns appropriate locations (in original axis coordinates) and labels for an x axis that is meant to span t1 to t2

    :param t1:
    :param t2:
    :param ax:
    :param tick_type:
    :return:
    """

    # input
    t1 = UTCDateTime(trange[0])  # convert t1 to UTCDateTime
    t2 = UTCDateTime(trange[1])  # convet t2 to UTCDateTime

    xlim = xrange  # original extent of x axis
    axis_length = xlim[1]-xlim[0]  # length of original axis
    plot_duration_min = (t2-t1)/60  # duration in minutes that axis is meant to represent
    axis_samp_rate = axis_length / (plot_duration_min * 60)  # sample rate of axis coordiantes
    # plot_duration_samp = 2400  # axis_length --> ax2.get_xlim()

    # determine axis format based on length of time represented
    # there's gotta be a better way to do this

    if plot_duration_min <= 2:  # 2 minutes or less
        nticks = 6
        tick_format_0 = "%Y/%m/%d %H:%M:%S"
        tick_format = '%H:%M:%S'
        t1 = pd.Timestamp(t1.datetime).round("10s")  # round to nearest 10s
        t2 = pd.Timestamp(t2.datetime).round("10s")
    elif plot_duration_min <= 10:  # 2 minutes to 10 minutes
        nticks = 6
        tick_format_0 = "%Y/%m/%d %H:%M"
        tick_format = "%H:%M"
        t1 = pd.Timestamp(t1.datetime).round("10T")  # round to nearest 10 minutes
        t2 = pd.Timestamp(t2.datetime).round("10T")
    elif plot_duration_min <= 60:  # 10 minutes to 1 hour
        nticks = 7
        tick_format_0 = "%Y/%m/%d %H:%M"
        tick_format = "%H:%M"
        t1 = pd.Timestamp(t1.datetime).round("10T")  # round to nearest 10 minutes
        t2 = pd.Timestamp(t2.datetime).round("10T")
    elif plot_duration_min <= 60*24:  # 1 hour to 1 day
        nticks = 7
        tick_format_0 = "%Y/%m/%d %H:00"
        tick_format = "%H"
        t1 = pd.Timestamp(t1.datetime).round("1H")  # round to nearest 1 hr
        t2 = pd.Timestamp(t2.datetime).round("1H")
    elif plot_duration_min <= 60*24*7:  # 1 day to 1 week
        nticks = 7
        tick_format_0 = "%Y/%m/%d"
        tick_format = "%m/%d"
        t1 = pd.Timestamp(t1.datetime).round("1D")  # round to nearest 1 day
        t2 = pd.Timestamp(t2.datetime).round("1D")
    else:  # greater than 1 week
        nticks = 7
        tick_format_0 = "%Y/%m/%d"
        tick_format = "%m/%d"
        t1 = pd.Timestamp(t1.datetime).round("1D")  # round to nearest 1 day
        t2 = pd.Timestamp(t2.datetime).round("1D")

    dr = pd.date_range(t1, t2, periods=nticks)

    ticks = np.linspace(xrange[0], xrange[1], nticks)

    if tick_type == "days":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60 / 60 / 24
        axis_label = "Days"
    elif tick_type == "hours":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60 / 60
        axis_label = "Hours"
    elif tick_type == "minutes":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate / 60
        axis_label = "Minutes"
    elif tick_type == "relative" or tick_type == "seconds":
        tick_labels = (ticks - xrange[0]) / axis_samp_rate
        axis_label = "Seconds"
    else:  #  tick_type == "datetime":
        tick_labels = [d.strftime(tick_format) for d in dr]
        tick_labels[0] = dr[0].strftime(tick_format_0)
        axis_label = "Time"

    return ticks, tick_labels, axis_label


def plot_clipboard(st, mode="wg", sharex=True):
    """
    This method plots all waveforms and spectrograms along an xaxis of samples, the xlim is adjusted +/- so that all
    waveforms and spectrograms are plotted in the right location relative to each other. Thus, it is necessary to know
    the xlim offsets for each trace/axis so that datetimes can be properly plotted later. This seems like a weird way to
    do it, but it is one of the easier ways to do it when trying to align data of different sampling rates (e.g.,
    waveforms and spectrograms). One alternative would be to use set_extent() for imshow, but the ObsPy g_kwargs()
    method does not allow this as an input parameter. Furthermore, ObsPy g_kwargs() sometimes uses pcolormesh(),
    which does not have an extent parameter.

    :param st:
    :return:
    """

    # User settings
    spectrogram_defaults = {"log": False, "samp_rate": 25, "dbscale": True, "per_lap": 0.5, "mult": 25.0, "wlen": 6,
                            "cmap": vdap_colors.inferno_u}

    # Figure settings
    mode = mode
    if mode == "wg":
        wg_ratio = [1, 3]  # height ratio of waveform axis : specgram axis
        nplots = 2
    else:
        wg_ratio = [1]
        nplots = 1
    sharex = True
    tick_type = "datetime"
    fig = plt.figure(figsize=(8.5, 10))
    gs = fig.add_gridspec(nrows=len(st)*nplots, ncols=1, height_ratios=wg_ratio * len(st))

    # Determine offsets, reftime, etc.
    starttimes = [tr.stats.starttime for tr in st]  # starttime for every trace
    endtimes =  [tr.stats.endtime for tr in st]  # endtime for every trace
    data_extent = [min(starttimes), max(endtimes)]  # maximum start:end extent across all traces
    extent_len = (data_extent[1] - data_extent[0]) * st[0].stats.sampling_rate  # length of maximum extent in samples
    offset_t = [tr.stats.starttime - data_extent[0] for tr in st]  # offset of each trace in seconds
    print(offset_t)

    # Plot the data
    for i, tr in enumerate(st):
        axn = i*nplots

        # First axis - waveform
        offset = tr.stats.starttime - data_extent[0]
        xlim = [0-(offset*tr.stats.sampling_rate), extent_len-(offset*tr.stats.sampling_rate)]  # necessary xlim in samples for trace to align with other traces
        fig.add_subplot(gs[axn])
        fig.axes[axn].plot(tr.data, "r")  # Just plot points, not time
        fig.axes[axn].set_xlim(xlim)
        ticks, labels, xlabel = __define_t_ticks__(data_extent, fig.axes[axn].get_xlim(), tick_type=tick_type)
        fig.axes[axn].set_xticks(ticks)
        fig.axes[axn].set_xticklabels(labels)

        # Secondary axis - specgram axis
        offset = tr.stats.starttime - data_extent[0]
        xlim = [xlim[0]/spectrogram_defaults["samp_rate"], xlim[1]/spectrogram_defaults["samp_rate"]]
        fig.add_subplot(gs[axn+1])
        tr.g_kwargs(axes=fig.axes[axn + 1], **spectrogram_defaults)
        fig.axes[axn+1].set_xlim(xlim)
        ticks, labels, xlabel = __define_t_ticks__(data_extent, fig.axes[axn+1].get_xlim(), tick_type=tick_type)
        fig.axes[axn+1].set_xticks(ticks)
        fig.axes[axn+1].set_xticklabels(labels)
        fig.axes[axn + 1].set_xlabel(xlabel)

    if sharex:
        pass
        # remove middle axes

    return fig


def main():
    ####################################################################################################################

    st = read("/home/jwellik/PROJECTS/Gallery/waveform_data/gareloi_test_data_20220710-010000.mseed")
    st2 = make_ugly_gareloi_stream(st)

    ####################################################################################################################

    st1 = st.slice(UTCDateTime("2022/07/10 01:07"), UTCDateTime("2022/07/10 01:16:59.999"))
    st1.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print(st1)
    st1.plot()

    # fig1 = plot_clipboard(st1)
    # fig1.axes[2].plot(0,0, "sk")
    # plt.show()

    fig1 = plt.figure(FigureClass=ClipboardClass, st=st1, mode="g")
    fig1.axvline("2022/07/10 01:12:00", color="r")
    fig1.suptitle("Gareloi | Datetime axis")
    plt.show()

    ####################################################################################################################

    st2.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print(st2)
    st2.plot()

    # fig2 = plot_clipboard(st2)
    # fig2.axes[2].plot(0,0, "sk")
    # plt.show()

    fig2 = plt.figure(FigureClass=ClipboardClass, st=st2, mode="wg", tick_type="relative",
                      g={"wlen": 2.0, "overlap": 0.86, "cmap":"viridis"})
    fig2.axvline("2022/07/10 01:05:00", color="r")
    fig2.axvline(350, color="k", unit="relative")
    fig2.suptitle("Gareloi | Gappy data, Relative time axis")
    plt.show()

    ####################################################################################################################

    st3 = read("/home/jwellik/PROJECTS/Gallery/waveform_data/Augustine_test_data_FI.mseed")
    print(st3)
    st3.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print(st3)
    # [tr.plot() for tr in st3]
    st3.plot()

    # fig3 = plot_clipboard(st3, tick_type="relative", sharex=False)
    # plt.show()

    fig3 = plt.figure(FigureClass=ClipboardClass, st=st3, figsize=(15, 6), mode="w", tick_type="relative", sharex=False,
                      g={"wlen": 2.0, "overlap": 0.86, "cmap":"viridis"})
    fig3.suptitle("Augustine | Not synced by time | Example earthquakes from FI reference")
    plt.show()


    ####################################################################################################################

    print("Done.")


if __name__ == "__main__":
    main()
