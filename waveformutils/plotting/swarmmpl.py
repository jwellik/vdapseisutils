import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from obspy import UTCDateTime

from vdapseisutils.waveformutils.nslcutils import getNSLCstr
from vdapseisutils.waveformutils.plotting.spectrogram import spectrogram


def swarmg(
        st,
        cmap='plasma', ylim=[0, 10],
        # power_range_db=[0.0,120.0],
        log_power=False, window_size=2, nfftpts=None, overlap=0.86,
        n_seis_ticks=6,
):
    print('Currently only plots one Trace')

    figax = spectrogram(
        st[0],
        cmap=cmap,
        per_lap=overlap,
        wlen=window_size,
        log=log_power,
        ylim=ylim,
    )

    return figax


def swarmw(st,
           color='k',
           n_seis_ticks=6,
           ):
    plot_duration = st[0].stats.endtime - st[0].stats.starttime

    plt.figure(figsize=(12, 2.5 * len(st)))
    for i, tr in enumerate(st):
        ax = plt.subplot(len(st), 1, i + 1)

        # Custom time series plot
        plt.plot(tr.times("matplotlib"), tr.data, color='k')

        # Customize axes
        ax.set_ylabel(getNSLCstr(tr), fontsize=12,
                      rotation='vertical',
                      multialignment='center',
                      horizontalalignment='center',
                      verticalalignment='center')
        ax.yaxis.set_ticks_position('right')
        ax.tick_params('y', labelsize=8)
        ax.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])

        # Add xticks on first and last plot
        if i == 0 or i == (len(st) - 1):  # Add ticks to top and bottom plot
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)  # Format xticks
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            # xticks go on top for first plot
            if i == 0: ax.set_title('Waveform');  # Add title to top plot
            if i == 0 and len(st) > 1: ax.xaxis.tick_top()  # If more than 1 plot, put xticks on top of first plot
        else:
            ax.set_xticks([])

    return ax


def swarmwg(st, gylim=[0.1, 10], cmap='plasma',
            wylim=None, wave_color='k'):
    """SWARMWG Plots waveform/spectrogram pairs for Stream objects with multiple Traces

    ARGS
    st          : Stream : Stream object
    
    KWARGS
    gylim       : list-like : 2 element list of Y-axis limits for spectrogram (Hz)
                  Default:[1,10]
    cmap        : Any ColorMap understood by Matplotlib (Default: 'plasma')
    wylim       : list-like : 2 element list of Y-axis limits for waveform (data units)
                  Default: None (auto-scale)
    wave_color  : Color of the waveform. Anything understood by Matplotlib as a color
    """

    import numpy as np

    #     st = replaceGapValue(st, gap_value=np.nan, fill_value=0) # Not sure this code is working

    nstreams = len(st)
    #    plot_duration = st[0].stats.endtime - st[0].stats.starttime
    figheight = 0.5 + 2.5 * nstreams
    left = 0.05
    right = 0.95
    top = 0.05
    bottom = 0.05
    space = 0.03
    wgratio = [1, 3]
    axheight = (1.0 - top - bottom - space * (nstreams - 1)) / nstreams

    fig = plt.figure(figsize=(12, figheight), constrained_layout=False)

    for n in range(len(st)):
        tr = st[n]

        # bottom to top
        # axbottom = bottom+axheight*n+space*(n)
        # axtop    = axbottom+axheight
        # top to bottom
        axtop = 1 - top - axheight * n - space * n
        axbottom = axtop - axheight
        # print(n, axbottom, axtop)

        wg = fig.add_gridspec(
            nrows=wgratio[1] + 1, ncols=1, left=left, right=right,
            # nrows=2, ncols=1, left=left, right=right,
            bottom=axbottom, top=axtop,
            wspace=0.00, hspace=0.00
        )

        w = fig.add_subplot(wg[0, :])  # Create subplot for Waveform
        w.plot(tr.times("matplotlib"), tr.data, color=wave_color)
        w.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])
        # if wylim not None: w.set_ylim(wylim)
        w.yaxis.set_ticks_position('right')

        g = fig.add_subplot(wg[1:, :])  # Create subplot for Spectrogram
        g.yaxis.set_ticks_position('right')
        g.text(
            -0.01, 0.67, getNSLCstr(tr), transform=g.transAxes, rotation='vertical',
            horizontalalignment='right', verticalalignment='center', fontsize=12,
        )
        spectrogram(tr, axes=g, cmap=cmap, ylim=gylim)
        g.set_ylim([0.1, 10])

    # Set xaxis ticks and labels
    # ??? settaxis( fig, n=2, minticks=3, maxticks=7 ) # n is how many plots per stream
    """
    The swarmwg() function makes two axes for each waveform - a waveform axis (w)
    and a spectrogram axis (g).
    First, 'locator' automatically decides how many xticks to create. Odd numbers ensure
    that no ticks are at the end of the graphs.
    Second, 'formatter' automatically decides the datetime format, based on how
    'zoomed in' the plot is.
    At the end, the last axis always gets x ticks.
    Before that - in the if/for loop - if there is more than 1 waveform (2 axes), xticks
    are added to the top axis, and all other axes are removed.
    This allows the plots to sit right on top of each other with xticks on the bottom
    (and top if there are multiple waveforms).
    """
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)  # Dynamic choice of xticks
    formatter = mdates.ConciseDateFormatter(locator)  # Automatic date formatter
    if len(fig.get_axes()) > 2:  # if len>2 bc 1 stream will produce two axes
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_locator(locator)  # Format top axis
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_formatter(formatter)
        for n in range(1, len(fig.get_axes())):  # Remove middle axes
            fig.get_axes()[n] = fig.get_axes()[n].set_xticks([])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_locator(
        locator)  # Format bottom axis (bottom axis is always [-1])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_formatter(formatter)

    return fig


def plot_triggers(triggers, cft, st, cftlim=None, trigonoff=(0.6, 1.5), nstatrig=None,
            wylim=None, wave_color='k'):
    """SWARMWG Plots waveform/spectrogram pairs for Stream objects with multiple Traces

    ARGS
    st          : Stream : Stream object of seismic waveform
    cft         : Stream : Stream object of cft function

    KWARGS
    cftlim      : list-like : 2 element list of Y-axis limits for spectrogram (Hz)
                  Default:[1,10]
    wylim       : list-like : 2 element list of Y-axis limits for waveform (data units)
                  Default: None (auto-scale)
    wave_color  : Color of the waveform. Anything understood by Matplotlib as a color
    """

    import numpy as np

    #     st = replaceGapValue(st, gap_value=np.nan, fill_value=0) # Not sure this code is working

    nstreams = len(st)
    #    plot_duration = st[0].stats.endtime - st[0].stats.starttime
    figheight = 0.5 + 2.5 * nstreams
    left = 0.05
    right = 0.95
    top = 0.05
    bottom = 0.05
    space = 0.03
    wgratio = [1, 3]
    axheight = (1.0 - top - bottom - space * (nstreams - 1)) / nstreams

    fig = plt.figure(figsize=(12, figheight), constrained_layout=False)

    for n in range(len(st)):
        tr = st[n]
        tr2 = cft[n]

        # bottom to top
        # axbottom = bottom+axheight*n+space*(n)
        # axtop    = axbottom+axheight
        # top to bottom
        axtop = 1 - top - axheight * n - space * n
        axbottom = axtop - axheight
        # print(n, axbottom, axtop)

        wg = fig.add_gridspec(
            nrows=wgratio[1] + 1, ncols=1, left=left, right=right,
            # nrows=2, ncols=1, left=left, right=right,
            bottom=axbottom, top=axtop,
            wspace=0.00, hspace=0.00
        )

        # Create subplots for two axes per station
        w = fig.add_subplot(wg[0, :])  # Create subplot for Waveform
        g = fig.add_subplot(wg[1:, :])  # Create subplot for Spectrogram

        # First, plot coincidence_triggers
        # {'time': 2004 - 09 - 28T00: 00:11.360200
        # Z,
        # 'stations': ['SEP', 'HSR'],
        # 'trace_ids': ['UW.SEP..EHZ', 'UW.HSR..EHZ'],
        # 'coincidence_sum': 2.0,
        # 'similarity': {},
        # 'duration': 13.940000057220459},
        if nstatrig is None: nstatrig = len(st)
        for trig in triggers:
            if trig["coincidence_sum"] >= nstatrig:
                t1 = UTCDateTime(trig["time"]).matplotlib_date
                t2 = UTCDateTime(trig["time"]+trig["duration"]).matplotlib_date
                w.axvspan(t1, t2, facecolor='y', alpha=0.5)
                g.axvspan(t1, t2, facecolor='y', alpha=0.5)


        w.plot(tr.times("matplotlib"), tr.data, color=wave_color)
        w.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])
        # if wylim is not None: w.set_ylim(wylim)
        w.yaxis.set_ticks_position('right')

        g.yaxis.set_ticks_position('right')
        g.text(
            -0.01, 0.67, getNSLCstr(tr), transform=g.transAxes, rotation='vertical',
            horizontalalignment='right', verticalalignment='center', fontsize=12,
        )
        g.plot(tr2.times("matplotlib"), tr2.data, color=wave_color)
        g.set_xlim([tr.times("matplotlib")[0], tr.times("matplotlib")[-1]])  # Use xlim of waveform just to be sure
        g.axhline(y=trigonoff[0], color="b", linestyle=":")
        g.axhline(y=trigonoff[1], color="r", linestyle=":")
        if cftlim is not None: g.set_ylim(cftlim)

    # Set xaxis ticks and labels
    # ??? settaxis( fig, n=2, minticks=3, maxticks=7 ) # n is how many plots per stream
    """
    The swarmwg() function makes two axes for each waveform - a waveform axis (w)
    and a spectrogram axis (g).
    First, 'locator' automatically decides how many xticks to create. Odd numbers ensure
    that no ticks are at the end of the graphs.
    Second, 'formatter' automatically decides the datetime format, based on how
    'zoomed in' the plot is.
    At the end, the last axis always gets x ticks.
    Before that - in the if/for loop - if there is more than 1 waveform (2 axes), xticks
    are added to the top axis, and all other axes are removed.
    This allows the plots to sit right on top of each other with xticks on the bottom
    (and top if there are multiple waveforms).
    """
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)  # Dynamic choice of xticks
    formatter = mdates.ConciseDateFormatter(locator)  # Automatic date formatter
    if len(fig.get_axes()) > 2:  # if len>2 bc 1 stream will produce two axes
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.tick_top()  # Put axis on top for top plot
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_locator(locator)  # Format top axis
        fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_formatter(formatter)
        for n in range(1, len(fig.get_axes())):  # Remove middle axes
            fig.get_axes()[n] = fig.get_axes()[n].set_xticks([])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_locator(
        locator)  # Format bottom axis (bottom axis is always [-1])
    fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_formatter(formatter)

    return fig


# PROGRAMMER'S TO DO:
# TODO: Functionalize key things like changing axes, setting up structure, etc.
# TODO: apply swarmwg() outline to swarmw() and swarmg()
# TODO: functionalize things shared by swarmw(), swarmg(), swarmwg(), .e.,g datetickformatter()
# TODO: give swarmwg() input arguments such as gargs=... & wargs=... to send to waveform and spectrogram routines
