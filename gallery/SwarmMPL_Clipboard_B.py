# %%
"""
SWARMmpl Clipboard

SWARMmpl Clipboard plots waveforms and spectrograms in the manner used by SWARM (https://volcanoes.usgs.gov/software/swarm/index.shtml).

These examples explore different colormaps for spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from vdapseisutils.sandbox.swarmmpl.clipboard import Clipboard
from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


def main():

    # %%

    suptitle = "Gareloi: Low Frequency Earthquakes"

    st = read("../data/waveforms/gareloi_test_data_20220710-010000.mseed")
    st = st.slice(UTCDateTime("2022/07/10 01:30:00"), UTCDateTime("2022/07/10 01:39:59.999"))
    st.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print(st)

    # Jet
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="jet")
    fig.plot()
    fig.axvline(["2022/07/10 01:30:15", "2022/07/10 01:32:08", "2022/07/10 01:34:54"], color="black",
                ls="--")  # Add more vertical axis spans
    fig.axvline(["2022/07/10 01:33:52", "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"],
                color="black")  # Add more vertical axis spans
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Hot (Reversed)
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="hot_r")
    fig.plot()
    fig.axvline(["2022/07/10 01:30:15", "2022/07/10 01:32:08", "2022/07/10 01:34:54"], color="black",
                ls="--")  # Add more vertical axis spans
    fig.axvline(["2022/07/10 01:33:52", "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"],
                color="black")  # Add more vertical axis spans
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Viridis (Upper half)
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.axvline("2022/07/10 01:30:15")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"],
                color="red")  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Binary
    fig = Clipboard(st, mode="wg", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="binary", dbscale=False)
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"],
                color="red")  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Plasma
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="plasma")
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Magma
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="magma")
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Inferno
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="inferno")
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Inferno (Upper half)
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap=vdap_colors.inferno_u)
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    # Inferno (Reversed)
    fig = Clipboard(st, mode="g", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="inferno_r")
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot


    # %%
    """
    Example 2: This example experiments with different axvline options
    """

    fig = Clipboard(st, mode="w", figsize=(10.0, 6.0))
    fig.set_wave(color="b")
    fig.plot()
    fig.axvline(["2022/07/10 01:30:15", "2022/07/10 01:32:08", "2022/07/10 01:34:54"], color="r",
                ls="--")  # Add more vertical axis spans
    fig.axvline(["2022/07/10 01:33:52", "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"],
                color="r")  # Add more vertical axis spans
    fig.scroll_traces(idx=[-1], seconds=[-60])
    fig.set_flim([-1000, 1000])
    fig.suptitle(suptitle)
    plt.show()  # show the plot


if __name__ == "__main__":
    main()
