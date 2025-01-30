import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from vdapseisutils.sandbox.swarmmpl.clipboard import Clipboard
from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors


def main():
    # IRIS_01()
    IRIS_02()
    Augustine()
    Gareloi()
    Agung()


def IRIS_01():

    st = read("../data/waveforms/IRIS_test_data_01.mseed")
    st = st.merge().select(station="SEP")
    fig = Clipboard(st, mode="wg")
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    plt.show()  # show the plot
    print("Done.")


def IRIS_02():

    st = read("../data/waveforms/IRIS_test_data_02.mseed")
    st = st.slice(UTCDateTime("2004/09/15 14:30"), UTCDateTime("2004/09/15 14:39:59.999"))

    from vdapseisutils.sandbox.swarmmpl.clipboard import plot_spectrogram, plot_wave
    plot_wave(st[0]); plt.show()
    plot_spectrogram(st[0]); plt.show()

    fig = Clipboard(st, mode="wg")
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    plt.show()  # show the plot
    print("Done.")


def Augustine():

    st = read("../data/waveforms/Augustine_test_data_FI.mseed")
    print(st)
    st.plot()
    fig = Clipboard(st, mode="w", tick_type="relative", sync_waves=False)
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.scroll_traces(idx=[0, 1, 2], seconds=[-0.5, 0.25, -0.25])
    # fig.set_flim([0.1, 10.0])
    plt.show()  # show the plot
    print("Done.")


def Gareloi():

    suptitle = "Gareloi: Low Frequency Earthquakes"

    st = read("../data/waveforms/gareloi_test_data_20220710-010000.mseed")
    st = st.slice(UTCDateTime("2022/07/10 01:30:00"), UTCDateTime("2022/07/10 01:39:59.999"))
    st.filter("bandpass", freqmin=1.0, freqmax=10.0)
    print(st)

    fig = Clipboard(st, mode="wg")
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.axvline("2022/07/10 01:30:15")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"], color="red")  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.scroll_traces(idx=[1], seconds=[-60])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    fig = Clipboard(st, mode="g", figsize=(10.0/3, 6.0/3))
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r", lw=0.5)  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"], color="red", lw=0.5)  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.remove_labels()
    fig.suptitle("")
    plt.show()  # show the plot

    fig = Clipboard(st, mode="wg", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="binary", dbscale=False)
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"], color="red")  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    fig = Clipboard(st, mode="wg", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="plasma")
    fig.plot()
    fig.axvline("2022/07/10 01:30:15", color="r")  # Next add vertical axis spans
    fig.axvline(["2022/07/10 01:32:08", "2022/07/10 01:33:52", "2022/07/10 01:34:54",
                 "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"])  # Add more vertical axis spans
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    fig = Clipboard(st, mode="wg", figsize=(10.0, 6.0))
    fig.set_spectrogram(cmap="jet")
    fig.plot()
    fig.axvline(["2022/07/10 01:30:15", "2022/07/10 01:32:08", "2022/07/10 01:34:54"], color="black", ls="--")  # Add more vertical axis spans
    fig.axvline(["2022/07/10 01:33:52", "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"], color="black")  # Add more vertical axis spans
    fig.set_flim([0.1, 10.0])
    fig.suptitle(suptitle)
    plt.show()  # show the plot

    fig = Clipboard(st, mode="w", figsize=(10.0, 6.0))
    fig.set_wave(color="b")
    fig.plot()
    fig.axvline(["2022/07/10 01:30:15", "2022/07/10 01:32:08", "2022/07/10 01:34:54"], color="r", ls="--")  # Add more vertical axis spans
    fig.axvline(["2022/07/10 01:33:52", "2022/07/10 01:36:26", "2022/07/10 01:37:59", "2022/07/10 01:39:48"], color="r")  # Add more vertical axis spans
    fig.scroll_traces(idx=[-1], seconds=[-60])
    fig.set_flim([-1000, 1000])
    fig.suptitle(suptitle)
    plt.show()  # show the plot


def Agung():

    st = read("../data/waveforms/Agung_test_data_01.mseed")
    st = st.merge()  #.select("VG", "TMKS", "00", "EHZ")
    print(st)
    st.plot()
    fig = Clipboard(st, mode="g")
    fig.set_spectrogram(overlap=0.86, cmap=vdap_colors.viridis_u)
    fig.set_wave(color="k")  # Default behavior
    fig.plot()
    fig.suptitle("Agung Example Data")
    fig.set_alim([-1000, 1000])
    fig.set_flim([0.1, 10.0])
    plt.show()  # show the plot
    print("Done.")


if __name__ == "__main__":
    main()
