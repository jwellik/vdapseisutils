from vdapseisutils.sandbox.swarmmpl import colors as vdap_colors

#
# def spectrogram(tr, samp_rate=None, wlen=6, overlap=0.5, dbscale=True, log_power=False,
#                 tick_type="datetime", cmap=vdap_colors.inferno_u,
#                 ax=None):
#
#     print(">>> swarmmpl.spectrogram.spectrogram")
#
#     if not ax:
#         fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))
#
#
#     # Set sample rate, if necessary
#     if samp_rate:
#         tr.resample(float(samp_rate))
#     else:
#         samp_rate = tr.stats.sampling_rate
#
#     # data and sample rates
#     fs = tr.stats.sampling_rate
#     signal = tr.data
#
#     # Define the start date and time
#     start_date = tr.stats.starttime
#
#     # Determine variables
#     if not wlen:
#         wlen = 128 / samp_rate
#
#     npts = len(signal)
#
#     nfft = int(_nearest_pow_2(wlen * samp_rate))
#
#     # stole this ValueError from ObsPy
#     if npts < nfft:
#         msg = (f'Input signal too short ({npts} samples, window length '
#               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
#               f'{samp_rate} Hz)')
#         raise ValueError(msg)
#
#     nlap = int(nfft * float(overlap))
#
#     signal = signal - signal.mean()
#
#     # times is returned in seconds after the start of the signal
#     frequencies, times, Sxx = scipy_spectrogram(signal, fs=fs, nperseg=nfft, noverlap=nlap, scaling='spectrum')
#
#     # db scale and remove zero/offset for amplitude
#     dbscale = True
#     if dbscale:
#         Sxx = 10 * np.log10(Sxx[1:, :])
#     else:
#         Sxx = np.sqrt(Sxx[1:, :])
#     frequencies = frequencies[1:]
#
#     # Convert time values to datetime objects
#     if tick_type == "datetime":
#         times_g = [start_date + timedelta(seconds=t) for t in times]  # time vector for g_kwargs (g)
#     else:  # "relative"
#         times_g = times
#
#     # Plot the g_kwargs with dates on the x-axis
#     ax.pcolormesh(times_g, frequencies, Sxx, shading='auto', cmap=cmap)  # plot g_kwargs
#     if log_power:
#         ax[1].set_yscale('log')  # Use a logarithmic scale for the y-axis
#
#     data = {"frequencies": frequencies, "times": times_g, "power": Sxx}
#
#     return ax, data


def swarmg(tr, samp_rate=None, wlen=6.0, overlap=0.5, dbscale=True, log_power=False,
                  cmap=vdap_colors.inferno_u, tick_type="datetime", relative_offset=0, ax=None):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    from datetime import timedelta
    import matplotlib.dates as mdates
    from obspy.imaging.spectrogram import _nearest_pow_2

    # TODO Don't make spectrogram if data are empty

    if samp_rate:
        tr.resample(float(samp_rate))
    else:
        samp_rate = tr.stats.sampling_rate

    # data and sample rates
    fs = tr.stats.sampling_rate
    signal = tr.data

    # Define the start date and time
    start_date = tr.stats.starttime.datetime

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
    if tick_type == "datetime":
        times_g = [start_date + timedelta(seconds=t) for t in times]  # time vector for g_kwargs (g)
    else:  # "relative"
        times_g = [relative_offset + t for t in times]

    # Plot the g_kwargs with dates on the x-axis
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))

    ax.pcolormesh(times_g, frequencies, Sxx, shading='auto', cmap=cmap)  # plot g_kwargs
    ax.set_xlabel('Date and Time')
    ax.set_ylabel('Frequency (Hz)')
    if log_power:
        ax.set_yscale('log')  # Use a logarithmic scale for the y-axis
    ax.set_ylim(0.1, samp_rate/2.0)  # Set the frequency range to 0.5 - 25 Hz
    if tick_type == "datetime":
        ax.set_xlim([tr.stats.starttime.datetime, tr.stats.endtime.datetime])
        loc = mdates.AutoDateLocator(minticks=5, maxticks=7)  # from matplotlib import dates as mdates
        ax.xaxis.set_major_locator(loc)
        formatter = mdates.ConciseDateFormatter(loc)
        ax.xaxis.set_major_formatter(formatter)
    else:  # "relative"
        ax.set_xlim([0, len(signal)/tr.stats.sampling_rate])

    data = {"freq": frequencies, "times": times_g, "power": Sxx}

    return ax, data