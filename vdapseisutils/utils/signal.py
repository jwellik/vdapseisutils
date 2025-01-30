import numpy as np

from obspy.imaging.cm import obspy_sequential, obspy_divergent
from obspy.signal import util
from obspy.signal.tf_misfit import cwt


def plot_tfr(st, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6, left=0.1,
             bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6, w_cb=0.01,
             d_cb=0.0, show=True, plot_args=['k', 'k'], clim=0.0,
             cmap=obspy_sequential, mode='absolute', fft_zero_pad_fac=0):
    """
    Plot time frequency representation, spectrum and time series of the signal.

    :param st: signal, type numpy.ndarray with shape (number of components,
        number of time samples) or (number of timesamples, ) for single
        component data
    :param dt: time step between two samples in st
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param left: plot distance from the left of the figure
    :param bottom: plot distance from the bottom of the figure
    :param h_1: height of the signal axis
    :param h_2: height of the TFR/spectrum axis
    :param w_1: width of the spectrum axis
    :param w_2: width of the TFR/signal axes
    :param w_cb: width of the colorbar axes
    :param d_cb: distance of the colorbar axes to the other axes
    :param show: show figure or return
    :param plot_args: list of plot arguments passed to the signal and spectrum
        plots
    :param clim: limits of the colorbars
    :param cmap: colormap for TFEM/TFPM, either a string or
        matplotlib.cm.Colormap instance
    :param mode: 'absolute' for absolute value of TFR, 'power' for ``|TFR|^2``
    :param fft_zero_pad_fac: integer, if > 0, the signal is zero padded to
        ``nfft = next_pow_2(len(st)) * fft_zero_pad_fac`` to get smoother
        spectrum in the low frequencies (has no effect on the TFR and might
        make demeaning/tapering necessary to avoid artifacts)

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    .. rubric:: Example

    >>> from obspy import read
    >>> tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
    >>> plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01, # doctest: +SKIP
    ...         fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)

    .. plot::

        from obspy.signal.tf_misfit import plot_tfr
        from obspy import read
        tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
        plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01,
                fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    npts = st.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0

    if fft_zero_pad_fac == 0:
        nfft = npts
    else:
        nfft = util.next_pow_2(npts) * fft_zero_pad_fac

    f_lin = np.linspace(0, 0.5 / dt, nfft // 2 + 1)

    if len(st.shape) == 1:
        _w = np.zeros((1, nf, npts), dtype=complex)
        _w[0] = cwt(st, dt, w0, fmin, fmax, nf)
        ntr = 1

        spec = np.zeros((1, nfft // 2 + 1), dtype=complex)
        spec[0] = np.fft.rfft(st, n=nfft) * dt

        st = st.reshape((1, npts))
    else:
        _w = np.zeros((st.shape[0], nf, npts), dtype=complex)
        spec = np.zeros((st.shape[0], nfft // 2 + 1), dtype=complex)

        for i in np.arange(st.shape[0]):
            _w[i] = cwt(st[i], dt, w0, fmin, fmax, nf)
            spec[i] = np.fft.rfft(st[i], n=nfft) * dt

        ntr = st.shape[0]

    if mode == 'absolute':
        _tfr = np.abs(_w)
        spec = np.abs(spec)
    elif mode == 'power':
        _tfr = np.abs(_w) ** 2
        spec = np.abs(spec) ** 2
    else:
        raise ValueError('mode "' + mode + '" not defined!')

    figs = []

    for itr in np.arange(ntr):
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom, w_2, h_1])
        ax_sig.plot(t, st[itr], plot_args[0])

        # plot TFR
        ax_tfr = fig.add_axes([left + w_1, bottom + h_1, w_2, h_2])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfr[itr].shape[0]))
        img_tfr = _pcolormesh_same_dim(ax_tfr, x, y, _tfr[itr], cmap=cmap)
        img_tfr.set_rasterized(True)
        ax_tfr.set_yscale("log")
        ax_tfr.set_ylim(fmin, fmax)
        ax_tfr.set_xlim(t[0], t[-1])

        # plot spectrum
        ax_spec = fig.add_axes([left, bottom + h_1, w_1, h_2])
        ax_spec.semilogy(spec[itr], f_lin, plot_args[1])

        # add colorbars
        ax_cb_tfr = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom +
                                  h_1, w_cb, h_2])
        fig.colorbar(img_tfr, cax=ax_cb_tfr)

        # set limits
        ax_sig.set_ylim(st.min() * 1.1, st.max() * 1.1)
        ax_sig.set_xlim(t[0], t[-1])

        xlim = spec.max() * 1.1

        ax_spec.set_xlim(xlim, 0.)
        ax_spec.set_ylim(fmin, fmax)

        if clim == 0.:
            clim = _tfr.max()

        img_tfr.set_clim(0., clim)

        ax_sig.set_xlabel('time')
        ax_spec.set_ylabel('frequency')

        # remove axis labels
        ax_tfr.xaxis.set_major_formatter(NullFormatter())
        ax_tfr.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs


def spectrum(data, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6, mode='absolute', fft_zero_pad_fac=0):
    """
    Plot time frequency representation, spectrum and time series of the signal.

    :param data: signal, type numpy.ndarray with shape (number of components,
        number of time samples) or (number of timesamples, ) for single
        component data
    :param dt: time step between two samples in st
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param mode: 'absolute' for absolute value of TFR, 'power' for ``|TFR|^2``
    :param fft_zero_pad_fac: integer, if > 0, the signal is zero padded to
        ``nfft = next_pow_2(len(st)) * fft_zero_pad_fac`` to get smoother
        spectrum in the low frequencies (has no effect on the TFR and might
        make demeaning/tapering necessary to avoid artifacts)

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    .. rubric:: Example

    >>> from obspy import read
    >>> tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
    >>> spectrum(tr.data, dt=tr.stats.delta, fmin=.01,
    ...         fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)

    .. plot::

        from obspy.signal.tf_misfit import plot_tfr
        from obspy import read
        tr = read("https://examples.obspy.org/a02i.2008.240.mseed")[0]
        plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01,
                fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)
    """

    npts = data.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0

    if fft_zero_pad_fac == 0:
        nfft = npts
    else:
        nfft = util.next_pow_2(npts) * fft_zero_pad_fac

    f_lin = np.linspace(0, 0.5 / dt, nfft // 2 + 1)

    if len(data.shape) == 1:
        _w = np.zeros((1, nf, npts), dtype=complex)
        _w[0] = cwt(data, dt, w0, fmin, fmax, nf)
        ntr = 1

        spec = np.zeros((1, nfft // 2 + 1), dtype=complex)
        spec[0] = np.fft.rfft(data, n=nfft) * dt

        st = data.reshape((1, npts))
    else:
        print("WARNING: Only provide 1 component data")

    if mode == 'absolute':
        _tfr = np.abs(_w)
        spec = np.abs(spec)
    elif mode == 'power':
        _tfr = np.abs(_w) ** 2
        spec = np.abs(spec) ** 2
    else:
        raise ValueError('mode "' + mode + '" not defined!')

    return spec[0, :][1:], _tfr[0, :, :], f_lin[1:]


def spectrogram(tr, samp_rate=None, wlen=6.0, overlap=0.86, units="amplitude"):

    import numpy as np
    from scipy.signal import spectrogram
    from obspy import Stream
    from obspy.imaging.spectrogram import _nearest_pow_2

    # tr = Stream(tr)

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

    if units == "psd":
        Sxx = Sxx[1:, :]
    elif units == "db":
        Sxx = 10 * np.log10(Sxx[1:, :])
    elif units == "amplitude":
        Sxx = np.sqrt(Sxx[1:, :])
    frequencies = frequencies[1:]

    return Sxx, frequencies
