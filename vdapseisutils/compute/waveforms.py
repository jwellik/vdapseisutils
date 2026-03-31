"""
Backend-neutral waveform and spectrogram compute (SciPy STFT).

Plotting code should call these helpers then map times to datetime or relative axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.imaging.spectrogram import _nearest_pow_2
from scipy.signal import spectrogram


@dataclass
class WaveformSeriesResult:
    """Relative times (seconds), amplitudes, and trace timing metadata."""

    times_s: np.ndarray
    amplitudes: np.ndarray
    starttime: UTCDateTime
    sampling_rate: float


def prepare_waveform_series(tr: Trace, *, relative_offset_s: float = 0.0) -> WaveformSeriesResult:
    """
    Build aligned time (seconds from trace start + offset) and amplitude arrays.

    Does not modify the input trace.
    """
    t_rel = np.asarray(tr.times(), dtype=float) + float(relative_offset_s)
    return WaveformSeriesResult(
        times_s=t_rel,
        amplitudes=np.asarray(tr.data, dtype=float),
        starttime=tr.stats.starttime,
        sampling_rate=float(tr.stats.sampling_rate),
    )


@dataclass
class SpectrogramResult:
    """Spectrogram grid from :func:`scipy.signal.spectrogram` (spectrum scaling)."""

    frequencies_hz: np.ndarray
    times_s: np.ndarray
    power: np.ndarray
    sampling_rate_hz: float


def compute_spectrogram(
    tr: Trace,
    *,
    samp_rate: float | None = None,
    wlen: float = 2.0,
    overlap: float = 0.86,
    dbscale: bool = True,
) -> SpectrogramResult:
    """
    Compute spectrogram power on a copy of the trace (optionally resampled).

    Matches the numerical path historically used in ``swarmmpl.clipboard.plot_spectrogram``:
    demean, ``_nearest_pow_2`` window, spectrum scaling, drop DC row, then dB or sqrt.
    """
    tr_work = tr.copy()
    if samp_rate is not None:
        tr_work.resample(float(samp_rate))
    fs = float(tr_work.stats.sampling_rate)
    signal = np.asarray(tr_work.data, dtype=float)
    wlen_use = wlen if wlen else 128 / fs
    npts = len(signal)
    nfft = int(_nearest_pow_2(wlen_use * fs))
    if npts < nfft:
        raise ValueError(
            f"Input signal too short ({npts} samples, window length {wlen_use} s, "
            f"nfft {nfft} samples, sampling rate {fs} Hz)"
        )
    nlap = int(nfft * float(overlap))
    signal = signal - signal.mean()
    frequencies, times, Sxx = spectrogram(
        signal, fs=fs, nperseg=nfft, noverlap=nlap, scaling="spectrum"
    )
    if dbscale:
        Sxx = 10 * np.log10(Sxx[1:, :])
    else:
        Sxx = np.sqrt(Sxx[1:, :])
    frequencies = frequencies[1:]
    return SpectrogramResult(
        frequencies_hz=frequencies,
        times_s=times,
        power=Sxx,
        sampling_rate_hz=fs,
    )
