"""Batch 4 Part E: shared waveform/spectrogram compute."""

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.imaging.spectrogram import _nearest_pow_2
from scipy.signal import spectrogram

from vdapseisutils.compute.waveforms import compute_spectrogram, prepare_waveform_series


def _synthetic_trace(npts=4096, fs=100.0):
    t0 = UTCDateTime("2020-01-01T00:00:00")
    data = np.random.default_rng(42).standard_normal(npts).astype(np.float64)
    return Trace(
        data=data,
        header={
            "starttime": t0,
            "sampling_rate": fs,
            "npts": npts,
        },
    )


def test_prepare_waveform_series_matches_trace_times():
    tr = _synthetic_trace()
    ser = prepare_waveform_series(tr, relative_offset_s=1.5)
    np.testing.assert_allclose(ser.times_s, tr.times() + 1.5)
    np.testing.assert_array_equal(ser.amplitudes, tr.data)


def test_compute_spectrogram_matches_legacy_scipy_path():
    tr = _synthetic_trace()
    wlen = 2.0
    overlap = 0.86
    dbscale = True

    spec = compute_spectrogram(tr, samp_rate=None, wlen=wlen, overlap=overlap, dbscale=dbscale)

    fs = float(tr.stats.sampling_rate)
    signal = np.asarray(tr.data, dtype=float)
    nfft = int(_nearest_pow_2(wlen * fs))
    nlap = int(nfft * float(overlap))
    signal_dm = signal - signal.mean()
    f_exp, t_exp, s_exp = spectrogram(
        signal_dm, fs=fs, nperseg=nfft, noverlap=nlap, scaling="spectrum"
    )
    if dbscale:
        s_exp = 10 * np.log10(s_exp[1:, :])
    else:
        s_exp = np.sqrt(s_exp[1:, :])
    f_exp = f_exp[1:]

    np.testing.assert_allclose(spec.frequencies_hz, f_exp)
    np.testing.assert_allclose(spec.times_s, t_exp)
    np.testing.assert_allclose(spec.power, s_exp)
    assert spec.sampling_rate_hz == pytest.approx(fs)
