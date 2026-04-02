"""
Convenience functions for quick plotting (v3 panel / TimeAxes stack).
"""

from obspy import Stream, Trace

from .clipboard import SwarmClipboard
from .panel import Panel
from .timeaxes import TimeAxes


def swarmw(data, ax=None, figsize=(10, 4), tick_type="absolute", **kwargs):
    """
    Plot waveform only.

    Parameters
    ----------
    data : obspy.Trace or obspy.Stream
        The data to plot. If Stream with multiple traces, creates SwarmClipboard.
        If single Trace, creates TimeAxes.
    """

    if isinstance(data, Stream):
        return SwarmClipboard(
            data,
            sync_waves=False,
            figsize=figsize,
            tick_type=tick_type,
            wave_settings=kwargs,
            mode="w",
        )
    if isinstance(data, Trace):
        timeaxes = TimeAxes(ax=ax, figsize=figsize, tick_type=tick_type)
        timeaxes.plot_waveform(data, **kwargs)
        return timeaxes
    raise TypeError("swarmw requires an ObsPy Trace or Stream object")


def swarmg(data, ax=None, figsize=(10, 4), tick_type="absolute", **kwargs):
    """Plot spectrogram only (Stream -> SwarmClipboard; Trace -> TimeAxes)."""

    if isinstance(data, Stream):
        return SwarmClipboard(
            data,
            sync_waves=False,
            figsize=figsize,
            tick_type=tick_type,
            spec_settings=kwargs,
            mode="g",
        )
    if isinstance(data, Trace):
        timeaxes = TimeAxes(ax=ax, figsize=figsize, tick_type=tick_type)
        timeaxes.plot_spectrogram(data, **kwargs)
        return timeaxes
    raise TypeError("swarmg requires an ObsPy Trace or Stream object")


def swarmwg(
    data,
    figsize=(10, 6),
    height_ratios=None,
    tick_type="absolute",
    wave_settings=None,
    spec_settings=None,
):
    """Plot waveform and spectrogram panel (Stream -> SwarmClipboard; Trace -> Panel)."""

    if height_ratios is None:
        height_ratios = [1, 3]

    if isinstance(data, Stream):
        return SwarmClipboard(
            data,
            sync_waves=False,
            figsize=figsize,
            tick_type=tick_type,
            wave_settings=wave_settings,
            spec_settings=spec_settings,
            mode="wg",
        )
    if isinstance(data, Trace):
        return Panel.from_trace_waveform_spectrogram(
            data,
            height_ratios=height_ratios,
            figsize=figsize,
            tick_type=tick_type,
            wave_settings=wave_settings,
            spec_settings=spec_settings,
        )
    raise TypeError("swarmwg requires an ObsPy Trace or Stream object")


def swarm_clipboard(
    data,
    sync_waves=True,
    figsize=(10, 12),
    tick_type="absolute",
    wave_settings=None,
    spec_settings=None,
    panel_spacing=0.02,
    mode="wg",
):
    """Create a SwarmClipboard with multiple traces."""

    return SwarmClipboard(
        data,
        sync_waves=sync_waves,
        figsize=figsize,
        tick_type=tick_type,
        wave_settings=wave_settings,
        spec_settings=spec_settings,
        panel_spacing=panel_spacing,
        mode=mode,
    )


__all__ = [
    "swarm_clipboard",
    "swarmg",
    "swarmw",
    "swarmwg",
]
