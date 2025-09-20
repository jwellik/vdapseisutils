"""
Convenience functions for quick plotting
"""

from obspy import Stream, Trace
from .timeaxes import TimeAxes
from .panel import Panel
from .clipboard import Clipboard


def swarmw(data, ax=None, figsize=(10, 4), tick_type="absolute", **kwargs):
    """
    Plot waveform only.
    
    Parameters:
    -----------
    data : obspy.Trace or obspy.Stream
        The data to plot. If Stream with multiple traces, creates Clipboard.
        If single Trace, creates TimeAxes.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on (only used for single Trace)
    figsize : tuple
        Figure size if creating new axes
    tick_type : str
        Type of tick formatting
    **kwargs : dict
        Additional arguments passed to plot_waveform()
        
    Returns:
    --------
    TimeAxes or Clipboard
        TimeAxes for single trace, Clipboard for multiple traces
    """
    if isinstance(data, Stream):
        # Multiple traces - create Clipboard with waveforms only
        return Clipboard(data, sync_waves=False, figsize=figsize,
                        tick_type=tick_type, wave_settings=kwargs, mode="w")
    elif isinstance(data, Trace):
        # Single trace - create TimeAxes
        timeaxes = TimeAxes(ax=ax, figsize=figsize, tick_type=tick_type)
        timeaxes.plot_waveform(data, **kwargs)
        return timeaxes
    else:
        raise TypeError("swarmw requires an ObsPy Trace or Stream object")


def swarmg(data, ax=None, figsize=(10, 4), tick_type="absolute", **kwargs):
    """
    Plot spectrogram only.
    
    Parameters:
    -----------
    data : obspy.Trace or obspy.Stream
        The data to plot. If Stream with multiple traces, creates Clipboard.
        If single Trace, creates TimeAxes.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on (only used for single Trace)
    figsize : tuple
        Figure size if creating new axes
    tick_type : str
        Type of tick formatting
    **kwargs : dict
        Additional arguments passed to plot_spectrogram()
        
    Returns:
    --------
    TimeAxes or Clipboard
        TimeAxes for single trace, Clipboard for multiple traces
    """
    if isinstance(data, Stream):
        # Multiple traces - create Clipboard with spectrograms only
        return Clipboard(data, sync_waves=False, figsize=figsize,
                        tick_type=tick_type, spec_settings=kwargs, mode="g")
    elif isinstance(data, Trace):
        # Single trace - create TimeAxes
        timeaxes = TimeAxes(ax=ax, figsize=figsize, tick_type=tick_type)
        timeaxes.plot_spectrogram(data, **kwargs)
        return timeaxes
    else:
        raise TypeError("swarmg requires an ObsPy Trace or Stream object")


def swarmwg(data, figsize=(10, 6), height_ratios=[1, 3], tick_type="absolute",
            wave_settings=None, spec_settings=None):
    """
    Plot waveform and spectrogram panel.
    
    Parameters:
    -----------
    data : obspy.Trace or obspy.Stream
        The data to plot. If Stream with multiple traces, creates Clipboard.
        If single Trace, creates Panel.
    figsize : tuple
        Figure size
    height_ratios : list
        Height ratios [waveform, spectrogram]
    tick_type : str
        Type of tick formatting
    wave_settings : dict, optional
        Settings for waveform plotting
    spec_settings : dict, optional
        Settings for spectrogram plotting
        
    Returns:
    --------
    Panel or Clipboard
        Panel for single trace, Clipboard for multiple traces
    """
    if isinstance(data, Stream):
        # Multiple traces - create Clipboard with both waveforms and spectrograms
        return Clipboard(data, sync_waves=False, figsize=figsize, 
                        tick_type=tick_type, wave_settings=wave_settings,
                        spec_settings=spec_settings, mode="wg")
    elif isinstance(data, Trace):
        # Single trace - create Panel with both waveforms and spectrograms
        return Panel.from_trace_waveform_spectrogram(
            data, height_ratios=height_ratios, figsize=figsize,
            tick_type=tick_type, wave_settings=wave_settings,
            spec_settings=spec_settings
        )
    else:
        raise TypeError("swarmwg requires an ObsPy Trace or Stream object")


# Additional convenience function for creating Clipboard directly
def swarm_clipboard(data, sync_waves=True, figsize=(10, 12), tick_type="absolute",
                   wave_settings=None, spec_settings=None, panel_spacing=0.02, mode="wg"):
    """
    Create a Clipboard with multiple traces.
    
    Parameters:
    -----------
    data : obspy.Stream or list of Traces
        The data to plot
    sync_waves : bool
        Whether to synchronize time axes across all panels
    figsize : tuple
        Figure size
    tick_type : str
        Type of tick formatting
    wave_settings : dict, optional
        Settings for waveform plotting
    spec_settings : dict, optional
        Settings for spectrogram plotting
    panel_spacing : float
        Spacing between panels
    mode : str
        What to plot: "w" (waveform), "g" (spectrogram), "wg" (both, default)
        
    Returns:
    --------
    Clipboard
        Clipboard containing all traces
    """
    return Clipboard(data, sync_waves=sync_waves, figsize=figsize,
                    tick_type=tick_type, wave_settings=wave_settings,
                    spec_settings=spec_settings, panel_spacing=panel_spacing, mode=mode)
