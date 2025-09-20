"""
swarmmpl3: Enhanced time-series plotting for seismic data

A more structured and object-oriented approach to plotting ObsPy Stream and Trace objects
with waveforms and spectrograms.

Classes:
    TimeAxes: Time-aware axes wrapper with flexible tick formatting
    Panel: Collection of TimeAxes sharing a time axis (e.g., waveform + spectrogram)
    Clipboard: Collection of Panels for multi-trace plotting

Convenience functions:
    swarmw: Plot waveform only
    swarmg: Plot spectrogram only  
    swarmwg: Plot waveform + spectrogram panel
"""

from .timeaxes import TimeAxes
from .panel import Panel
from .clipboard import Clipboard
from .convenience import swarmw, swarmg, swarmwg, swarm_clipboard

__all__ = ['TimeAxes', 'Panel', 'Clipboard', 'swarmw', 'swarmg', 'swarmwg', 'swarm_clipboard']
