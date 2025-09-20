# swarmmpl3 - Enhanced Seismic Data Plotting

A modern, object-oriented approach to plotting ObsPy seismic data with waveforms and spectrograms.

## Quick Start

```python
from vdapseisutils.core.swarmmpl3 import swarmw, swarmg, swarmwg, Clipboard

# Simple waveform plot
p = swarmw(trace, tick_type="seconds")
p.axvline("2024/10/14T10:30:23", color="red")

# Simple spectrogram plot  
p = swarmg(trace, tick_type="minutes")
p.axvline(60, t_units="seconds", color="white")

# Waveform + spectrogram panel
p = swarmwg(trace, tick_type="seconds")
p.axvline("2024/10/14T10:30:23", axes=[0], color="red")  # waveform only
p.axvline(30, t_units="seconds", color="blue")  # both axes

# Multiple traces in clipboard
cb = Clipboard(stream, sync_waves=True, tick_type="minutes", panel_spacing=0.02)
cb = Clipboard(stream, mode="w")   # Waveforms only
cb = Clipboard(stream, mode="g")   # Spectrograms only  
cb = Clipboard(stream, mode="wg")  # Both waveforms and spectrograms (default)
cb.axvline("2024/10/14T10:30:23", panels=[0], color="red")  # first panel
cb.axvline("2024/10/14T10:30:23", panels=[0], axes=[1], color="blue")  # first panel, spectrogram

# Plot catalog events and picks
cb.plot_catalog(catalog, verbose=True)                    # All panels, all picks/origins
cb.plot_catalog(event, verbose=True)                      # Single event (auto-converted to catalog)
cb.plot_catalog(catalog, stations=["ANMO"], axes=[1])     # ANMO spectrogram only

# Multicomponent plotting (Z, N, E components)
cb = swarmw(z_stream)                                      # Plot Z components
cb.plot_trace(n_stream, color="red", alpha=0.6, zorder=-1) # Add N components underneath
cb.plot_trace(e_stream, color="blue", alpha=0.6, zorder=-1) # Add E components underneath
cb.plot_horizontals(full_stream)                          # Convenience method for N/E components
cb.plot_catalog(catalog, plot_origins=False, p_color="orange")  # Picks only, custom colors
```

## Architecture

- **TimeAxes**: Time-aware axes wrapper with flexible tick formatting
- **Panel**: Collection of TimeAxes sharing a time axis (e.g., waveform + spectrogram)  
- **Clipboard**: Collection of Panels for multi-trace plotting

## Features

- ✅ **Flexible time formatting**: absolute, relative, seconds, minutes, hours, days
- ✅ **ObsPy integration**: UTCDateTime parsing for time inputs
- ✅ **Shared time axes**: Automatic synchronization within panels (always)
- ✅ **Multi-trace support**: Clipboard with optional time synchronization
- ✅ **Targeted operations**: axvline() can target specific axes/panels
- ✅ **Convenience functions**: swarmw(), swarmg(), swarmwg()
- ✅ **Catalog integration**: plot_catalog() with metadata-based targeting
- ✅ **Flexible modes**: Waveforms only, spectrograms only, or both
- ✅ **Multicomponent support**: plot_trace() and plot_horizontals() for Z/N/E plotting

## Testing

Run the comprehensive test suite:

```python
from vdapseisutils.core.swarmmpl3.test import run_all_tests
run_all_tests()
```

Or test individual components:

```python
from vdapseisutils.core.swarmmpl3.test import test_single_plots, test_panel_plots
test_single_plots()
test_panel_plots()
```

The test suite downloads real seismic data from IRIS and demonstrates all major functionality.

## Tick Types

- `"absolute"`: Standard datetime labels (uses ObsPy formatting)
- `"relative"` or `"seconds"`: Relative seconds from start
- `"minutes"`: Relative minutes from start  
- `"hours"`: Relative hours from start
- `"days"`: Relative days from start

## Clipboard Modes

- `mode="w"`: Waveforms only - faster rendering, good for amplitude analysis (trace IDs on y-axis)
- `mode="g"`: Spectrograms only - focus on frequency content (trace IDs on y-axis)
- `mode="wg"`: Both waveforms and spectrograms (default) - complete view (trace IDs on spectrogram y-axis)

## Multicomponent Plotting

Plot multiple seismic components (Z, N, E) together:

```python
# Manual control - plot Z components, then add N/E underneath
cb = swarmw(z_stream)
cb.plot_trace(n_stream, color="red", alpha=0.6, zorder=-1)
cb.plot_trace(e_stream, color="blue", alpha=0.6, zorder=-1)

# Convenience method - automatically finds and plots N/E components on matching stations
cb = swarmw(z_stream)
cb.plot_horizontals(full_stream, color="gray", alpha=0.7)  # Only plots N/E on stations with Z

# Target specific stations or axes
cb.plot_trace(n_stream, stations=["ANMO"], axes=[0])  # N on ANMO waveforms only
cb.plot_horizontals(stream, axes=[0])                 # N/E on waveforms only (not spectrograms)
```

## Time Synchronization

- **Within Panel**: All TimeAxes in a Panel are always synchronized to the original data extent
- **Between Panels**: Only synchronized when `sync_waves=True` in Clipboard
- **Manual Control**: Use `set_xlim()` to override automatic synchronization

## Examples

See `test.py` for comprehensive examples with real seismic data.
