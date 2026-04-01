<img src="https://github.com/jwellik/vdapseisutils/blob/main/img/vseis-logo.png" width=1510 alt="VDAP" />

## Overview
VDAPSEISUTILS is a set of (mostly) Python code that provides easy methods for common tasks in operational volcano seismology.

At the moment, core tasks include:
- VolcanoMap: Plot a basic map and cross section of earthquakes around a volcano.
- ObsPy Catalog & Inventory IO: Import/Export ObsPy Catalog & Inventory formats from Swarm, Earthworm, and NonLinLoc.

Sandbox tasks:
(These routines are available, but I may change them significantly before they are stored in core. I am still working on them.)
- Velocity: Load, save, and plot velocity models.
- SwarmMPL: MatPlotLib routines for [Swarm](https://volcanoes.usgs.gov/software/swarm/index.shtml)-like plots (Helicorders, waveform traces, spectrograms, spectra).

Pending tasks:
- CCMatrix: Create, save, load, and plot cross correlation matrices.
- Waveform statistics: E.g., compute Frequency Index for a list of Stream objects and compare results across events.
- DataSource: A wrapper for ObsPy Clients with a more universal usage (automatically determines Client type).

## Installation

This package is not yet on PyPI. Clone the repo and use [uv](https://docs.astral.sh/uv/) to create the environment and install the package (Python 3.11+).

**One-time setup**

This project uses a named environment **vseis311**. Set the env path, then sync:

```bash
git clone https://github.com/jwellik/vdapseisutils.git
cd vdapseisutils
export UV_PROJECT_ENVIRONMENT=vseis311
uv sync
```

To make that permanent in this repo (optional), use [direnv](https://direnv.net/): run `direnv allow` once; the project’s `.envrc` will set `UV_PROJECT_ENVIRONMENT=vseis311` whenever you `cd` here.

**Activate and run Python**

```bash
source vseis311/bin/activate
python
```

You activate by sourcing the *script inside* the environment directory (`vseis311/bin/activate`). There is no global `activate` command that takes a name—that’s Conda’s interface. With Python’s venv (and uv), the environment is just a folder; you always run `source <env-dir>/bin/activate` (e.g. `source vseis311/bin/activate` or `source .venv/bin/activate` if you didn’t set a custom name).

Optional: run without activating: `uv run python ...`, `uv run pytest`, etc.

**Optional: PyTables (HDF5)**  
The package does not require `tables` (PyTables) by default. If you need it, install the system HDF5 library first, then the extra:

- **macOS (Homebrew):** `brew install hdf5`, then `export HDF5_DIR=$(brew --prefix hdf5)` and `pip install ".[tables]"` (or `uv sync --extra tables`).
- **Conda:** `conda install hdf5` then `pip install ".[tables]"`.

## Usage

This package is still in development. If you have trouble with these codes, let me know.

```bash
source vseis311/bin/activate
cd gallery
python
>>> from vdapseisutils.gallery import Mapping_tutorial
>>> Mapping_tutorial.main()   # graphics forwarding if remote
```

This runs a script that reads .arc files from Wy'East/Mt Hood, Oregon and plots them on a map and cross sections. See the [Gallery](https://github.com/jwellik/vdapseisutils/tree/main/gallery) for more examples and usage.

### Maps imports (API v1)

Map classes are implemented under `vdapseisutils.core.maps` (e.g. `map.py`, `volcano_figure.py`). The module `vdapseisutils.core.maps.maps` is a **compatibility shim** that re-exports those symbols; existing imports such as `from vdapseisutils.core.maps.maps import Map` remain valid. Prefer `from vdapseisutils.core.maps import Map` in new code. Backend-neutral compute will grow under `vdapseisutils.compute` (see the package docstring there).

**Catalog points and origins:** Use `vdapseisutils.compute.catalog.prepare_catalog_points` (or `prepare_catalog_points_from_time_format` for legacy `time_format` strings) for matplotlib-free tables of hypocenter data. `vdapseisutils.utils.obspyutils.catalog.origin.get_primary_origin` selects the QuakeML preferred origin when present, otherwise the first origin—the same rule used for VCatalog event-rate extraction, scatter maps, and `catalog2txyzm` by default. Pass `origin_policy="last"` to `catalog2txyzm` if you need the old “last origin in the list” behavior.

**Maps matplotlib style (no import-time rc):** Pure style dicts live in `vdapseisutils.core.maps.defaults_constants`. Importing that module does not load Matplotlib or change `rcParams`. To apply the maps theme globally, call `vdapseisutils.core.maps.defaults.register_maps_mpl_style()` once (e.g. at app startup). Map constructors (`Map`, `CrossSection`, `VolcanoFigure`, `TimeSeries`) call `ensure_maps_mpl_style()` so figures still get the intended defaults when those classes are used without a prior explicit registration.

**Waveform and spectrogram compute:** Shared, backend-neutral helpers live in `vdapseisutils.compute.waveforms`: `prepare_waveform_series` returns a `WaveformSeriesResult` (time axis and data); `compute_spectrogram` returns a `SpectrogramResult` (frequency, time offset, power) using the SciPy spectrogram path aligned with Swarm-style plots. `swarmmpl` and `swarmmpl3` plotting code calls these helpers then draws; use the same APIs if you need arrays without Matplotlib.

### Pyplot helpers (`register_pyplot`)

You can attach VDAPSeisUtils figure constructors to Matplotlib’s pyplot module (API v1 section 3 in `.local/api-v1-coord/API_V1_CANONICAL.md`):

```python
import matplotlib.pyplot as plt
import vdapseisutils.plot.mpl as vsmpl

vsmpl.register_pyplot()
fig = plt.helicorder(...)   # or plt.clipboard / plt.eqmap / plt.volcano / plt.swarm
```

After `register_pyplot()`, those names construct the corresponding project types (`Helicorder`, clipboard figure, `Map`, `VolcanoFigure`; `plt.swarm` is an interim alias to the clipboard backend—see `vdapseisutils.plot.mpl`).

