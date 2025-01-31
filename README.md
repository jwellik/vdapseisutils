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
This package is not quite ready yet for installation with pip. Instead, download the repository and add it to your path before running other Python code.

Download the [zip file](https://github.com/jwellik/vdapseisutils/archive/main.zip) or use `git` to clone the entire repository to a working directory (e.g., mine is `/home/jwellik/PYTHON/vdapseisutils`).

VDAPSEISUTILS runs on Python 3.12. The suite of codes in this repository is comprehensive. Thus, many dependencies are required to run all of them. Common packages include: 

[numpy](http://www.numpy.org/) | [scipy](http://www.scipy.org/) | [matplotlib](http://www.matplotlib.org/) | [obspy](http://www.obspy.org/) | [pytables](http://www.pytables.org/) | [pandas](http://pandas.pydata.org/) | [bokeh](http://bokeh.pydata.org/) | [cartopy](http://scitools.org.uk/cartopy/) | [timezonefinder](https://pypi.org/project/timezonefinder/)

Other git repositories are also installed as dependencies:
- Claudio Satriano's [nllgrid](https://github.com/claudiodsf/nllgrid)

All of these dependencies can be easily installed via [Anaconda](https://www.continuum.io/) on the command line. I *highly* recommend using a virtual environment so that your environment does not conflict with any other Python packages you may be using. This can be done with the following commands:
```
$ conda config --add channels conda-forge
$ conda create -n seismology312 python=3.12 obspy pandas cartopy pygmt bokeh timezonefinder
$ conda install -c conda-forge nllgrid
```

More packages to use for seismology312:
```angular2html
seismology312:

conda install -c conda-forge numpy scipy pandas pytables scikit-learn
conda install -c conda-forge matplotlib bokeh
conda install -c conda-forge obspy pygmt cartopy
conda install -c conda-forge timezonefinder
```

## Usage
This package is still in development. If you have trouble with these codes, let me know.
```
$ cd /home/jwellik/PYTHON
$ mv vdapseisutils-main vdapseisutils
$ cd /home/jwellik/PYTHON/vdapseisutils/gallery
$ conda activate seismology312
$ python
>>> import sys
>>> sys.path.append("/home/jwellik/PYTHON")  # Add all codes in the repository to your path
>>> from vdapseisutils.gallery import Mapping_tutorial.py
>>> Mapping_tutorial.main()  # Make sure your terminal has graphics forwarding
```
This will run a script that reads .arc files from Wy'East/Mt Hood, Oregon and plots them on a map and cross sections. Look at the [Gallery](https://github.com/jwellik/vdapseisutils/tree/main/gallery) for more examlpes and detailed usage.

