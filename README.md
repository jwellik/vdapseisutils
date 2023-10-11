<img src="https://github.com/jwellik/vdapseisutils/blob/main/img/vseis-logo.png" width=1510 alt="VDAP" />

## Overview
VDAPSEISUTILS is a set of (mostly) Python code that provides easy methods for common tasks in operational volcano seismology.

At the moment, core tasks include:
- VolcanoMap: Plot a basic map and cross section of earthquakes around a volcano.
- ObsPy Catalog & Inventory IO: Import/Export ObsPy Catalog & Inventory formats from Swarm, Earthworm, and NonLinLoc

Sandbox tasks:
(These routines are available, but I may change them significantly before they are stored in core. I am still working on them.)
- Velocity: Load, save, and plot velocity models.

Pending tasks:
- CCMatrix: Create, save, load, and plot cross correlation matrices.
- Waveform statistics: E.g., compute Frequency Index for a Catalog of events and compare results across events

## Installation
This package is not quite ready yet for installation with pip. Instead, download the repository and add it to your path before running other Python code.

Download the [zip file](https://github.com/jwellik/vdapseisutils/archive/main.zip) or use `git` to clone the entire repository to a working directory (e.g., mine is `/home/jwellik/PYTHON/vdapseisutils`).

VDAPSEISUTILS runs on Python 3.9. The suite of codes in this repository is comprehensive. Thus, many dependencies are required to run all of them.  
[numpy](http://www.numpy.org/) | [scipy](http://www.scipy.org/) | [matplotlib](http://www.matplotlib.org/) | [obspy](http://www.obspy.org/) | [pytables](http://www.pytables.org/) | [pandas](http://pandas.pydata.org/) | [bokeh](http://bokeh.pydata.org/) | [cartopy](http://scitools.org.uk/cartopy/)

These dependencies can be easily installed via [Anaconda](https://www.continuum.io/) on the command line. I *highly* recommend using a virtual environment so that your environment does not conflict with any other Python packages you may be using. This can be done with the following commands:
```
$ conda config --add channels conda-forge
$ conda create -n vseis399 python=3.9 obspy pandas cartopy pygmt bokeh
```

Example:
```
$ cd /home/jwellik/PYTHON
$ mv vdapseisutils-main vdapseisutils
$ cd /home/jwellik/PYTHON/vdapseisutils/gallery
$ conda activate vseis399
$ python
>>> import os
>>> os.addpath("/home/jwellik/PYTHON")  # Add all codes in the repository to your path
>>> from vdapseisutils.gallery import VolcanoMap_Hood_earthquakes
>>> VolcanoMap_VolcanoMap-Hood_earthquakes.main()
```
This will run a script that reads .arc files from Wy'East/Mt Hood, Oregon and plots them on a map and cross sections.

## Usage
```
$ cd /home/jwellik/PYTHON/vdapseisutils
$ conda activate vseis399
$ python gallery/Plot_Map_of_Volcanic_Earthquakes.py  # Make sure your terminal can forward graphics
$ conda deactivate vseis399
```

