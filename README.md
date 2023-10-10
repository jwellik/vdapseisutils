<img src="https://github.com/jwellik/vdapseisutils/blob/main/img/vseis-logo.png" width=1510 alt="VDAP" />

## Overview
VDAPSEISUTILS is a set of (mostly) Python code that provides easy methods for common tasks in operational volcano seismology.

## Installation
If you are familiar with the usage of [REDPy](https://github.com/ahotovec/REDPy), you will be familiar with VDAPSEISUTILS.

Download the [zip file](https://github.com/jwellik/vdapseisutils/archive/main.zip) or use `git` to clone the entire repository to a working directory (e.g., mine is `/home/jwellik/PYTHON/vdapseisutils`).

VDAPSEISUTILS runs on Python 3.9. The suite of codes in this repository is comprehensive. Thus, many dependencies are required to run all of them.  
[numpy](http://www.numpy.org/) | [scipy](http://www.scipy.org/) | [matplotlib](http://www.matplotlib.org/) | [obspy](http://www.obspy.org/) | [pytables](http://www.pytables.org/) | [pandas](http://pandas.pydata.org/) | [bokeh](http://bokeh.pydata.org/) | [cartopy](http://scitools.org.uk/cartopy/)

These dependencies can be easily installed via [Anaconda](https://www.continuum.io/) on the command line. I *highly* recommend using a virtual environment so that your environment does not conflict with any other Python packages you may be using. This can be done with the following commands:
```
$ conda config --add channels conda-forge
$ conda create -n vseis399 python=3.9 obspy pandas cartopy pygmt bokeh
$ conda activate vseis399  # run this before you use VDAPSEISUTILS
$ cd /home/jwellik/PYTHON/vdapseisutils
$ pip install -e .
$ conda deactivate vseis399  # run this when you are done with VDAPSEISUTILS
```
The 'pip install -e' command installs the package in "editable" mode, which means that you can update it with a simple git pull in your local repository. This install command only needs to be run once.



## Usage
```
$ cd /home/jwellik/PYTHON/vdapseisutils
$ conda activate vseis399
$ python gallery/Plot_Map_of_Volcanic_Earthquakes.py  # Make sure your terminal can forward graphics
$ conda deactivate vseis399
```

