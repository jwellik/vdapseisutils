#import setuptools
#
#setuptools.setup(name='vdapseisutils')

from setuptools import setup, find_packages

setup(
    name="vdapseisutils",
    version="0.1.0",
    description="Advanced utilities for volcano seismology",
    author="Jay Wellik",
    author_email="jwellik@usgs.gov",
    url="https://github.com/jwellik/vdapseisutils",
    packages=find_packages(),  # Automatically finds all submodules
    install_requires=[
        "obspy",              # solid codebase for seismology
        "cartopy",            # mapping utilities
        "pygmt",              # mainly used for downloading terrain data
        "geopy",              # gets lat,lon from address or geopolitical string
        "pandas",             # standard Python pkg for tables and dataframes (comes with obspy?)
        "matplotlib==3.9.0",  # I had issues w 3.10.0, which would be installed by default
        "bokeh",              # interactive plotting
        "timezonefinder",     # helps with timezone management
        "unidecode",          # misc pkg.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)

