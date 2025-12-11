"""
Written and maintained by Jay Wellik
USAID/USGS Volcano Disaster Assistance Program
@jwellik@usgs.gov

LAST UPDATED: 2023 August 9

* THIS SHOULD BE DELETED ONCE THIS BECOMES A PACKAGE
"""

# Basic utilities
import vdapseisutils
import numpy as np


__version__ = "vdapseisutils : beta 0.0.1"


def varinfo(var):

    vname = 'varname'
    vtype = type(var)
    if vtype == 'list':
        vshape = np.shape(var)
    if vtype == 'ndarray':
        vshape = var.shape()
    else:
        vshape = '?'

    print(':' * 30)
    print('{:10} ({:10}) ({:10})'.format(vname,vtype,vshape))
    print(var)
    print(':'*30)

"""
GOAL USAGE:

# CORE - Waveform Data Sources
from vdapseisutils import DataSource
from vdapseisutils import Stream
from vdapseisutils import NSLC

# CORE - Plotting (matplotlib)
from vdapseisutils import VolcanoMap
from vdapseisutils import CrossSection
from vdapseisutils import VelocityModel
from vdapseisutils import Catalog
from vdapseisutils import Helicorder
from vdapseisutils import Clipboard

# CORE - Plotting (bokeh)
from vdapseisutils.swarmbk import Clipboard_bk

# SANDBOX
from vdapseisutils.sandbox import Trigger


"""