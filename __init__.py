"""
Written and maintained by Jay Wellik
USAID/USGS Volcano Disaster Assistance Program
@jwellik@usgs.gov

LAST UPDATED: 2023 August 9
"""

# Basic utilities
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



### TO-DO ###
# TODO vdapseisutils.velocitymodels read, write, plot