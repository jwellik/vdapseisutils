"""
Bokeh-backed map figures (Web Mercator).

Import the figure class as::

    from vdapseisutils.core.maps.bokeh import Map

This shadows only the submodule namespace ``vdapseisutils.core.maps.bokeh``;
the third-party ``bokeh`` package remains available as top-level ``import bokeh``.
"""

from __future__ import annotations

from .cross_section import CrossSection
from .map import Map

__all__ = ["Map", "CrossSection"]
