"""
Catalog package for volcano seismology workflows.

This package provides the VCatalog class and related functionality for working with
earthquake catalogs in volcano seismology applications.
"""

from .core import VCatalog, VEvent

# Export the main class for easy importing
__all__ = ['VCatalog', 'VEvent']

# Version info
__version__ = '1.0.0' 