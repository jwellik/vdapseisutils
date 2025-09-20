"""
VInventory module for volcano seismology workflows.

This module provides the VInventory class which extends ObsPy's Inventory
with additional functionality for volcano seismology data processing.
"""

from .core import VInventory

__all__ = ['VInventory']
