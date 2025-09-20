"""
Core VCatalog class with mixin functionality.

This module contains the main VCatalog class that extends ObsPy's Catalog
with volcano seismology specific functionality through mixins.
"""

from obspy import Catalog
from obspy.core.event import Event
from .plotting import VCatalogPlottingMixin
from .analysis import VCatalogAnalysisMixin
from .conversion import VCatalogConversionMixin
from .comparison import VCatalogComparisonMixin
from .utils import VCatalogUtilsMixin
from .pickqc import VCatalogPickQCMixin


class VEvent(Event):
    """
    Extended Event class that includes VCatalog conversion methods.
    
    This class wraps a single Event and provides access to VCatalog methods
    like to_picklog(), to_txyzm(), etc.
    """
    
    def __init__(self, event=None, **kwargs):
        if event is not None:
            # Copy all attributes from the original event
            super().__init__()
            for attr in dir(event):
                if not attr.startswith('_') and hasattr(event, attr):
                    try:
                        setattr(self, attr, getattr(event, attr))
                    except:
                        pass  # Skip attributes that can't be set
        else:
            super().__init__(**kwargs)
    
    # Conversion methods
    def to_picklog(self, verbose=False):
        """Convert event picks to a DataFrame (picklog format)."""
        temp_catalog = VCatalog([self])
        return temp_catalog.to_picklog(verbose=verbose)
    
    def to_txyzm(self, **kwargs):
        """Convert event to time/lat/lon/depth/mag format."""
        return VCatalogConversionMixin.to_txyzm(self, **kwargs)
    
    def to_basics(self, **kwargs):
        """Alias for to_txyzm()."""
        return self.to_txyzm(**kwargs)
    
    # Utility methods
    def get_eventid(self, fallback_prefix="id", verbose=False):
        """Extract event ID from this Event object."""
        return VCatalogUtilsMixin.get_eventid(self, self, fallback_prefix, verbose)
    
    def extract_event_id(self, fallback_prefix="id", verbose=False):
        """Alias for get_eventid()."""
        return self.get_eventid(fallback_prefix, verbose)
    
    def get_waveforms(self, client, **kwargs):
        """Get waveforms for this single event."""
        # Create temporary catalog with just this event
        temp_catalog = VCatalog([self])
        return temp_catalog.get_waveforms(client, **kwargs)
    
    def sort_picks(self, inplace=True, verbose=False):
        """
        Sort picks by earliest arrival time for this event.
        
        Parameters
        ----------
        inplace : bool, default True
            If True, modify the event in place. If False, return a new VEvent.
        verbose : bool, default False
            If True, print detailed information about the sorting process
            
        Returns
        -------
        VEvent or None
            If inplace=False, returns a new VEvent with sorted picks.
            If inplace=True, returns None and modifies the current event.
        """
        from copy import deepcopy
        
        # Work on copy if not inplace
        if inplace:
            target_event = self
        else:
            target_event = deepcopy(self)
        
        if not target_event.picks:
            if verbose:
                print("Event has no picks to sort")
            if not inplace:
                return target_event
            return
        
        pick_count = len(target_event.picks)
        
        if verbose:
            print(f">>> VEvent.sort_picks()")
            print(f"Sorting {pick_count} picks")
        
        # Filter out picks without time information
        picks_with_time = []
        picks_without_time = []
        
        for pick in target_event.picks:
            if hasattr(pick, 'time') and pick.time is not None:
                picks_with_time.append(pick)
            else:
                picks_without_time.append(pick)
                if verbose:
                    station = "unknown"
                    try:
                        if hasattr(pick, 'waveform_id') and pick.waveform_id:
                            parts = pick.waveform_id.id.split('.')
                            if len(parts) >= 2:
                                station = parts[1]
                    except:
                        pass
                    print(f"  WARNING: Pick at station {station} has no time, placing at end")
        
        # Sort picks with time by arrival time
        if picks_with_time:
            picks_with_time.sort(key=lambda pick: pick.time)
            
            if verbose:
                earliest_time = picks_with_time[0].time
                latest_time = picks_with_time[-1].time
                time_span = latest_time - earliest_time
                print(f"  Time range: {earliest_time} to {latest_time} (span: {time_span:.2f}s)")
        
        # Combine sorted picks with time + picks without time at the end
        sorted_picks = picks_with_time + picks_without_time
        target_event.picks = sorted_picks
        
        # If this VEvent is linked to a catalog, also update the original event
        if inplace and hasattr(target_event, '_original_event') and hasattr(target_event, '_parent_catalog'):
            target_event._original_event.picks = sorted_picks
        
        if verbose:
            print(f"  Sorted {len(picks_with_time)} picks by time, {len(picks_without_time)} without time placed at end")
        
        if not inplace:
            return target_event


class VCatalog(VCatalogConversionMixin, VCatalogPlottingMixin, VCatalogAnalysisMixin, 
               VCatalogComparisonMixin, VCatalogUtilsMixin, VCatalogPickQCMixin, Catalog):
    """Extended Catalog class for volcano seismology workflows."""

    def __init__(self, events=None, resource_id=None, description=None,
                 comments=None, creation_info=None, catalog=None):
        """
        Initialize VCatalog.

        Parameters:
        -----------
        events : list, optional
            List of Event objects (standard ObsPy way)
        resource_id : ResourceIdentifier, optional
            Resource identifier (standard ObsPy way)
        description : str, optional
            Description (standard ObsPy way)
        comments : list, optional
            Comments (standard ObsPy way)
        creation_info : CreationInfo, optional
            Creation info (standard ObsPy way)
        catalog : obspy.Catalog, optional
            Existing Catalog object to convert to VCatalog
        """
        # Handle case where first argument is a Catalog object
        if hasattr(events, 'events') and hasattr(events, 'resource_id'):
            # events is actually a Catalog object
            catalog_obj = events
            super().__init__(
                events=catalog_obj.events,
                resource_id=catalog_obj.resource_id,
                description=getattr(catalog_obj, 'description', None),
                comments=getattr(catalog_obj, 'comments', None),
                creation_info=getattr(catalog_obj, 'creation_info', None)
            )
        elif catalog is not None:
            # Initialize from catalog parameter
            super().__init__(
                events=catalog.events,
                resource_id=catalog.resource_id,
                description=getattr(catalog, 'description', None),
                comments=getattr(catalog, 'comments', None),
                creation_info=getattr(catalog, 'creation_info', None)
            )
        else:
            # Standard initialization
            super().__init__(
                events=events,
                resource_id=resource_id,
                description=description,
                comments=comments,
                creation_info=creation_info
            )

    def __getitem__(self, index):
        """
        Override indexing to return VEvent objects instead of plain Event objects.
        
        This returns VEvent objects for single indices, VCatalog for slices.
        The returned objects maintain references to the original events for modification.
        """
        if isinstance(index, slice):
            # For slice operations, return a new VCatalog that references the same events
            sliced_catalog = super().__getitem__(index)
            new_catalog = VCatalog()
            # Directly add the same event objects (not copies) to maintain references
            for event in sliced_catalog.events:
                new_catalog.events.append(event)  # Direct reference, not copy
            return new_catalog
        else:
            # For single index, return a VEvent that wraps the original event
            # but maintains references to the original event's attributes
            original_event = super().__getitem__(index)
            
            # Create a VEvent that acts as a proxy to the original event
            vevent = VEvent()
            
            # Instead of copying attributes, we'll make the VEvent reference the original
            # This is a bit tricky - we need to make sure modifications go to the original
            vevent._original_event = original_event
            vevent._catalog_index = index
            vevent._parent_catalog = self
            
            # Copy all attributes from original event to VEvent
            for attr_name in dir(original_event):
                if not attr_name.startswith('_') and hasattr(original_event, attr_name):
                    try:
                        attr_value = getattr(original_event, attr_name)
                        if not callable(attr_value):  # Don't copy methods
                            setattr(vevent, attr_name, attr_value)
                    except:
                        pass
            
            return vevent

    def write(self, filename, format=None, **kwargs):
        """
        Override the default write method to handle NLLOC_OBS format specially.
        
        For NLLOC_OBS format, this method writes one file per event to the specified directory.
        For all other formats, it delegates to the parent class write method.
        
        Parameters
        ----------
        filename : str
            For NLLOC_OBS format: Directory path where files should be written
            For other formats: File path as usual
        format : str, optional
            Output format. If "NLLOC_OBS", writes one file per event.
        **kwargs
            Additional arguments passed to the parent write method
        """
        print(f"DEBUG: VCatalog.write() called with filename={filename}, format={format}")
        
        if format == "NLLOC_OBS":
            print("DEBUG: Using VCatalog.write_nlloc_obs() method")
            return self.write_nlloc_obs(filename, **kwargs)
        else:
            print("DEBUG: Delegating to parent class write() method")
            # Delegate to parent class for all other formats
            return super().write(filename, format=format, **kwargs)


# Example usage:
#
# # Method 1: Standard ObsPy way
# vcat = VCatalog(events=[event1, event2], description="My catalog")
#
# # Method 2: From existing catalog (simple form)
# regular_cat = client.get_events(...)
# vcat = VCatalog(regular_cat)  # Clean conversion!
#
# # Method 3: From existing catalog (explicit parameter)
# vcat = VCatalog(catalog=regular_cat)
#
# # Instance methods (work on VCatalog instances)
# vcat.plot_eventrate(freq="1D")
# vcat.print_eventrate(freq="1H", top_n=10)
#
# # Class methods (work on any catalog)
# VCatalog.plot_eventrate_from_catalog(regular_cat, freq="1D")
# VCatalog.print_eventrate_from_catalog(regular_cat, freq="1H", top_n=10)
#
# # Catalog comparison methods:
# # Class method - compare two catalogs
# matching_cat, idx1, idx2 = VCatalog.compare_catalogs(cat1, cat2, threshold_seconds=5)
#
# # Instance method - unified compare() that handles both catalogs and times
# matching_cat, idx_self, idx_other = vcat.compare(other_catalog, threshold_seconds=5)  # with catalog
# matching_cat, idx_self, idx_times = vcat.compare(times, threshold_seconds=5)         # with times
#
# # Utility method - compare times between two lists
# matches = VCatalog.compare_times(times1, times2, threshold_seconds=5)

# TODO get_event_waveforms - Return a list of Streams, 1 for each event 